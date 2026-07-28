#!/usr/bin/env python3
"""
Dual-Arm MPC Controller for ROS2.

This node implements a Model Predictive Controller for dual-arm robot manipulation
using cuRobo's MPC solver. It serves as a drop-in replacement for the
dual_arm_safe_controller, providing trajectory optimization and collision avoidance.

Key Features:
- Real-time MPC optimization at 100Hz
- Task-space control (Cartesian pose tracking)
- Collision avoidance with scene obstacles
- CSV data logging for analysis

Topics (Subscribers):
    /dualarm_teleop_cmd (Float64MultiArray): Teleoperation commands
    /joint_states (JointState): Current joint states

Topics (Publishers):
    /mj_left/joints_desired (JointState): Left arm joint commands
    /mj_right/joints_desired (JointState): Right arm joint commands
    /mpc_status (Float64MultiArray): MPC status (position error, computation time)

Parameters:
    robot_config: Robot configuration file (default: dual_franka.yml)
    scene_config: Scene configuration file (default: collision_table.yml)
    optimization_dt: MPC time step in seconds (default: 0.025)
    interpolation_steps: Interpolation steps between optimization knots (default: 4)
    data_record_path: Path for CSV data recording (default: /tmp/mpc_data/)
    data_record_enabled: Enable CSV recording (default: false)

Usage:
    python3 dual_arm_mpc_controller.py --ros-args -p data_record_enabled:=true
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState as RosJointState
import threading
import numpy as np
import time
import csv
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import torch

# Try to import cuRobo
try:
    from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
    from curobo.types import JointState, Pose, GoalToolPose
    CUROBO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: cuRobo not available: {e}")
    CUROBO_AVAILABLE = False


# Command type constants (matching base_teleop.py)
JOINT_POSITION_INCREMENT = 0
JOINT_VELOCITY = 1
TASK_SPACE_INCREMENT = 2
TASK_SPACE_VELOCITY = 3
JOINT_POSITION = 4
TASK_SPACE_POSE = 5

# Arm selector constants
LEFT_ARM = 1
RIGHT_ARM = 2
BOTH_ARMS = 3


class DualArmMPCController(Node):
    """Dual-Arm MPC Controller using cuRobo."""

    def __init__(self):
        super().__init__('dual_arm_mpc_controller')

        if not CUROBO_AVAILABLE:
            self.get_logger().error("cuRobo is not available. Cannot run MPC controller.")
            raise RuntimeError("cuRobo not available")

        # Declare parameters
        self.declare_parameter('robot_config', 'dual_franka.yml')
        self.declare_parameter('scene_config', 'collision_table.yml')
        self.declare_parameter('optimization_dt', 0.025)
        self.declare_parameter('interpolation_steps', 4)
        self.declare_parameter('control_rate', 100.0)
        self.declare_parameter('data_record_path', '/tmp/mpc_data/')
        self.declare_parameter('data_record_enabled', False)
        self.declare_parameter('use_cuda_graph', True)

        # Get parameters
        robot_config = self.get_parameter('robot_config').value
        scene_config = self.get_parameter('scene_config').value
        optimization_dt = self.get_parameter('optimization_dt').value
        interpolation_steps = self.get_parameter('interpolation_steps').value
        control_rate = self.get_parameter('control_rate').value
        self.data_record_path = self.get_parameter('data_record_path').value
        self.data_record_enabled = self.get_parameter('data_record_enabled').value
        use_cuda_graph = self.get_parameter('use_cuda_graph').value

        # Joint names for ROS2 interface (matching mj_left_joint1-7, mj_right_joint1-7)
        self.ros_joint_names = [f'mj_left_joint{i}' for i in range(1, 8)] + [f'mj_right_joint{i}' for i in range(1, 8)]

        # Joint name mapping: ROS2 -> cuRobo (mj_left_joint1 -> panda_left_joint1, etc.)
        self.joint_name_map = {}
        for i in range(1, 8):
            self.joint_name_map[f'mj_left_joint{i}'] = f'panda_left_joint{i}'
            self.joint_name_map[f'mj_right_joint{i}'] = f'panda_right_joint{i}'

        # Control state
        self.current_joint_positions = np.zeros(14)
        self.current_joint_velocities = np.zeros(14)
        self.joint_states_received = False
        self.teleop_cmd_active = False
        self.last_teleop_time = None
        self.teleop_timeout = 0.1  # 100ms timeout

        # Initialize MPC solver
        self.get_logger().info(f"Initializing MPC with robot={robot_config}, scene={scene_config}")
        try:
            config = ModelPredictiveControlCfg.create(
                robot=robot_config,
                scene_model=scene_config,
                use_cuda_graph=use_cuda_graph,
                optimization_dt=optimization_dt,
                interpolation_steps=interpolation_steps,
            )
            self.mpc = ModelPredictiveControl(config)

            # Get joint names from MPC (should match our expected names)
            self.get_logger().info(f"MPC joint names: {self.mpc.joint_names}")

            # Check if joint names match
            if len(self.mpc.joint_names) != 14:
                self.get_logger().warn(f"MPC has {len(self.mpc.joint_names)} joints, expected 14")

            # Initialize current state from default position
            initial_state = JointState.from_position(
                self.mpc.default_joint_position.clone().unsqueeze(0),
                joint_names=self.mpc.joint_names,
            )
            initial_state.velocity = torch.zeros_like(initial_state.position)
            initial_state.acceleration = torch.zeros_like(initial_state.position)

            self.mpc.setup(initial_state)
            self.get_logger().info("MPC initialized successfully")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize MPC: {e}")
            raise

        # Create QoS profile for reliable communication
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.left_joint_cmd_pub = self.create_publisher(
            RosJointState, '/mj_left/joints_desired', qos)
        self.right_joint_cmd_pub = self.create_publisher(
            RosJointState, '/mj_right/joints_desired', qos)
        self.mpc_status_pub = self.create_publisher(
            Float64MultiArray, '/mpc_status', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            RosJointState, '/joint_states', self.joint_state_callback, 10)
        self.teleop_sub = self.create_subscription(
            Float64MultiArray, '/dualarm_teleop_cmd', self.teleop_callback, qos)

        # Control timer
        self.control_period = 1.0 / control_rate
        self.control_timer = self.create_timer(
            self.control_period, self.control_loop)

        # Data recording
        self.start_time = None
        self.csv_files = {}
        self.data_record_initialized = False

        if self.data_record_enabled:
            self.initialize_data_recording()

        self.get_logger().info(f"Dual-Arm MPC Controller ready at {control_rate}Hz")

    def initialize_data_recording(self):
        """Initialize CSV files for data recording."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            record_dir = Path(self.data_record_path) / timestamp
            record_dir.mkdir(parents=True, exist_ok=True)

            # Create CSV files
            self.csv_files['reference'] = open(record_dir / 'reference.csv', 'w')
            self.csv_files['command'] = open(record_dir / 'command.csv', 'w')
            self.csv_files['mpc_status'] = open(record_dir / 'mpc_status.csv', 'w')
            self.csv_files['teleop'] = open(record_dir / 'teleop.csv', 'w')

            # Write headers
            self.csv_files['reference'].write('timestamp,' + ','.join(self.ros_joint_names) + '\n')
            self.csv_files['command'].write('timestamp,' + ','.join(self.ros_joint_names) + ','
                                           + ','.join([f'{n}_vel' for n in self.ros_joint_names]) + '\n')
            self.csv_files['mpc_status'].write('timestamp,position_error,computation_time,success\n')
            self.csv_files['teleop'].write('timestamp,left_x,left_y,left_z,left_qx,left_qy,left_qz,left_qw,'
                                         'right_x,right_y,right_z,right_qx,right_qy,right_qz,right_qw\n')

            self.start_time = time.time()
            self.data_record_initialized = True
            self.get_logger().info(f"Data recording initialized at {record_dir}")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize data recording: {e}")
            self.data_record_enabled = False

    def record_data(self, reference_pos, command_pos, command_vel, position_error,
                   computation_time, success, teleop_poses=None):
        """Record data to CSV files."""
        if not self.data_record_enabled or not self.data_record_initialized:
            return

        try:
            if self.start_time is None:
                self.start_time = time.time()
            timestamp = time.time() - self.start_time

            # Record reference
            if reference_pos is not None:
                ref_str = ','.join([str(x) for x in reference_pos])
                self.csv_files['reference'].write(f'{timestamp},{ref_str}\n')

            # Record command
            if command_pos is not None and command_vel is not None:
                cmd_str = ','.join([str(x) for x in command_pos] + [str(x) for x in command_vel])
                self.csv_files['command'].write(f'{timestamp},{cmd_str}\n')

            # Record MPC status
            self.csv_files['mpc_status'].write(f'{timestamp},{position_error},{computation_time},{success}\n')

            # Record teleop poses
            if teleop_poses is not None:
                left_pose, right_pose = teleop_poses
                teleop_str = ','.join([str(x) for x in left_pose] + [str(x) for x in right_pose])
                self.csv_files['teleop'].write(f'{timestamp},{teleop_str}\n')

        except Exception as e:
            self.get_logger().error(f"Failed to record data: {e}")

    def joint_state_callback(self, msg: RosJointState):
        """Update current joint states from /joint_states.

        Maps ROS2 joint names (mj_left_joint1-7, mj_right_joint1-7) to
        cuRobo joint names (panda_left_joint1-7, panda_right_joint1-7).
        """
        try:
            # Get current joint positions from ROS2 topic (order may vary)
            ros_joint_positions = np.zeros(14)
            ros_joint_velocities = np.zeros(14)

            for i, ros_name in enumerate(self.ros_joint_names):
                if ros_name in msg.name:
                    idx = msg.name.index(ros_name)
                    ros_joint_positions[i] = msg.position[idx]
                    if len(msg.velocity) > idx:
                        ros_joint_velocities[i] = msg.velocity[idx]

            # Map to cuRobo joint order (panda_left_joint1-7, panda_right_joint1-7)
            curobo_joint_positions = np.zeros(14)
            curobo_joint_velocities = np.zeros(14)

            for i, curobo_name in enumerate(self.mpc.joint_names):
                if curobo_name in self.joint_name_map.values():
                    # Find the ROS name that maps to this cuRobo name
                    for ros_name, curobo_mapped_name in self.joint_name_map.items():
                        if curobo_mapped_name == curobo_name:
                            ros_idx = self.ros_joint_names.index(ros_name)
                            curobo_joint_positions[i] = ros_joint_positions[ros_idx]
                            curobo_joint_velocities[i] = ros_joint_velocities[ros_idx]
                            break
                else:
                    # Direct mapping (finger joints, etc.)
                    if curobo_name in self.ros_joint_names:
                        ros_idx = self.ros_joint_names.index(curobo_name)
                        curobo_joint_positions[i] = ros_joint_positions[ros_idx]
                        curobo_joint_velocities[i] = ros_joint_velocities[ros_idx]

            # Update current state
            self.current_joint_positions = curobo_joint_positions
            self.current_joint_velocities = curobo_joint_velocities

            if not self.joint_states_received:
                self.joint_states_received = True
                self.get_logger().info("Joint states received. Controller ready.")

        except Exception as e:
            self.get_logger().error(f"Error in joint state callback: {e}")

    def teleop_callback(self, msg: Float64MultiArray):
        """Handle teleoperation command from /dualarm_teleop_cmd."""
        try:
            if len(msg.data) < 16:
                self.get_logger().warn(f"Invalid teleop command length: {len(msg.data)}")
                return

            # Extract message data
            # Format: [14 elements data, command_type, arm_selector, allow_safety_violation]
            data = msg.data[:14]
            command_type = int(msg.data[14])
            arm_selector = int(msg.data[15]) if len(msg.data) > 15 else BOTH_ARMS
            allow_safety_violation = msg.data[16] > 0.5 if len(msg.data) > 16 else False

            # Only handle task space pose commands
            if command_type == TASK_SPACE_POSE:
                self.last_teleop_cmd = {
                    'data': data,
                    'arm_selector': arm_selector,
                    'allow_safety_violation': allow_safety_violation
                }
                self.last_teleop_time = time.time()
                self.teleop_cmd_active = True
            else:
                self.get_logger().warn(f"Unsupported command type: {command_type}")

        except Exception as e:
            self.get_logger().error(f"Error in teleop callback: {e}")

    def teleop_to_goal_poses(self, teleop_cmd: Dict) -> Tuple[Pose, Pose, Optional[np.ndarray], Optional[np.ndarray]]:
        """Convert teleop command to goal poses for left and right arms."""
        data = teleop_cmd['data']
        arm_selector = teleop_cmd['arm_selector']

        # Data format for TASK_SPACE_POSE:
        # [left_x,y,z,qx,qy,qz,qw, right_x,y,z,qx,qy,qz,qw]
        left_data = data[:7]
        right_data = data[7:14]

        left_pose = None
        right_pose = None
        left_pose_array = None
        right_pose_array = None

        if arm_selector in [LEFT_ARM, BOTH_ARMS]:
            left_pose = Pose(
                position=torch.tensor([left_data[0], left_data[1], left_data[2]]),
                quaternion=torch.tensor([left_data[6], left_data[3], left_data[4], left_data[5]])  # w, x, y, z
            )
            left_pose_array = left_data

        if arm_selector in [RIGHT_ARM, BOTH_ARMS]:
            right_pose = Pose(
                position=torch.tensor([right_data[0], right_data[1], right_data[2]]),
                quaternion=torch.tensor([right_data[6], right_data[3], right_data[4], right_data[5]])  # w, x, y, z
            )
            right_pose_array = right_data

        return left_pose, right_pose, left_pose_array, right_pose_array

    def control_loop(self):
        """Main control loop running at control_rate."""
        if not self.joint_states_received:
            return

        start_time = time.time()
        position_error = 0.0
        computation_time = 0.0
        success = False

        reference_pos = None
        command_pos = None
        command_vel = None
        teleop_poses = None

        try:
            # Check for teleop timeout
            if self.teleop_cmd_active and self.last_teleop_time is not None:
                if time.time() - self.last_teleop_time > self.teleop_timeout:
                    self.teleop_cmd_active = False
                    self.get_logger().warn("Teleop timeout. Holding position.")

            # Get current state
            current_pos_tensor = torch.tensor(self.current_joint_positions).unsqueeze(0)
            current_vel_tensor = torch.tensor(self.current_joint_velocities).unsqueeze(0)

            current_state = JointState.from_position(
                current_pos_tensor,
                joint_names=self.mpc.joint_names,
            )
            current_state.velocity = current_vel_tensor
            current_state.acceleration = torch.zeros_like(current_vel_tensor)

            if self.teleop_cmd_active and hasattr(self, 'last_teleop_cmd'):
                # Use teleop command as goal
                left_pose, right_pose, left_pose_array, right_pose_array = self.teleop_to_goal_poses(
                    self.last_teleop_cmd)

                goal_dict = {}
                if left_pose is not None and 'panda_left_hand' in self.mpc.tool_frames:
                    goal_dict['panda_left_hand'] = left_pose
                if right_pose is not None and 'panda_right_hand' in self.mpc.tool_frames:
                    goal_dict['panda_right_hand'] = right_pose

                if goal_dict:
                    self.mpc.update_goal_tool_poses(
                        GoalToolPose.from_poses(goal_dict, ordered_tool_frames=self.mpc.tool_frames, num_goalset=1),
                        run_ik=True,
                    )

                    teleop_poses = (left_pose_array, right_pose_array)

            # Run MPC optimization
            result = self.mpc.optimize_action_sequence(current_state)

            if result.action_sequence is not None and result.action_sequence.position.shape[1] > 0:
                # Extract next action (last in the sequence)
                next_position = result.action_sequence.position[:, -1, :]
                next_velocity = result.action_sequence.velocity[:, -1, :]

                # Update state for next iteration
                current_state = JointState.from_position(
                    next_position.clone(),
                    joint_names=self.mpc.joint_names,
                )
                current_state.velocity = next_velocity
                current_state.acceleration = result.action_sequence.acceleration[:, -1, :]

                # Convert to numpy for publishing
                command_pos = next_position.squeeze(0).cpu().numpy()
                command_vel = next_velocity.squeeze(0).cpu().numpy()

                # Split into left and right arms
                left_cmd_pos = command_pos[:7]
                right_cmd_pos = command_pos[7:14]
                left_cmd_vel = command_vel[:7]
                right_cmd_vel = command_vel[7:14]

                # Publish joint commands
                self.publish_joint_commands(left_cmd_pos, right_cmd_pos, left_cmd_vel, right_cmd_vel)

                # Get position error for status
                if result.position_error is not None:
                    position_error = result.position_error.item()

                success = True
            else:
                # MPC failed, hold current position
                self.get_logger().warn("MPC optimization failed. Holding position.")
                command_pos = self.current_joint_positions
                command_vel = np.zeros(14)
                self.publish_joint_commands(
                    self.current_joint_positions[:7], self.current_joint_positions[7:14],
                    np.zeros(7), np.zeros(7)
                )

            computation_time = (time.time() - start_time) * 1000  # ms

            # Publish status
            self.publish_status(position_error, computation_time, success)

            # Record data
            self.record_data(
                reference_pos=None,  # MPC doesn't have explicit reference
                command_pos=command_pos,
                command_vel=command_vel,
                position_error=position_error,
                computation_time=computation_time,
                success=success,
                teleop_poses=teleop_poses
            )

        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")
            import traceback
            traceback.print_exc()

    def publish_joint_commands(self, left_pos, right_pos, left_vel, right_vel):
        """Publish joint commands to both arms.

        Converts from cuRobo joint order to ROS2 joint names.
        """
        # Combine all positions and velocities
        all_pos = np.concatenate([left_pos, right_pos])
        all_vel = np.concatenate([left_vel, right_vel])

        # Create ROS2 joint messages with proper names
        # cuRobo order: panda_left_joint1-7, panda_right_joint1-7
        # ROS2 order: mj_left_joint1-7, mj_right_joint1-7

        left_msg = RosJointState()
        left_msg.header.stamp = self.get_clock().now().to_msg()
        left_msg.name = self.ros_joint_names[:7]  # mj_left_joint1-7
        left_msg.position = all_pos[:7].tolist()
        left_msg.velocity = all_vel[:7].tolist()
        self.left_joint_cmd_pub.publish(left_msg)

        right_msg = RosJointState()
        right_msg.header.stamp = self.get_clock().now().to_msg()
        right_msg.name = self.ros_joint_names[7:]  # mj_right_joint1-7
        right_msg.position = all_pos[7:14].tolist()
        right_msg.velocity = all_vel[7:14].tolist()
        self.right_joint_cmd_pub.publish(right_msg)

    def publish_status(self, position_error, computation_time, success):
        """Publish MPC status."""
        msg = Float64MultiArray()
        msg.data = [position_error, computation_time, 1.0 if success else 0.0]
        self.mpc_status_pub.publish(msg)

    def shutdown(self):
        """Clean up resources."""
        if self.data_record_enabled and self.data_record_initialized:
            for f in self.csv_files.values():
                f.close()
            self.get_logger().info("Data recording closed.")


def main(args=None):
    rclpy.init(args=args)

    try:
        controller = DualArmMPCController()

        thread = threading.Thread(target=rclpy.spin, args=(controller,), daemon=True)
        thread.start()

        while rclpy.ok():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nShutting down MPC controller...")
    except Exception as e:
        print(f'\nError: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if 'controller' in locals():
            controller.shutdown()
            controller.destroy_node()
        rclpy.shutdown()
        print("MPC controller stopped.")


if __name__ == '__main__':
    main()
