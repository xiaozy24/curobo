#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
ROS2 Bridge Node for cuRobo MPC reactive control.

Integrates cuRobo's Model Predictive Control with the dual_panda_ws ROS2 system
as a baseline controller对标 dualarm_mprc's TELEOPERATING mode.

Control pipeline:
    csv_teleop_cartesian_absolute.py → /dualarm_teleop_cmd → this node → curobo.MPC
    → /mj_*_arm/joints_desired →底层控制器

Subscribes:
    - /joint_states (sensor_msgs/JointState): Current joint states (mapped from mj_* to panda_*)
    - /dualarm_teleop_cmd (std_msgs/Float64MultiArray): Task space pose goals

Publishes:
    - /mj_left/joints_desired (sensor_msgs/JointState): Left arm joint commands
    - /mj_right/joints_desired (sensor_msgs/JointState): Right arm joint commands

Usage:
    # Terminal 1: Start the simulation
    ros2 launch my_task_description my_task_sim.launch.py

    # Terminal 2: Run the MPC bridge node
    python -m curobo.examples.getting_started.reactive_control_ros2

    # Terminal 3: Play back CSV trajectory
    python3 src/multipanda_ros2/teleop/csv_teleop_cartesian_absolute.py --csv-file trajectory.csv
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import torch
import time
from pathlib import Path
from typing import Dict, Optional
import csv
from collections import deque

# cuRobo imports
from curobo import runtime
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.types import GoalToolPose, Pose, JointState as CuRoboJointState


# Joint name mapping: ROS2/MuJoCo (mj_*) ↔ cuRobo (panda_*)
ROS_TO_CUROBO_MAP = {
    # Left arm
    'mj_left_joint1': 'panda_left_joint1',
    'mj_left_joint2': 'panda_left_joint2',
    'mj_left_joint3': 'panda_left_joint3',
    'mj_left_joint4': 'panda_left_joint4',
    'mj_left_joint5': 'panda_left_joint5',
    'mj_left_joint6': 'panda_left_joint6',
    'mj_left_joint7': 'panda_left_joint7',
    # Right arm
    'mj_right_joint1': 'panda_right_joint1',
    'mj_right_joint2': 'panda_right_joint2',
    'mj_right_joint3': 'panda_right_joint3',
    'mj_right_joint4': 'panda_right_joint4',
    'mj_right_joint5': 'panda_right_joint5',
    'mj_right_joint6': 'panda_right_joint6',
    'mj_right_joint7': 'panda_right_joint7',
}

CUROBO_TO_ROS_MAP = {v: k for k, v in ROS_TO_CUROBO_MAP.items()}

# Joint order for cuRobo (matches dual_panda_mpc.yml)
# Left arm first, then right arm (14 joints, no fingers)
CUROBO_JOINT_ORDER = [
    'panda_left_joint1', 'panda_left_joint2', 'panda_left_joint3',
    'panda_left_joint4', 'panda_left_joint5', 'panda_left_joint6',
    'panda_left_joint7',
    'panda_right_joint1', 'panda_right_joint2', 'panda_right_joint3',
    'panda_right_joint4', 'panda_right_joint5', 'panda_right_joint6',
    'panda_right_joint7',
]

# Command type from base_teleop.py
TASK_SPACE_POSE = 5

# Tool frames for cuRobo
LEFT_TOOL_FRAME = 'panda_left_hand_tcp'
RIGHT_TOOL_FRAME = 'panda_right_hand_tcp'


class DataRecorder:
    """Record MPC control data to CSV files."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize data recorder.

        Args:
            output_dir: Directory to save CSV files. If None, uses default cache dir.
        """
        if output_dir is None:
            output_dir = Path(runtime.cache_dir) / "dual_panda_mpc"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.base_path = output_dir / f"mpc_{timestamp}"

        # Data buffers
        self.teleop_input_buffer = []
        self.joint_command_buffer = []
        self.computation_time_buffer = []
        self.position_error_buffer = []

        self.start_time = None
        self.recording = False

    def start(self):
        """Start recording."""
        self.start_time = time.time()
        self.recording = True
        print(f"[MPC Recorder] Started recording to {self.base_path}_*.csv")

    def stop(self):
        """Stop recording and save to CSV files."""
        if not self.recording:
            return

        self.recording = False
        self._save_all()
        print(f"[MPC Recorder] Saved all data to {self.base_path}_*.csv")

    def _save_all(self):
        """Save all buffers to CSV files."""
        # Save teleop input
        if self.teleop_input_buffer:
            with open(f"{self.base_path}_teleop_input.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'left_x', 'left_y', 'left_z',
                               'left_qx', 'left_qy', 'left_qz', 'left_qw',
                               'right_x', 'right_y', 'right_z',
                               'right_qx', 'right_qy', 'right_qz', 'right_qw'])
                writer.writerows(self.teleop_input_buffer)

        # Save joint commands
        if self.joint_command_buffer:
            with open(f"{self.base_path}_joint_command.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['timestamp']
                header.extend([f'left_j{i}' for i in range(1, 8)])
                header.extend([f'right_j{i}' for i in range(1, 8)])
                writer.writerow(header)
                writer.writerows(self.joint_command_buffer)

        # Save computation time
        if self.computation_time_buffer:
            with open(f"{self.base_path}_computation_time.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'compute_time_ms'])
                writer.writerows(self.computation_time_buffer)

        # Save position error
        if self.position_error_buffer:
            with open(f"{self.base_path}_position_error.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'left_error', 'right_error'])
                writer.writerows(self.position_error_buffer)

    def record_teleop_input(self, left_pose: np.ndarray, right_pose: np.ndarray):
        """Record teleop input pose.

        Args:
            left_pose: [x, y, z, qx, qy, qz, qw]
            right_pose: [x, y, z, qx, qy, qz, qw]
        """
        if not self.recording:
            return
        t = time.time() - self.start_time if self.start_time else 0
        self.teleop_input_buffer.append([t] + left_pose.tolist() + right_pose.tolist())

    def record_joint_command(self, left_q: np.ndarray, right_q: np.ndarray):
        """Record joint command.

        Args:
            left_q: 7-element joint position
            right_q: 7-element joint position
        """
        if not self.recording:
            return
        t = time.time() - self.start_time if self.start_time else 0
        self.joint_command_buffer.append([t] + left_q.tolist() + right_q.tolist())

    def record_computation_time(self, compute_time_ms: float):
        """Record MPC computation time.

        Args:
            compute_time_ms: Computation time in milliseconds
        """
        if not self.recording:
            return
        t = time.time() - self.start_time if self.start_time else 0
        self.computation_time_buffer.append([t, compute_time_ms])

    def record_position_error(self, left_error: float, right_error: float):
        """Record position error.

        Args:
            left_error: Left arm position error
            right_error: Right arm position error
        """
        if not self.recording:
            return
        t = time.time() - self.start_time if self.start_time else 0
        self.position_error_buffer.append([t, left_error, right_error])


class MPCControlNode(Node):
    """ROS2 bridge node for cuRobo MPC control."""

    def __init__(
        self,
        control_rate: float = 100.0,
        record_data: bool = True,
        output_dir: Optional[Path] = None
    ):
        """Initialize MPC control node.

        Args:
            control_rate: Control loop frequency in Hz
            record_data: Whether to record control data
            output_dir: Directory to save recorded data
        """
        super().__init__('reactive_control_mpc')

        # Set use_sim_time parameter
        self.set_parameters([
            rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
        ])

        self.control_rate = control_rate
        self.control_period = 1.0 / control_rate

        # State variables
        self.current_joint_states_ros = {}  # From /joint_states (mj_* names)
        self.current_joint_states_curobo = {}  # Mapped to panda_* names
        self.goal_poses = None  # Current goal poses (left, right)
        self.new_goal_available = False  # Flag for new goal
        self.mpc_initialized = False
        self.recording_enabled = record_data

        # Failure tracking for robustness
        self.mpc_failure_count = 0
        self.max_consecutive_failures = 10  # 100ms = 10 control cycles
        self.last_successful_commands = {'left': None, 'right': None}  # For hold mode

        # Data recorder
        self.recorder = DataRecorder(output_dir) if record_data else None

        # Initialize MPC
        self._init_mpc()

        # ROS2 interfaces
        self._init_ros_interfaces()

        # Control timer
        self.control_timer = self.create_timer(
            self.control_period,
            self.control_callback
        )

        self.get_logger().info(f'MPC Control Node initialized at {control_rate} Hz')
        self.get_logger().info('Waiting for joint states...')

    def _init_mpc(self):
        """Initialize cuRobo MPC solver."""
        self.get_logger().info('Initializing cuRobo MPC...')

        config = ModelPredictiveControlCfg.create(
            robot="dual_panda_mpc.yml",
            scene_model="my_task_scene.yml",
            use_cuda_graph=True,
            optimization_dt=0.01,  # 10ms optimization timestep
            interpolation_steps=4,  # Must be 4 for MPC
            optimizer_collision_activation_distance=0.05,  # 5cm collision activation distance
        )

        self.mpc = ModelPredictiveControl(config)

        # Get default joint position as initial state
        default_pos = self.mpc.default_joint_position.clone()

        # Create initial joint state (no velocity/acceleration)
        self.current_state = CuRoboJointState.from_position(
            default_pos.unsqueeze(0),
            joint_names=self.mpc.joint_names,
        )
        self.current_state.velocity = torch.zeros_like(self.current_state.position)
        self.current_state.acceleration = torch.zeros_like(self.current_state.position)

        # Setup MPC with initial state
        self.mpc.setup(self.current_state)

        # Compute initial end-effector poses as default goals
        kin_result = self.mpc.compute_kinematics(self.current_state)
        self.goal_poses = kin_result.tool_poses.to_dict()

        # Store the state after setup for consistency
        self.mpc_state_after_setup = CuRoboJointState.from_position(
            default_pos.unsqueeze(0),
            joint_names=self.mpc.joint_names,
        )
        self.mpc_state_after_setup.velocity = torch.zeros_like(self.mpc_state_after_setup.position)
        self.mpc_state_after_setup.acceleration = torch.zeros_like(self.mpc_state_after_setup.acceleration)

        # Mark MPC as initialized
        self.mpc_initialized = True
        self.get_logger().info('cuRobo MPC initialized successfully')

    def _init_ros_interfaces(self):
        """Initialize ROS2 publishers and subscribers."""
        # Publishers for joint commands
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.left_joint_pub = self.create_publisher(
            JointState,
            '/mj_left/joints_desired',
            qos
        )

        self.right_joint_pub = self.create_publisher(
            JointState,
            '/mj_right/joints_desired',
            qos
        )

        # Subscriber for current joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            QoSProfile(depth=10)
        )

        # Subscriber for teleop commands
        self.teleop_sub = self.create_subscription(
            Float64MultiArray,
            '/dualarm_teleop_cmd',
            self.teleop_callback,
            qos
        )

    def joint_state_callback(self, msg: JointState):
        """Callback for /joint_states.

        Maps ROS2 joint names (mj_*) to cuRobo joint names (panda_*).
        """
        # Store ROS2 joint states
        for i, name in enumerate(msg.name):
            if name in ROS_TO_CUROBO_MAP:
                self.current_joint_states_ros[name] = msg.position[i]

        # Map to cuRobo names
        self.current_joint_states_curobo = {
            ROS_TO_CUROBO_MAP[ros_name]: pos
            for ros_name, pos in self.current_joint_states_ros.items()
            if ros_name in ROS_TO_CUROBO_MAP
        }

        # Update cuRobo state if we have all joints
        if len(self.current_joint_states_curobo) >= 14:  # 7 per arm (ignoring fingers)
            self._update_curobo_state()

    def _update_curobo_state(self):
        """Update cuRobo joint state from ROS2 /joint_states.

        NOTE: This is only called during initialization. During control,
        the state is updated by MPC results to maintain continuity.
        """
        # Only update from /joint_states during initialization phase
        # Once MPC is running, use MPC's state to maintain continuity
        if self.mpc_initialized:
            return

        # Extract joint positions in cuRobo order (left arm first, then right arm)
        joint_pos = []
        for name in CUROBO_JOINT_ORDER:
            if name in self.current_joint_states_curobo:
                joint_pos.append(self.current_joint_states_curobo[name])
            else:
                # This should not happen if ROS2 is publishing correctly
                self.get_logger().warn(f'Missing joint: {name}')
                return

        # Convert to tensor and update state
        position_tensor = torch.tensor([joint_pos], dtype=torch.float32)

        self.current_state = CuRoboJointState.from_position(
            position_tensor,
            joint_names=self.mpc.joint_names,
        )
        self.current_state.velocity = torch.zeros_like(self.current_state.position)
        self.current_state.acceleration = torch.zeros_like(self.current_state.position)

    def teleop_callback(self, msg: Float64MultiArray):
        """Callback for /dualarm_teleop_cmd.

        Expected message format: [data14, command_type, arm_selector, allow_safety_violation]
        - data14 for TASK_SPACE_POSE: [left_x,y,z,qx,qy,qz,qw, right_x,y,z,qx,qy,qz,qw]

        Only processes TASK_SPACE_POSE (command_type=5) messages.
        """
        if len(msg.data) < 17:
            self.get_logger().warn(f'Invalid teleop message length: {len(msg.data)}')
            return

        data14 = msg.data[:14]
        command_type = int(msg.data[14])
        arm_selector = int(msg.data[15])
        # allow_safety_violation = msg.data[16]  # Not used in MPC

        if command_type != TASK_SPACE_POSE:
            return  # Only handle TASK_SPACE_POSE

        # Extract poses
        left_pose = data14[0:7]   # [x, y, z, qx, qy, qz, qw]
        right_pose = data14[7:14]  # [x, y, z, qx, qy, qz, qw]

        # Record input
        if self.recorder:
            self.recorder.record_teleop_input(
                np.array(left_pose), np.array(right_pose)
            )

        # Update MPC goals
        self._update_mpc_goals(left_pose, right_pose)

    def _update_mpc_goals(self, left_pose, right_pose):
        """Update MPC goal poses.

        Args:
            left_pose: [x, y, z, qx, qy, qz, qw]
            right_pose: [x, y, z, qx, qy, qz, qw]
        """
        if not self.mpc_initialized:
            return

        # Convert to cuRobo Pose objects
        # ROS format: [x, y, z, qx, qy, qz, qw] -> cuRobo format: [w, x, y, z]
        goal_dict = {
            LEFT_TOOL_FRAME: Pose(
                position=torch.tensor(left_pose[:3]),
                quaternion=torch.tensor([left_pose[6], left_pose[3], left_pose[4], left_pose[5]])  # wxyz
            ),
            RIGHT_TOOL_FRAME: Pose(
                position=torch.tensor(right_pose[:3]),
                quaternion=torch.tensor([right_pose[6], right_pose[3], right_pose[4], right_pose[5]])  # wxyz
            )
        }

        self.goal_poses = goal_dict
        self.new_goal_available = True

    def control_callback(self):
        """Main control loop callback."""
        if not self.mpc_initialized:
            return

        # Start recording on first control iteration
        if self.recorder and not self.recorder.recording:
            self.recorder.start()
            self.get_logger().info(f'[MPC] Recording started, data will be saved to {self.recorder.base_path}_*.csv')

        # Update goals if new target is available
        if self.new_goal_available and self.goal_poses is not None:
            try:
                self.mpc.update_goal_tool_poses(
                    GoalToolPose.from_poses(self.goal_poses, ordered_tool_frames=self.mpc.tool_frames, num_goalset=1),
                    run_ik=False,
                )
                self.new_goal_available = False
                self.mpc_failure_count = 0  # Reset on successful goal update
            except Exception as e:
                self.get_logger().warn(f'Failed to update goals: {e}')
                self.new_goal_available = False  # Reset to allow retry with new goal
                self.mpc_failure_count += 1

        try:
            # Run MPC optimization
            start_time = time.time()
            result = self.mpc.optimize_action_sequence(self.current_state)
            compute_time = (time.time() - start_time) * 1000  # ms

            # Record computation time
            if self.recorder:
                self.recorder.record_computation_time(compute_time)

            # Extract action
            if result.action_sequence is not None and result.action_sequence.position.shape[1] > 0:
                next_position = result.action_sequence.position[:, -1, :]  # (1, 14)
                next_velocity = result.action_sequence.velocity[:, -1, :]

                # Update current state for next iteration
                self.current_state = CuRoboJointState.from_position(
                    next_position.clone(),
                    joint_names=self.mpc.joint_names,
                )
                self.current_state.velocity = next_velocity

                # Extract left and right arm joint positions
                # cuRobo order: left arm (0-6), right arm (7-13)
                pos_np = next_position.squeeze(0).cpu().numpy()

                # Left arm: indices 0-6
                left_q = pos_np[0:7]
                # Right arm: indices 7-13
                right_q = pos_np[7:14]

                # Store last successful commands for hold mode
                self.last_successful_commands['left'] = left_q
                self.last_successful_commands['right'] = right_q

                # Publish joint commands
                self._publish_joint_commands(left_q, right_q)

                # Record joint commands
                if self.recorder:
                    self.recorder.record_joint_command(left_q, right_q)

                # Record position error if available
                if result.position_error is not None:
                    pos_err = result.position_error.cpu().item()
                    # Split error roughly in half for each arm (simplified)
                    if self.recorder:
                        self.recorder.record_position_error(pos_err, pos_err)

                # Reset failure count on success
                self.mpc_failure_count = 0
            else:
                self.mpc_failure_count += 1
                self.get_logger().warn(f'MPC returned None result, failure count: {self.mpc_failure_count}')

        except Exception as e:
            self.mpc_failure_count += 1
            self.get_logger().error(f'MPC control error (failure #{self.mpc_failure_count}): {e}')

            # Try to recover from joint_states after multiple failures
            if self.mpc_failure_count >= self.max_consecutive_failures:
                self.get_logger().error('Too many MPC failures, attempting recovery from joint_states...')
                if self._recover_from_joint_states():
                    self.mpc_failure_count = 0
                    self.get_logger().info('Recovery successful')
                else:
                    self.get_logger().error('Recovery failed, entering hold mode')
                    self._publish_hold_command()

    def _publish_joint_commands(self, left_q, right_q):
        """Publish joint commands to ROS2.

        Args:
            left_q: 7-element left arm joint position
            right_q: 7-element right arm joint position
        """
        # Publish left arm command
        left_msg = JointState()
        left_msg.name = [f'mj_left_joint{i}' for i in range(1, 8)]
        left_msg.position = left_q.tolist()
        left_msg.velocity = [0.0] * 7
        left_msg.effort = [0.0] * 7
        self.left_joint_pub.publish(left_msg)

        # Publish right arm command
        right_msg = JointState()
        right_msg.name = [f'mj_right_joint{i}' for i in range(1, 8)]
        right_msg.position = right_q.tolist()
        right_msg.velocity = [0.0] * 7
        right_msg.effort = [0.0] * 7
        self.right_joint_pub.publish(right_msg)

    def _publish_hold_command(self):
        """Publish hold command (keep last known position) when MPC fails repeatedly."""
        if self.last_successful_commands['left'] is not None:
            left_msg = JointState()
            left_msg.name = [f'mj_left_joint{i}' for i in range(1, 8)]
            left_msg.position = self.last_successful_commands['left'].tolist()
            left_msg.velocity = [0.0] * 7
            left_msg.effort = [0.0] * 7
            self.left_joint_pub.publish(left_msg)

        if self.last_successful_commands['right'] is not None:
            right_msg = JointState()
            right_msg.name = [f'mj_right_joint{i}' for i in range(1, 8)]
            right_msg.position = self.last_successful_commands['right'].tolist()
            right_msg.velocity = [0.0] * 7
            right_msg.effort = [0.0] * 7
            self.right_joint_pub.publish(right_msg)

    def _recover_from_joint_states(self) -> bool:
        """Attempt to recover MPC state from /joint_states.

        Returns:
            True if recovery successful, False otherwise.
        """
        if len(self.current_joint_states_curobo) < 14:
            self.get_logger().warn('Insufficient joint states for recovery')
            return False

        try:
            # Extract joint positions in cuRobo order
            joint_pos = []
            for name in CUROBO_JOINT_ORDER:
                if name not in self.current_joint_states_curobo:
                    self.get_logger().warn(f'Missing joint for recovery: {name}')
                    return False
                joint_pos.append(self.current_joint_states_curobo[name])

            # Create new state with correct shape
            position_tensor = torch.tensor([joint_pos], dtype=torch.float32)

            # Re-setup MPC with fresh state
            new_state = CuRoboJointState.from_position(
                position_tensor,
                joint_names=self.mpc.joint_names,
            )
            new_state.velocity = torch.zeros_like(new_state.position)
            new_state.acceleration = torch.zeros_like(new_state.position)

            # Update current state
            self.current_state = new_state

            # Re-compute kinematics for goals
            kin_result = self.mpc.compute_kinematics(self.current_state)
            self.goal_poses = kin_result.tool_poses.to_dict()

            self.get_logger().info('Successfully recovered MPC state from joint_states')
            return True

        except Exception as e:
            self.get_logger().error(f'Recovery failed: {e}')
            return False

    def shutdown(self):
        """Clean shutdown."""
        if self.recorder:
            self.recorder.stop()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    try:
        node = MPCControlNode(
            control_rate=100.0,
            record_data=True
        )

        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nShutting down MPC Control Node...")
    except Exception as e:
        print(f'\nError: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()
        print("MPC Control Node stopped.")


if __name__ == '__main__':
    main()
