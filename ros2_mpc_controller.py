#!/usr/bin/env python3
"""
ROS2 MPC Controller - Integrates curobo MPC with dual Panda teleop interface.

This node provides a baseline controller using curobo's MPC for comparison
with the dualarm_mprc safe controller.

Robot Configuration:
- Uses dual_franka_ros2.yml which matches the ROS2 simulation setup
- Base link: base_link
- Left arm: mj_left_* (positioned at y=+0.26m, no rotation)
- Right arm: mj_right_* (positioned at y=-0.26m, no rotation)
- Tool frames: mj_left_hand_tcp, mj_right_hand_tcp

Interface:
- Subscribes: /joint_states (current joint positions)
- Subscribes: /dualarm_teleop_cmd (teleop commands, compatible with csv_teleop_cartesian_absolute.py)
- Publishes: /mj_left/joints_desired, /mj_right/joints_desired (joint commands)
- Publishes: /ee_pose_mpc (end-effector poses for debugging)
- Publishes: /mpc_timing (computation timing)

CSV Recording:
- Records teleop commands, MPC results, timing, and collision data
- Compatible format with dual_arm_safe_controller_sim.cpp for comparison

Usage:
    source ~/myenv/bin/activate
    python3 src/curobo/ros2_mpc_controller.py --data-record-prefix exp_name
"""

from __future__ import annotations

import os
import sys
import time
import csv
import threading
from pathlib import Path
from typing import Optional, Tuple
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

# curobo imports
import torch
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.types import GoalToolPose, JointState as CuroboJointState, Pose
from curobo._src.cost.cost_relative_pose_cfg import RelativePoseCostCfg


# Quaternion helper functions (not available in curobo's quaternion module)
def quat_conjugate(q):
    """Compute quaternion conjugate [w, x, y, z]."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def quat_rotate_vector(q, v):
    """Rotate vector v by quaternion q."""
    # v_rotated = q * v * q_conjugate (treating v as pure quaternion)
    q_vec = q[..., 1:]
    uv = torch.cross(q_vec, v, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return v + 2.0 * (q[..., 0:1] * uv + uuv)


class ROS2MPCController(Node):
    """ROS2 node that wraps curobo MPC for dual Panda control."""

    # Joint names (same for ROS2 and curobo - matching simulation)
    LEFT_JOINT_PREFIX = "mj_left_joint"
    RIGHT_JOINT_PREFIX = "mj_right_joint"
    JOINT_NAMES = (
        "mj_left_joint1", "mj_left_joint2", "mj_left_joint3", "mj_left_joint4",
        "mj_left_joint5", "mj_left_joint6", "mj_left_joint7",
        "mj_right_joint1", "mj_right_joint2", "mj_right_joint3", "mj_right_joint4",
        "mj_right_joint5", "mj_right_joint6", "mj_right_joint7",
    )

    # Tool frames (must match robot config)
    LEFT_TOOL_FRAME = "mj_left_hand_tcp"
    RIGHT_TOOL_FRAME = "mj_right_hand_tcp"

    # Command type constants (must match csv_teleop_cartesian_absolute.py)
    TASK_SPACE_POSE = 5

    def __init__(
        self,
        robot_file: str = "dual_franka_ros2.yml",
        scene_dict: Optional[dict] = None,
        optimization_dt: float = 0.025,
        cold_start_iters: int = 100,
        warm_start_iters: int = 200,
        data_record_path: str = "",
        data_record_prefix: str = "",
        publish_rate: float = 100.0,
    ):
        super().__init__('ros2_mpc_controller')

        self.robot_file = robot_file
        self.publish_rate = publish_rate
        self.optimization_dt = optimization_dt
        self.data_record_path = data_record_path
        self.data_record_prefix = data_record_prefix
        self.data_record_on = bool(data_record_prefix)

        # State variables
        self.current_joint_positions = np.zeros(14)
        self.current_joint_velocities = np.zeros(14)
        self.joint_states_received = False

        # Teleop state
        self.teleop_cmd_active = False
        self.last_teleop_time = self.get_clock().now()
        self.teleop_cmd_timeout = 0.5  # seconds
        self.teleop_mutex = threading.Lock()

        # Goal poses (initially unknown)
        self.left_goal_pose = None  # (position, quaternion)
        self.right_goal_pose = None
        self.goals_initialized = False

        # Timing statistics
        self.solve_times = []
        self.collision_times = []

        # Recording files
        self.record_files = {}
        self.record_writers = {}
        self.record_row_counts = {}
        self.record_start_time = 0.0

        # Initialize MPC
        self._init_mpc(scene_dict, cold_start_iters, warm_start_iters)

        # Create ROS2 interfaces
        self._create_publishers()
        self._create_subscribers()
        self._create_timer()

        self.get_logger().info("ROS2 MPC Controller initialized")
        self.get_logger().info(f"Robot config: {robot_file}")
        self.get_logger().info(f"Optimization dt: {optimization_dt}s")
        self.get_logger().info(f"Publish rate: {publish_rate}Hz")

        # Open recording files if enabled
        if self.data_record_on:
            self._open_record_files()

    def _init_mpc(
        self,
        scene_dict: Optional[dict],
        cold_start_iters: int,
        warm_start_iters: int,
    ):
        """Initialize curobo MPC."""
        self.get_logger().info("Initializing curobo MPC...")

        config = ModelPredictiveControlCfg.create(
            robot=self.robot_file,
            scene_model=scene_dict,
            self_collision_check=True,
            collision_cache={"cuboid": 50, "mesh": 10},
            optimization_dt=self.optimization_dt,
            use_cuda_graph=True,
            cold_start_optimization_num_iters=cold_start_iters,
            warm_start_optimization_num_iters=warm_start_iters,
        )

        # Inject relative pose constraint (optional, disabled by default)
        self._inject_relative_pose_config(config)

        self.mpc = ModelPredictiveControl(config)

        # Disable IK CUDA graph (prevents issues with dynamic goal updates)
        self._disable_ik_cuda_graph()

        # Initialize current state
        self.current_state = CuroboJointState.from_position(
            self.mpc.default_joint_position.clone().unsqueeze(0),
            joint_names=self.mpc.joint_names,
        )
        self.current_state.velocity = torch.zeros_like(self.current_state.position)
        self.current_state.acceleration = torch.zeros_like(self.current_state.position)

        # Setup MPC (will be called once we have actual joint states)
        self.mpc_setup_complete = False

        # Initialize relative pose control (disabled by default)
        self._init_relative_pose_control()

        self.get_logger().info("MPC initialization complete")

    def _inject_relative_pose_config(self, config) -> None:
        """Inject relative pose constraint into MPC config (disabled by default)."""
        rel_cfg = RelativePoseCostCfg(
            weight=torch.tensor([7500.0, 750.0]),
            primary_tool_frame=self.LEFT_TOOL_FRAME,
            secondary_tool_frame=self.RIGHT_TOOL_FRAME,
            position_tolerance=0.0,
            orientation_tolerance=0.0,
        )
        core = config.core_cfg
        for rc in core.optimizer_rollout_configs:
            if rc.constraint_cfg is not None:
                rc.constraint_cfg.relative_pose_cfg = rel_cfg
        mr = core.metrics_rollout_config
        if mr is not None and mr.constraint_cfg is not None:
            mr.constraint_cfg.relative_pose_cfg = rel_cfg

    def _disable_ik_cuda_graph(self) -> None:
        """Disable IK solver CUDA graph for dynamic goal updates."""
        try:
            for key in list(self.mpc.ik_solver.__dict__.keys()):
                if "seed" in key and hasattr(
                    getattr(self.mpc.ik_solver, key), "config"
                ):
                    getattr(self.mpc.ik_solver, key).config.use_cuda_graph = False
        except Exception:
            pass

    def _init_relative_pose_control(self) -> None:
        """Initialize relative pose control (disabled by default)."""
        for rollout in self.mpc.core.get_all_rollout_instances():
            for mgr in rollout._cost_manager_list:
                if mgr.has_cost("relative_pose"):
                    mgr.get_cost("relative_pose").disable_cost()

    def _create_publishers(self):
        """Create ROS2 publishers."""
        # Joint command publishers (to low-level controllers)
        self.left_joint_pub = self.create_publisher(
            JointState,
            '/mj_left/joints_desired',
            10
        )
        self.right_joint_pub = self.create_publisher(
            JointState,
            '/mj_right/joints_desired',
            10
        )

        # End-effector pose publisher (for debugging)
        self.ee_pose_pub = self.create_publisher(
            Float64MultiArray,
            '/ee_pose_mpc',
            10
        )

        # Timing publisher
        self.timing_pub = self.create_publisher(
            Float64MultiArray,
            '/mpc_timing',
            10
        )

    def _create_subscribers(self):
        """Create ROS2 subscribers."""
        # Joint states subscriber
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        # Teleop command subscriber
        self.teleop_cmd_sub = self.create_subscription(
            Float64MultiArray,
            '/dualarm_teleop_cmd',
            self.teleop_cmd_callback,
            10
        )

    def _create_timer(self):
        """Create control loop timer."""
        timer_period = 1.0 / self.publish_rate
        self.control_timer = self.create_timer(timer_period, self.control_loop)

    def joint_state_callback(self, msg: JointState):
        """Update current joint states from /joint_states."""
        try:
            # Extract left arm joint states
            left_indices = []
            for i in range(1, 8):
                joint_name = f'{ROS2MPCController.LEFT_JOINT_PREFIX}{i}'
                if joint_name in msg.name:
                    left_indices.append(msg.name.index(joint_name))

            # Extract right arm joint states
            right_indices = []
            for i in range(1, 8):
                joint_name = f'{ROS2MPCController.RIGHT_JOINT_PREFIX}{i}'
                if joint_name in msg.name:
                    right_indices.append(msg.name.index(joint_name))

            if len(left_indices) == 7 and len(right_indices) == 7:
                # Update positions
                for i, idx in enumerate(left_indices):
                    self.current_joint_positions[i] = msg.position[idx]
                    if idx < len(msg.velocity):
                        self.current_joint_velocities[i] = msg.velocity[idx]

                for i, idx in enumerate(right_indices):
                    self.current_joint_positions[7 + i] = msg.position[idx]
                    if idx < len(msg.velocity):
                        self.current_joint_velocities[7 + i] = msg.velocity[idx]

                # Update MPC internal state
                self._update_mpc_current_state()

                if not self.joint_states_received:
                    self.joint_states_received = True
                    self.get_logger().info("Joint states received")

        except Exception as e:
            self.get_logger().error(f'Error in joint_state_callback: {e}')

    def _update_mpc_current_state(self):
        """Update MPC's current joint state from ROS2 joint states."""
        device = self.mpc.device_cfg.device
        joint_pos_tensor = torch.tensor(
            self.current_joint_positions,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        self.current_state = CuroboJointState.from_position(
            joint_pos_tensor,
            joint_names=self.mpc.joint_names,
        )
        self.current_state.velocity = torch.tensor(
            self.current_joint_velocities,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        self.current_state.acceleration = torch.zeros_like(self.current_state.velocity)

        # Setup MPC on first call
        if not self.mpc_setup_complete:
            self.mpc.setup(self.current_state.clone())
            self.mpc_setup_complete = True
            self.get_logger().info("MPC setup complete")

            # Initialize goals to current pose
            self._initialize_goals_from_current_pose()

    def _initialize_goals_from_current_pose(self):
        """Initialize goal poses to current end-effector poses."""
        kin = self.mpc.compute_kinematics(self.current_state.clone())
        tp = kin.tool_poses

        # Extract left arm pose
        left_pos = tp[self.LEFT_TOOL_FRAME].position.detach().cpu().squeeze().numpy()
        left_quat = tp[self.LEFT_TOOL_FRAME].quaternion.detach().cpu().squeeze().numpy()
        self.left_goal_pose = (left_pos, left_quat)

        # Extract right arm pose
        right_pos = tp[self.RIGHT_TOOL_FRAME].position.detach().cpu().squeeze().numpy()
        right_quat = tp[self.RIGHT_TOOL_FRAME].quaternion.detach().cpu().squeeze().numpy()
        self.right_goal_pose = (right_pos, right_quat)

        self.goals_initialized = True

    def teleop_cmd_callback(self, msg: Float64MultiArray):
        """Handle teleop commands from /dualarm_teleop_cmd."""
        with self.teleop_mutex:
            # Validate message size
            if len(msg.data) < 15:
                self.get_logger().warning(
                    f'Teleop command has insufficient data: {len(msg.data)} elements (expected at least 15)'
                )
                return

            # Decode message format:
            # [0-13]: 14 data elements
            # [14]: command_type
            # [15]: arm_selector
            # [16]: allow_safety_violation

            command_type = int(msg.data[14])
            arm_selector = int(msg.data[15]) if len(msg.data) > 15 else 3  # 3=both arms

            # Only handle TASK_SPACE_POSE commands
            if command_type != self.TASK_SPACE_POSE:
                self.get_logger().warning(
                    f'Unsupported command type: {command_type} (only TASK_SPACE_POSE=5 supported)'
                )
                return

            # Process task space pose
            self._process_task_space_pose(msg.data, arm_selector)

            self.last_teleop_time = self.get_clock().now()
            self.teleop_cmd_active = True

    def _process_task_space_pose(self, data: np.ndarray, arm_selector: int):
        """Process TASK_SPACE_POSE command."""
        # Format: [left_x,y,z, qx,qy,qz,qw, right_x,y,z, qx,qy,qz,qw] (14 elements)

        # Process left arm
        if arm_selector == 1 or arm_selector == 3:  # left or both
            left_pos = np.array([data[0], data[1], data[2]])
            left_quat = np.array([data[4], data[5], data[6], data[3]])  # Convert to wxyz
            self.left_goal_pose = (left_pos, left_quat)

        # Process right arm
        if arm_selector == 2 or arm_selector == 3:  # right or both
            right_pos = np.array([data[7], data[8], data[9]])
            right_quat = np.array([data[11], data[12], data[13], data[10]])  # Convert to wxyz
            self.right_goal_pose = (right_pos, right_quat)

    def control_loop(self):
        """Main control loop - runs MPC optimization and publishes commands."""
        if not self.joint_states_received or not self.mpc_setup_complete:
            return

        try:
            # Check for teleop timeout
            with self.teleop_mutex:
                time_since_last_cmd = (self.get_clock().now() - self.last_teleop_time).nanoseconds / 1e9
                if time_since_last_cmd > self.teleop_cmd_timeout:
                    self.teleop_cmd_active = False

            # Run MPC optimization
            start_time = time.time()
            result = self._run_mpc_step()
            solve_time = time.time() - start_time

            if result and result['success']:
                # Publish joint commands
                self._publish_joint_commands()

                # Publish end-effector poses
                self._publish_ee_poses()

                # Publish timing
                self._publish_timing(solve_time, result.get('min_scene_dist', 0.0), result.get('min_self_dist', 0.0))

                # Record data if enabled
                if self.data_record_on:
                    self._record_step(solve_time, result)

        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')
            import traceback
            traceback.print_exc()

    def _run_mpc_step(self) -> Optional[dict]:
        """Run one MPC optimization step."""
        if not self.goals_initialized:
            return None

        device = self.mpc.device_cfg.device

        # Update goal poses
        self.mpc.update_goal_tool_poses(
            self._numpy_poses_to_goal_tool_pose(self.left_goal_pose, self.right_goal_pose, device),
            run_ik=False,
        )

        # Optimize
        mpc_result = self.mpc.optimize_action_sequence(self.current_state.clone())

        if mpc_result is None or mpc_result.action_sequence is None:
            return {'success': False}

        # Update current state with result
        if mpc_result.action_sequence.position.shape[1] > 0:
            next_position = mpc_result.action_sequence.position[:, -1, :]
            self.current_state = CuroboJointState.from_position(
                next_position.clone(),
                joint_names=self.mpc.joint_names,
            )
            self.current_state.velocity = mpc_result.action_sequence.velocity[:, -1, :]
            self.current_state.acceleration = mpc_result.action_sequence.acceleration[:, -1, :]

        # Update numpy state
        self.current_joint_positions = self.current_state.position.detach().cpu().squeeze().numpy()
        self.current_joint_velocities = self.current_state.velocity.detach().cpu().squeeze().numpy()

        # Compute collision distances
        min_scene_dist = self._get_min_scene_distance()
        min_self_dist = self._get_min_self_distance()

        return {
            'success': True,
            'solve_time': mpc_result.solve_time,
            'position_error': float(mpc_result.position_error.max().item()) if mpc_result.position_error is not None else None,
            'min_scene_dist': min_scene_dist,
            'min_self_dist': min_self_dist,
        }

    def _numpy_poses_to_goal_tool_pose(
        self,
        left_pose: Tuple[np.ndarray, np.ndarray],
        right_pose: Tuple[np.ndarray, np.ndarray],
        device: torch.device,
    ) -> GoalToolPose:
        """Convert numpy poses to GoalToolPose."""
        left_pos, left_quat = left_pose
        right_pos, right_quat = right_pose

        return GoalToolPose.from_poses(
            {
                self.LEFT_TOOL_FRAME: Pose(
                    position=torch.from_numpy(left_pos).float().to(device).unsqueeze(0),
                    quaternion=torch.from_numpy(left_quat).float().to(device).unsqueeze(0),
                ),
                self.RIGHT_TOOL_FRAME: Pose(
                    position=torch.from_numpy(right_pos).float().to(device).unsqueeze(0),
                    quaternion=torch.from_numpy(right_quat).float().to(device).unsqueeze(0),
                ),
            },
            ordered_tool_frames=self.mpc.tool_frames,
            num_goalset=1,
        )

    def _publish_joint_commands(self):
        """Publish joint commands to low-level controllers."""
        # Left arm
        left_msg = JointState()
        left_msg.header.stamp = self.get_clock().now().to_msg()
        left_msg.name = [f'{ROS2MPCController.LEFT_JOINT_PREFIX}{i}' for i in range(1, 8)]
        left_msg.position = self.current_joint_positions[:7].tolist()
        left_msg.velocity = self.current_joint_velocities[:7].tolist()
        self.left_joint_pub.publish(left_msg)

        # Right arm
        right_msg = JointState()
        right_msg.header.stamp = self.get_clock().now().to_msg()
        right_msg.name = [f'{ROS2MPCController.RIGHT_JOINT_PREFIX}{i}' for i in range(1, 8)]
        right_msg.position = self.current_joint_positions[7:14].tolist()
        right_msg.velocity = self.current_joint_velocities[7:14].tolist()
        self.right_joint_pub.publish(right_msg)

    def _publish_ee_poses(self):
        """Publish current end-effector poses for debugging."""
        kin = self.mpc.compute_kinematics(self.current_state.clone())
        tp = kin.tool_poses

        # Left arm
        left_pos = tp[self.LEFT_TOOL_FRAME].position.detach().cpu().squeeze().numpy()
        left_quat = tp[self.LEFT_TOOL_FRAME].quaternion.detach().cpu().squeeze().numpy()

        # Right arm
        right_pos = tp[self.RIGHT_TOOL_FRAME].position.detach().cpu().squeeze().numpy()
        right_quat = tp[self.RIGHT_TOOL_FRAME].quaternion.detach().cpu().squeeze().numpy()

        # Publish as Float64MultiArray: [left_x,y,z,qx,qy,qz,qw, right_x,y,z,qx,qy,qz,qw]
        msg = Float64MultiArray()
        msg.data = [
            float(left_pos[0]), float(left_pos[1]), float(left_pos[2]),
            float(left_quat[1]), float(left_quat[2]), float(left_quat[3]), float(left_quat[0]),
            float(right_pos[0]), float(right_pos[1]), float(right_pos[2]),
            float(right_quat[1]), float(right_quat[2]), float(right_quat[3]), float(right_quat[0]),
        ]
        self.ee_pose_pub.publish(msg)

    def _publish_timing(self, solve_time: float, min_scene_dist: float, min_self_dist: float):
        """Publish computation timing and collision info."""
        msg = Float64MultiArray()
        msg.data = [solve_time, min_scene_dist, min_self_dist]
        self.timing_pub.publish(msg)

    def _get_min_scene_distance(self) -> float:
        """Compute minimum distance to scene obstacles."""
        sc = self.mpc.scene_collision_checker
        if sc is None:
            return 1.0

        try:
            cdata = sc.data.cuboids
            if cdata is None or cdata.count[0].item() == 0:
                return 1.0

            kin = self.mpc.compute_kinematics(self.current_state)
            spheres = kin.robot_spheres.squeeze()
            active = spheres[:, 3] > 1e-6
            if not active.any():
                return 1.0

            sc_m = spheres[active, :3]
            sr = spheres[active, 3]

            n_cub = int(cdata.count[0].item())
            enabled = cdata.enable[0, :n_cub].bool()
            if not enabled.any():
                return 1.0

            half = cdata.dims[0, :n_cub, :3][enabled] / 2.0
            c_pos = -cdata.inv_pose[0, :n_cub, :3][enabled]
            c_min = c_pos - half
            c_max = c_pos + half

            closest = torch.clamp(sc_m.unsqueeze(1), c_min.unsqueeze(0), c_max.unsqueeze(0))
            dist = torch.norm(sc_m.unsqueeze(1) - closest, dim=2)
            signed = dist - sr.unsqueeze(1)

            return float(signed.min().item())
        except Exception:
            return 1.0

    def _get_min_self_distance(self) -> float:
        """Read self-collision penetration depth."""
        try:
            for rollout in self.mpc.core.get_all_rollout_instances():
                for mgr in rollout._cost_manager_list:
                    if not mgr.has_cost("self_collision"):
                        continue
                    cost = mgr.get_cost("self_collision")
                    if not cost.enabled:
                        continue
                    if getattr(cost.config, "convert_to_binary", False):
                        continue
                    if hasattr(cost, "_out_distance") and cost._out_distance.numel() > 0:
                        d = cost._out_distance.detach()
                        w = cost._weight.detach().item()
                        return float(-d.max().item() / w)
        except Exception:
            pass
        return 0.0

    def _open_record_files(self):
        """Open CSV recording files."""
        if not self.data_record_on:
            return

        record_dir = Path(self.data_record_path) / self.data_record_prefix
        record_dir.mkdir(parents=True, exist_ok=True)

        self.record_start_time = time.time()
        self.get_logger().info(f"Data recording enabled: {record_dir}")

        # Open files
        self.record_files = {}
        self.record_writers = {}

        # Teleop tracking data (similar to safe controller)
        teleop_file = record_dir / f"teleop_tracking_{self.data_record_prefix}.csv"
        self.record_files['teleop'] = open(teleop_file, 'w', newline='')
        self.record_writers['teleop'] = csv.writer(self.record_files['teleop'])
        self.record_row_counts['teleop'] = 0
        # Write header
        self.record_writers['teleop'].writerow([
            'timestamp', 'arm_id',
            'target_ee_x', 'target_ee_y', 'target_ee_z', 'target_ee_qx', 'target_ee_qy', 'target_ee_qz', 'target_ee_qw',
            'target_q1', 'target_q2', 'target_q3', 'target_q4', 'target_q5', 'target_q6', 'target_q7',
            'actual_q1', 'actual_q2', 'actual_q3', 'actual_q4', 'actual_q5', 'actual_q6', 'actual_q7',
            'min_scene_dist', 'min_self_dist', 'solve_time', 'success'
        ])
        self.get_logger().info(f"Opened: {teleop_file}")

        # Timing data
        timing_file = record_dir / f"computation_time_{self.data_record_prefix}.csv"
        self.record_files['timing'] = open(timing_file, 'w', newline='')
        self.record_writers['timing'] = csv.writer(self.record_files['timing'])
        self.record_writers['timing'].writerow(['timestamp', 'solve_time', 'collision_time'])
        self.get_logger().info(f"Opened: {timing_file}")

        # End-effector pose data
        ee_file = record_dir / f"ee_pose_{self.data_record_prefix}.csv"
        self.record_files['ee'] = open(ee_file, 'w', newline='')
        self.record_writers['ee'] = csv.writer(self.record_files['ee'])
        self.record_writers['ee'].writerow([
            'timestamp', 'arm_id',
            'ee_x', 'ee_y', 'ee_z', 'ee_qx', 'ee_qy', 'ee_qz', 'ee_qw'
        ])
        self.get_logger().info(f"Opened: {ee_file}")

        # Collision distance data
        collision_file = record_dir / f"collision_distance_{self.data_record_prefix}.csv"
        self.record_files['collision'] = open(collision_file, 'w', newline='')
        self.record_writers['collision'] = csv.writer(self.record_files['collision'])
        self.record_writers['collision'].writerow([
            'timestamp', 'min_scene_dist', 'min_self_dist'
        ])
        self.get_logger().info(f"Opened: {collision_file}")

    def _close_record_files(self):
        """Close CSV recording files."""
        if not self.data_record_on:
            return

        for file in self.record_files.values():
            file.close()
        self.get_logger().info("Data recording files closed")

    def _record_step(self, solve_time: float, result: dict):
        """Record data for CSV logging."""
        if not self.data_record_on or not self.record_files:
            return

        timestamp = time.time() - self.record_start_time

        # Record teleop tracking data for both arms
        with self.teleop_mutex:
            if self.goals_initialized and self.left_goal_pose and self.right_goal_pose:
                goals = [(self.left_goal_pose, 1), (self.right_goal_pose, 2)]
                for (goal_pos, goal_quat), arm_id in goals:
                    offset = (arm_id - 1) * 7
                    row = [
                        timestamp, arm_id,
                        # Target end-effector pose
                        goal_pos[0], goal_pos[1], goal_pos[2],
                        goal_quat[1], goal_quat[2], goal_quat[3], goal_quat[0],  # wxyz
                        # Target joint positions (from MPC)
                        *self.current_joint_positions[offset:offset+7].tolist(),
                        # Actual joint positions (from joint states)
                        *self.current_joint_positions[offset:offset+7].tolist(),
                    ]
                    row.extend([
                        result.get('min_scene_dist', 0.0),
                        result.get('min_self_dist', 0.0),
                        solve_time,
                        1 if result.get('success', False) else 0,
                    ])
                    self.record_writers['teleop'].writerow(row)
                    self.record_row_counts['teleop'] += 1

        # Record timing
        self.record_writers['timing'].writerow([
            timestamp, solve_time, 0.0  # collision time (not separately measured)
        ])

        # Record end-effector poses
        if self.goals_initialized:
            kin = self.mpc.compute_kinematics(self.current_state.clone())
            tp = kin.tool_poses

            frames = [(self.LEFT_TOOL_FRAME, 1), (self.RIGHT_TOOL_FRAME, 2)]
            for tool_frame, arm_id in frames:
                pose = tp[tool_frame]
                pos = pose.position.detach().cpu().squeeze().numpy()
                quat = pose.quaternion.detach().cpu().squeeze().numpy()
                self.record_writers['ee'].writerow([
                    timestamp, arm_id,
                    pos[0], pos[1], pos[2],
                    quat[1], quat[2], quat[3], quat[0],  # wxyz
                ])

        # Record collision distances
        self.record_writers['collision'].writerow([
            timestamp,
            result.get('min_scene_dist', 0.0),
            result.get('min_self_dist', 0.0),
        ])

    def destroy(self):
        """Clean up resources."""
        self._close_record_files()
        self.mpc.destroy()
        super().destroy()


def main(args=None):
    parser = argparse.ArgumentParser(description='ROS2 MPC Controller with curobo')
    parser.add_argument('--robot-file', type=str, default='dual_franka_ros2.yml',
                        help='Robot configuration file')
    parser.add_argument('--optimization-dt', type=float, default=0.025,
                        help='Optimization timestep (seconds)')
    parser.add_argument('--cold-start-iters', type=int, default=100,
                        help='Cold start iterations')
    parser.add_argument('--warm-start-iters', type=int, default=200,
                        help='Warm start iterations')
    parser.add_argument('--publish-rate', type=float, default=100.0,
                        help='Publish rate (Hz)')
    parser.add_argument('--data-record-path', type=str, default='',
                        help='Data recording path')
    parser.add_argument('--data-record-prefix', type=str, default='',
                        help='Data recording prefix (enables recording if set)')
    args = parser.parse_args(args)

    rclpy.init()

    try:
        node = ROS2MPCController(
            robot_file=args.robot_file,
            optimization_dt=args.optimization_dt,
            cold_start_iters=args.cold_start_iters,
            warm_start_iters=args.warm_start_iters,
            data_record_path=args.data_record_path,
            data_record_prefix=args.data_record_prefix,
            publish_rate=args.publish_rate,
        )

        thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        thread.start()

        print("ROS2 MPC Controller running. Press Ctrl+C to stop.")
        while rclpy.ok():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f'\nError: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy()
        rclpy.shutdown()
        print("ROS2 MPC Controller stopped.")


if __name__ == '__main__':
    main()
