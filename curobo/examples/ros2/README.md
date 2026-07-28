# Dual-Arm MPC Controller for ROS2

This directory contains a ROS2 node that implements a Model Predictive Controller for dual-arm robot manipulation using cuRobo's MPC solver. It serves as a drop-in replacement for the `dual_arm_safe_controller` from the dualarm_mprc package.

## Features

- Real-time MPC optimization at 100Hz
- Task-space control (Cartesian pose tracking)
- Collision avoidance with scene obstacles
- CSV data logging for analysis
- Compatible with existing teleoperation interfaces

## Architecture

The controller implements the following pipeline:

```
Upper Layer (csv_teleop_cartesian_absolute.py)
    ↓
/dualarm_teleop_cmd (Float64MultiArray)
    ↓
MPC Controller (dual_arm_mpc_controller.py)
    ↓
/mj_left/joints_desired & /mj_right/joints_desired (JointState)
    ↓
Lower Layer (MuJoCo + impedance controller)
```

## Installation

1. Ensure cuRobo is installed with ROS2 dependencies:
```bash
cd /home/xiaozy24/dual_panda_ws/src/curobo
pip install -e .[cu13-torch]
pip install rclpy sensor_msgs std_msgs
```

## Usage

### Running the Controller

```bash
# Terminal 1: Start simulation
source /home/xiaozy24/dual_panda_ws/install/setup.bash
ros2 launch my_task_description my_task_sim.launch.py

# Terminal 2: Start MPC controller
source /home/xiaozy24/dual_panda_ws/src/curobo/.venv/bin/activate  # or your venv
python3 -m curobo.examples.ros2.dual_arm_mpc_controller --ros-args -p data_record_enabled:=true

# Terminal 3: Run teleoperation
source /home/xiaozy24/dual_panda_ws/install/setup.bash
python3 src/multipanda_ros2/teleop/csv_teleop_cartesian_absolute.py --csv-file trajectory.csv
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `robot_config` | string | `dual_franka_home.yml` | Robot configuration file (uses home position) |
| `scene_config` | string | `collision_table.yml` | Scene configuration file |
| `optimization_dt` | float | 0.025 | MPC time step (seconds) |
| `interpolation_steps` | int | 4 | Interpolation steps |
| `control_rate` | float | 100.0 | Control loop rate (Hz) |
| `data_record_path` | string | `/tmp/mpc_data/` | CSV recording path |
| `data_record_enabled` | bool | false | Enable CSV recording |
| `use_cuda_graph` | bool | true | Use CUDA graph optimization |

### ROS2 Topics

**Subscribed Topics:**
- `/dualarm_teleop_cmd` (std_msgs/Float64MultiArray): Teleoperation commands
- `/joint_states` (sensor_msgs/JointState): Current joint states

**Published Topics:**
- `/mj_left/joints_desired` (sensor_msgs/JointState): Left arm joint commands
- `/mj_right/joints_desired` (sensor_msgs/JointState): Right arm joint commands
- `/mpc_status` (std_msgs/Float64MultiArray): MPC status (position_error, computation_time, success)

## Data Recording

When `data_record_enabled:=true`, the controller records the following CSV files:

- `reference.csv`: Target joint positions
- `command.csv`: Actual joint commands sent to lower controller
- `mpc_status.csv`: MPC optimization status (position error, computation time, success)
- `teleop.csv`: Received teleoperation commands

## Comparison with dual_arm_safe_controller

| Feature | MPC Controller | safe_controller |
|---------|----------------|-----------------|
| Optimization Method | Model Predictive Control (cuRobo) | Hierarchical QP (HQP) |
| Collision Avoidance | GPU-accelerated distance queries | Control Barrier Function (CBF) |
| Constraint Handling | Soft constraints with weights | Hierarchical priority constraints |
| Relative Pose Constraint | Via RelativePoseCost | Via HQP constraint |
| Singularity Avoidance | Via cspace cost | Via explicit constraint |
| Computation | GPU (CUDA) | CPU (Eigen/OSQP) |

## Troubleshooting

### Import Error: No module named 'curobo'
Ensure cuRobo is installed:
```bash
cd /home/xiaozy24/dual_panda_ws/src/curobo
pip install -e .[cu13-torch]
```

### CUDA Error
Make sure CUDA is available and cuRobo is built with CUDA support:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Joint states not received
Check that MuJoCo simulation is running and publishing `/joint_states`:
```bash
ros2 topic echo /joint_states
```

## File Structure

```
ros2/
├── dual_arm_mpc_controller.py   # Main controller node
├── mpc_controller.launch.py     # Launch file
└── README.md                     # This file
```

## References

- cuRobo MPC: `curobo/examples/getting_started/reactive_control.py`
- dualarm_safe_controller: `src/dualarm_mprc/dualarm_reactive_control/src/dual_arm_safe_controller_sim.cpp`
