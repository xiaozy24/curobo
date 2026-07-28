#!/usr/bin/env python3
"""
Launch file for Dual-Arm MPC Controller.

This launch file starts the dual-arm MPC controller node with configurable parameters.

Usage:
    python3 mpc_controller.launch.py
    # With data recording enabled:
    python3 mpc_controller.launch.py data_record_enabled:=true
    # Custom robot config:
    python3 mpc_controller.launch.py robot_config:=dual_franka.yml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    # Declare launch arguments
    robot_config_arg = DeclareLaunchArgument(
        'robot_config',
        default_value='dual_franka.yml',
        description='Robot configuration file for cuRobo MPC'
    )

    scene_config_arg = DeclareLaunchArgument(
        'scene_config',
        default_value='collision_table.yml',
        description='Scene configuration file for cuRobo MPC'
    )

    optimization_dt_arg = DeclareLaunchArgument(
        'optimization_dt',
        default_value='0.025',
        description='MPC optimization time step (seconds)'
    )

    interpolation_steps_arg = DeclareLaunchArgument(
        'interpolation_steps',
        default_value='4',
        description='Interpolation steps between optimization knots'
    )

    control_rate_arg = DeclareLaunchArgument(
        'control_rate',
        default_value='100.0',
        description='Control loop rate (Hz)'
    )

    data_record_path_arg = DeclareLaunchArgument(
        'data_record_path',
        default_value='/tmp/mpc_data/',
        description='Path for CSV data recording'
    )

    data_record_enabled_arg = DeclareLaunchArgument(
        'data_record_enabled',
        default_value='false',
        description='Enable CSV data recording'
    )

    use_cuda_graph_arg = DeclareLaunchArgument(
        'use_cuda_graph',
        default_value='true',
        description='Use CUDA graph for faster optimization'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    # MPC Controller node
    mpc_controller_node = Node(
        package='curobo.examples',  # This needs to match the package setup
        executable='dual_arm_mpc_controller',
        name='dual_arm_mpc_controller',
        output='screen',
        parameters=[{
            'robot_config': LaunchConfiguration('robot_config'),
            'scene_config': LaunchConfiguration('scene_config'),
            'optimization_dt': LaunchConfiguration('optimization_dt'),
            'interpolation_steps': LaunchConfiguration('interpolation_steps'),
            'control_rate': LaunchConfiguration('control_rate'),
            'data_record_path': LaunchConfiguration('data_record_path'),
            'data_record_enabled': LaunchConfiguration('data_record_enabled'),
            'use_cuda_graph': LaunchConfiguration('use_cuda_graph'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )

    return LaunchDescription([
        robot_config_arg,
        scene_config_arg,
        optimization_dt_arg,
        interpolation_steps_arg,
        control_rate_arg,
        data_record_path_arg,
        data_record_enabled_arg,
        use_cuda_graph_arg,
        use_sim_time_arg,
        mpc_controller_node,
    ])


if __name__ == '__main__':
    generate_launch_description()
