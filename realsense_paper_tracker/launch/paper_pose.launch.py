from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='realsense_paper_tracker',
            executable='paper_pose',
            name='paper_pose',
            output='screen'
        )
    ])
