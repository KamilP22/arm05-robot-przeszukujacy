import os
 
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
 
 
def generate_launch_description():
    params = os.path.join(get_package_share_directory('arm05_explorer'),
                          'config', 'params.yaml')
    return LaunchDescription([
        Node(package='arm05_explorer', executable='aruco_detector',
             name='aruco_detector', parameters=[params], output='screen'),
        Node(package='arm05_explorer', executable='search_server',
             name='search_server', parameters=[params], output='screen'),
    ])