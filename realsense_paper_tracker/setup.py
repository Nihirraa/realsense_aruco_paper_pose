from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'realsense_paper_tracker'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='RealSense paper pose tracking with ArUco markers',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'paper_pose = realsense_paper_tracker.paper_pose:main',
            'aruco_detect = realsense_paper_tracker.aruco_detect:main',
        ],
    },
)
