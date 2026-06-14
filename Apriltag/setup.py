import os
from glob import glob
from setuptools import setup

package_name = 'tello_localization'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'map'), ['map/apriltag_map.yaml']),
    ],
    install_requires=['setuptools', 'pupil-apriltags', 'opencv-python'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Detect AprilTags from a camera stream using pupil_apriltags.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'apriltag_detector_node = tello_localization.apriltag_detector_node:main',
            'tag_tf_broadcaster = tello_localization.tag_tf_broadcaster:main',
            'ekf_localization_node = tello_localization.ekf_localization_node:main',
            'control_tello_ekf = tello_localization.control_tello_ekf:main',
        ],
    },
)

