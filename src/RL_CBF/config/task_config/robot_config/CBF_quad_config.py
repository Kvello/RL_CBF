from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg

from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)
from aerial_gym.config.sensor_config.lidar_config.osdome_64_config import OSDome_64_Config
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig


class CBFQuadCfg(BaseQuadCfg):
    class sensor_config:
        enable_camera = True
        camera_config = BaseDepthCameraConfig

        enable_lidar = True
        lidar_config = BaseLidarConfig  # OSDome_64_Config

        enable_imu = False
        imu_config = BaseImuConfig 
    
    class disturbance:
        enable_disturbance = False # False for simplicity
        # prob_apply_disturbance = 0.02
        # max_force_and_torque_disturbance = [0.75, 0.75, 0.75, 0.004, 0.004, 0.004]