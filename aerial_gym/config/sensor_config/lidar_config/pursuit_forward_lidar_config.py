from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import BaseLidarConfig


class PursuitForwardM3_120x25_UltraHighResLidarConfig(BaseLidarConfig):
    """Heavy ultra-high-resolution forward profile.

    This approximates the public RoboSense M3 ROI headline specs: 120 deg x
    25 deg FOV with 0.05 deg x 0.05 deg angular resolution and 300 m class
    ranging. It is intended for feasibility probes, not high-env-count PPO.
    """

    num_sensors = 1
    sensor_type = "lidar"

    height = 501
    width = 2401
    horizontal_fov_deg_min = -60.0
    horizontal_fov_deg_max = 60.0
    vertical_fov_deg_min = -12.5
    vertical_fov_deg_max = +12.5
    max_range = 300.0
    min_range = 1.0

    return_pointcloud = False
    pointcloud_in_world_frame = False
    segmentation_camera = False

    # Warp LiDAR rays use local +X as the forward axis.
    euler_frame_rot_deg = [0.0, 0.0, 0.0]

    normalize_range = True
    far_out_of_range_value = max_range
    near_out_of_range_value = -max_range

    randomize_placement = True
    min_translation = [0.10, 0.0, 0.03]
    max_translation = [0.10, 0.0, 0.03]
    min_euler_rotation_deg = [0.0, 0.0, 0.0]
    max_euler_rotation_deg = [0.0, 0.0, 0.0]

    nominal_position = [0.10, 0.0, 0.03]
    nominal_orientation_euler_deg = [0.0, 0.0, 0.0]

    class sensor_noise:
        enable_sensor_noise = False
        std_a = 3.08287454e-06
        std_b = -4.07347360e-06
        std_c = 5.30757302e-03
        mean_offset = -0.025
        pixel_dropout_prob = 0.0
