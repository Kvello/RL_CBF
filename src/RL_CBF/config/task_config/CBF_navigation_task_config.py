from aerial_gym.config.task_config.navigation_task_config import task_config
from exponential_CBF_quadrotor.C_safety_filters.Composite_ECBF_safety_filter_efficient import QuadrotorCompositeCBFSafetyFilterEfficient
from vae_lidar import LIDAR_VAE_DIRECTORY
from aerial_gym import AERIAL_GYM_DIRECTORY
from ..sensor_config.lidar_config.CBF_lidar_config import CBFLidarConfig
import torch

class task_config:
    class vae_config:
        use_camera_vae = False
        use_lidar_vae = True
        latent_dims = 128
        model_file = (
            LIDAR_VAE_DIRECTORY
            + "/vae_lidar.pth"
        )
        # model_file = (
        #     AERIAL_GYM_DIRECTORY
        #     + "/aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"
        # )
        model_folder = AERIAL_GYM_DIRECTORY
        image_size = (CBFLidarConfig.height, CBFLidarConfig.width)
        #image_res = (270, 480)
        #interpolation_mode = "nearest"
        #return_sampled_latent = True

    seed = -1
    sim_name = "base_sim"
    env_name = "env_with_obstacles"
    robot_name = "CBF_quadrotor"
    controller_name = "lee_velocity_control"
    args = {}
    num_envs = 265
    use_warp = True
    headless = True
    device = "cuda:0"
    range_cbf_img_size = {
        "height": 16,
        "width": 64,
    }
    CBF_safe_dist = CBFLidarConfig.min_range + 0.5
    cbf_kappa_gain = 0.5
    plot_cbf_constraint = True
    penalize_cbf_constraint = False
    penalize_cbf_corrections = False
    filter_actions = True
    max_speed = 2.0 # m/s
    max_yawrate = 20.0*torch.pi/180.0 # rad/s max yawrate
    max_angle_of_attack = CBFLidarConfig.vertical_fov_deg_max*torch.pi/180.0 # rad max angle of attack
    # Luckily our LiDAR has symmetric vertical field of view, so we don't need a min value
    observation_space_dim = 13 + 4 + vae_config.latent_dims + 1 # root state + action_dim _ + downsampled_lidar_dims + z-position
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 150  # real physics time for simulation is this value multiplied by sim.dt

    return_state_before_reset = (
        False  # False as usually state is returned for next episode after reset
    )
    # user can set the above to true if they so desire

    target_min_ratio = [0.90, 0.1, 0.1]  # target ratio w.r.t environment bounds in x,y,z
    target_max_ratio = [0.94, 0.90, 0.90]  # target ratio w.r.t environment bounds in x,y,z
    reward_parameters = {
        "success_reward": 100.0,
        "getting_closer_reward_multiplier": 5.0,
        "gamma": 0.995, # discount factor. Note that this is defined in the .yaml file,
        # and needs to be set to the same value here
        # It is used for reward shaping following the shaping theorem
        "x_action_diff_penalty_magnitude": 0.1,
        "x_action_diff_penalty_exponent": 3.333,
        "y_action_diff_penalty_magnitude": 0.1,
        "y_action_diff_penalty_exponent": 3.333,
        "z_action_diff_penalty_magnitude": 0.1,
        "z_action_diff_penalty_exponent": 5.0,
        "yawrate_action_diff_penalty_magnitude": 0.1,
        "yawrate_action_diff_penalty_exponent": 3.33,
        "x_absolute_action_penalty_magnitude": 0.2,
        "x_absolute_action_penalty_exponent": 0.3,
        "y_absolute_action_penalty_magnitude": 0.2,
        "y_absolute_action_penalty_exponent": 0.3,
        "z_absolute_action_penalty_magnitude": 0.2,
        "z_absolute_action_penalty_exponent": 0.3,
        "yawrate_absolute_action_penalty_magnitude": 0.2,
        "yawrate_absolute_action_penalty_exponent": 2.0,
        "collision_penalty": -100.0,
        "cbf_correction_penalty_magnitude" : -20.0,
        "cbf_invariance_penalty_magnitude" : 20.0,
    }

    class curriculum:
        min_level = 10
        max_level = 45
        check_after_log_instances = 2048
        increase_step = 2
        decrease_step = 1
        success_rate_for_increase = 0.8
        success_rate_for_decrease = 0.6
