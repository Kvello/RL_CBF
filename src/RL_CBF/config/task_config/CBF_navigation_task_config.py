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
    CBF_safe_dist = CBFLidarConfig.min_range + 0.4
    cbf_kappa_gain = 3.0
    plot_cbf_constraint = True
    penalize_cbf_constraint = False
    penalize_cbf_corrections = True
    filter_actions = True
    observation_space_dim = 13 + 4 + vae_config.latent_dims #+1+1# root_state + action_dim _ + downsampled_lidar_dims + CBF_dim + CBF_derivative_dim
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 100  # real physics time for simulation is this value multiplied by sim.dt

    return_state_before_reset = (
        False  # False as usually state is returned for next episode after reset
    )
    # user can set the above to true if they so desire

    target_min_ratio = [0.90, 0.1, 0.1]  # target ratio w.r.t environment bounds in x,y,z
    target_max_ratio = [0.94, 0.90, 0.90]  # target ratio w.r.t environment bounds in x,y,z
    reward_parameters = {
        "pos_reward_magnitude": 5.0,
        "pos_reward_exponent": 1.0 / 3.5,
        "very_close_to_goal_reward_magnitude": 5.0,
        "very_close_to_goal_reward_exponent": 2.0,
        "getting_closer_reward_multiplier": 10.0,
        "x_action_diff_penalty_magnitude": 0.8,
        "x_action_diff_penalty_exponent": 3.333,
        "z_action_diff_penalty_magnitude": 0.8,
        "z_action_diff_penalty_exponent": 5.0,
        "yawrate_action_diff_penalty_magnitude": 0.8,
        "yawrate_action_diff_penalty_exponent": 3.33,
        "x_absolute_action_penalty_magnitude": 1.6,
        "x_absolute_action_penalty_exponent": 0.3,
        "z_absolute_action_penalty_magnitude": 1.5,
        "z_absolute_action_penalty_exponent": 1.0,
        "yawrate_absolute_action_penalty_magnitude": 1.5,
        "yawrate_absolute_action_penalty_exponent": 2.0,
        "collision_penalty": -50.0,
        "cbf_correction_penalty_magnitude" : -50.0,
        "cbf_invariance_penalty_magnitude" : 50.0,
    }

    class curriculum:
        min_level = 10
        max_level = 45
        check_after_log_instances = 2048
        increase_step = 2
        decrease_step = 1
        success_rate_for_increase = 0.7
        success_rate_for_decrease = 0.6
