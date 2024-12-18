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

    seed = 8
    sim_name = "base_sim"
    env_name = "env_with_obstacles"
    robot_name = "CBF_quadrotor"
    controller_name = "lee_velocity_control"
    args = {}
    num_envs = 256
    use_warp = True
    headless = True
    device = "cuda:0"
    range_cbf_img_size = {
        "height": 16,
        "width": 64,
    }
    CBF_safe_dist = CBFLidarConfig.min_range + 0.4
    cbf_kappa_gain = 1.0
    plot_cbf_constraint = True
    penalize_cbf_constraint = True
    penalize_cbf_corrections = False
    filter_actions = False
    max_speed = 2.0 # m/s
    max_yawrate = 20.0*torch.pi/180.0 # rad/s max yawrate
    max_angle_of_attack = CBFLidarConfig.vertical_fov_deg_max*torch.pi/180.0 # rad max angle of attack
    # Luckily our LiDAR has symmetric vertical field of view, so we don't need a min value

    # These values define the goal state
    goal_speed_limit = 2.0 # m/s
    goal_distance_limit = 1.0 # m
    observation_space_dim = 13 + 4 + vae_config.latent_dims # root state + action_dim _ + lidar_vae_latent_dim
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
        "success_reward": 1000.0,
        "potential_function_mag": 0.0,
        "potential_function_shape": 0.5,
        "linear_potential_function_mag": 20.0,
        "stop_potential_function_mag": 0.0,
        "gamma": 0.99, # discount factor. Note that this is defined in the .yaml file,
        # and needs to be set to the same value here
        # It is used for reward shaping following the shaping theorem
        "not_finished_penalty": -0.5, # To incentivize the agent to finish the episode
        "x_action_diff_penalty_magnitude": 0.8,
        "x_action_diff_penalty_exponent": 3.333,
        "y_action_diff_penalty_magnitude": 0.8,
        "y_action_diff_penalty_exponent": 3.333,
        "z_action_diff_penalty_magnitude": 0.8,
        "z_action_diff_penalty_exponent": 5.0,
        "yawrate_action_diff_penalty_magnitude": 0.8,
        "yawrate_action_diff_penalty_exponent": 3.33,
        "x_absolute_action_penalty_magnitude": 1.6,
        "x_absolute_action_penalty_exponent": 0.3,
        "y_absolute_action_penalty_magnitude": 1.6,
        "y_absolute_action_penalty_exponent": 0.3,
        "z_absolute_action_penalty_magnitude": 1.6,
        "z_absolute_action_penalty_exponent": 0.3,
        "yawrate_absolute_action_penalty_magnitude": 1.6,
        "yawrate_absolute_action_penalty_exponent": 2.0,
        "collision_penalty": -1000.0,
        "cbf_correction_penalty_magnitude" : -0.0,
        "cbf_invariance_penalty_magnitude" : 50.0,
    }

    class curriculum:
        min_level = 30
        max_level = 45
        check_after_log_instances = 2048
        increase_step = 0
        decrease_step = 0
        success_rate_for_increase = 0.8
        success_rate_for_decrease = 0.6

"""
Experiments to consider:
1. Increase the invariance penalty substantially, to see if the agent learns a safer policy
(Hypothesis: No, a large increase in the invariance penalty will make the problem harder to learn
and the agent will only learn a marginally safer policy)
Hypothesis was correct
2. Consider curriculum learning. Figure out how to start from level 0, and gradually increase
3. See what happens if the horizon length is decreased. Do we still learn a safe policy?
4. Need a solid benchmrk. One of the previous benchmarks converged much faster than the others,
need to figure out why. My hypothesis is that the getting_closer_reward_multiplier was too high.
5. Setting gamma lower gives better performance in terms of successrate, but the rewards are low(negative)
The learning is stable however. The problem seems to be that with a low gamma the "getting closer" reward
is mostly negaative, even though the agent is getting closer to the goal. This is because the agent is
not moving fast enough to make the reward positive. I should look into a way to make the reward positive
with low gamma even though the agent is moving slowly. Some sort of function on the potential function
is needed, my intuition is telling me that an exponential function might be the way to go.
6. If gamma is set to 1, it works decently, but the timeout rate is too high. The learning
is quite sensitive to the horizon lenght(probably in combination with lr etc.) however. Try including a
fixed step penalty for intcentivizing the agent to finish the episode.
7. Observation: There seem to be at least two different but not independent issues.
First, the rewards sometimes don't increase, even though we have more successes. This must be due to
wrong reward magnitudes.
Second, the agent seems to struggle with stopping when close to the goal. It seems so, because
the rewards flatten out after a while. This hypothesis can be tested by setting the
goal_speed_limit to 2.0, and see how the rewards behave. It might be that both these issues can be
solved jointly by providing a stopping potential function. I have tried one that is discontinuous
but it didn't work. I should try a continuous one.
8. If th CBF penalty is tuned wrong(so it seems), the agent learns to set u=0.
As this make the CBF penalty 0. What is weird is that the quadcopter then moves, essentially
falling downwards. Not sure if this is intended behaviout. Need to debug this.
Can print in the lee velocity control funciton.
"""