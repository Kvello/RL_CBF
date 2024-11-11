from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np
from gym.spaces import Dict, Box
from aerial_gym.utils.vae.vae_image_encoder import VAEImageEncoder
from ..config.sensor_config.lidar_config.CBF_lidar_config import CBFLidarConfig
from aerial_gym.utils.math import *
from aerial_gym.utils.logging import CustomLogger
from exponential_CBF_quadrotor.C_safety_filters.Composite_first_order_CBF import FirstOrderCompositeQuadCollisionCBF
logger = CustomLogger("CBF_navigation_task")

class LiDARDownsampler(torch.nn.Module):
    def __init__(self,width = 32, height = 8):
        super(LiDARDownsampler, self).__init__()
        downsample_factor_width = round(CBFLidarConfig.width/width)
        downsample_factor_height = round(CBFLidarConfig.height/height)
        self.downsample = torch.nn.MaxPool2d(kernel_size=(downsample_factor_height,
                                                          downsample_factor_width),)
        self.direction_map = None
        self.width = width
        self.height = height
        self.range_limits = (CBFLidarConfig.min_range,CBFLidarConfig.max_range)
    def forward(self, x):
        # Normalize before or after downsampling?
        # min-max normalization
        x = (x-self.range_limits[0])/(self.range_limits[1]-self.range_limits[0])
        ret = -self.downsample(-x)
        return ret
    def get_displacements(self,lidar_image)->torch.Tensor:
        # TODO: test this function
        if(self.direction_map == None):
            theta_angles = torch.linspace(CBFLidarConfig.horizontal_fov_deg_min,
                                          CBFLidarConfig.horizontal_fov_deg_max,
                                          self.width,device=lidar_image.device)
            phi_angles = torch.linspace(CBFLidarConfig.vertical_fov_deg_min,
                                        CBFLidarConfig.vertical_fov_deg_max,
                                        self.height,device=lidar_image.device)
            angle_space_p, angle_space_t = torch.meshgrid(phi_angles, theta_angles)
            self.direction_map = torch.stack([torch.cos(angle_space_t)*torch.cos(angle_space_p),
                                              torch.sin(angle_space_t)*torch.cos(angle_space_p),
                                              torch.sin(angle_space_p)], dim = -1)
        ranges = lidar_image.reshape(-1, self.height, self.width)
        displacements = ranges.unsqueeze(-1) * self.direction_map
        return displacements.reshape(-1, self.height*self.width, 3)
# Simple RL task with CBF based safety filter
class CBFNavigationTask(BaseTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        # overwrite the params if user has provided them
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp
        super().__init__(task_config)
        self.device = task_config.device
        self.lidar_downsampler = LiDARDownsampler(
            task_config.lidar_downsampler_config["width"], 
            task_config.lidar_downsampler_config["height"]
        )
        # Put all reward parameters to torch tensor on device
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device)
        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )
        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.target_min_ratio = torch.tensor(
            self.task_config.target_min_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs,-1)
        self.target_max_ratio = torch.tensor(
            self.task_config.target_max_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs,-1)

        self.success_aggregate = 0
        self.crashes_aggregate = 0
        self.timeouts_aggregate = 0
        self.pos_error_vehicle_frame_prev = torch.zeros_like(self.target_position)

        self.pos_error_vehicle_frame = torch.zeros_like(self.target_position)

        if self.task_config.vae_config.use_vae:
            self.vae_model = VAEImageEncoder(config=self.task_config.vae_config, device=self.device)
            self.image_latents = torch.zeros(
                (self.sim_env.num_envs, self.task_config.vae_config.latent_dims),
                device=self.device,
                requires_grad=False,
            )
        else:
            self.vae_model = lambda x: x
        self.downsampled_lidar_displacements = torch.zeros(
            (self.sim_env.num_envs, 
             self.lidar_downsampler.width*self.lidar_downsampler.height
             ,3), device=self.device, requires_grad=False
        )
        self.downsampled_lidar = torch.zeros(
            (self.sim_env.num_envs,
             self.lidar_downsampler.width*self.lidar_downsampler.height), device=self.device, requires_grad=False
        )
        # Get the dictionary once from the environment and use it to get the observations later.
        # This is to avoid constant retuning of data back anf forth across functions as the tensors update and can be read in-place.
        self.obs_dict = self.sim_env.get_obs()
        if "curriculum_level" not in self.obs_dict.keys():
            self.curriculum_level = self.task_config.curriculum.min_level
            self.obs_dict["curriculum_level"] = self.curriculum_level
        else:
            self.curriculum_level = self.obs_dict["curriculum_level"]
        self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
        self.curriculum_progress_fraction = (
            self.curriculum_level - self.task_config.curriculum.min_level
        ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)
        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_transformation_function = self.task_config.action_transformation_function

        self.num_envs = self.sim_env.num_envs

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.sim_env.num_envs,
                    self.task_config.privileged_observation_space_dim,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }
        self.num_task_steps = 0

        self.mass = 0.25 # Read this from the terminal
        # Should find a way to read it from the simulation
        self.collision_cbf = FirstOrderCompositeQuadCollisionCBF(
            mass=self.mass,
            eps = task_config.CBF_safe_dist,
            num_obstacles=task_config.lidar_num_obs,
            kappa=1e2)
    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self.reset_idx(torch.arange(self.sim_env.num_envs))
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        target_ratio = torch_rand_float_tensor(self.target_min_ratio, self.target_max_ratio)
        self.target_position[env_ids] = torch_interpolate_ratio(
            min=self.obs_dict["env_bounds_min"][env_ids],
            max=self.obs_dict["env_bounds_max"][env_ids],
            ratio=target_ratio[env_ids],
        )
        # logger.warning(f"reset envs: {env_ids}")
        self.infos = {}
        return

    def render(self):
        return self.sim_env.render()

    def logging_sanity_check(self, infos):
        successes = infos["successes"]
        crashes = infos["crashes"]
        timeouts = infos["timeouts"]
        time_at_crash = torch.where(
            crashes > 0,
            self.sim_env.sim_steps,
            self.task_config.episode_len_steps * torch.ones_like(self.sim_env.sim_steps),
        )
        env_list_for_toc = (time_at_crash < 5).nonzero(as_tuple=False).squeeze(-1)
        crash_envs = crashes.nonzero(as_tuple=False).squeeze(-1)
        success_envs = successes.nonzero(as_tuple=False).squeeze(-1)
        timeout_envs = timeouts.nonzero(as_tuple=False).squeeze(-1)

        if len(env_list_for_toc) > 0:
            logger.critical("Crash is happening too soon.")
            logger.critical(f"Envs crashing too soon: {env_list_for_toc}")
            logger.critical(f"Time at crash: {time_at_crash[env_list_for_toc]}")

        if torch.sum(torch.logical_and(successes, crashes)) > 0:
            logger.critical("Success and crash are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, successes))}"
            )
        if torch.sum(torch.logical_and(successes, timeouts)) > 0:
            logger.critical("Success and timeout are occuring at the same time")
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of timeouts: {torch.count_nonzero(timeouts)}, Timeout envs: {timeout_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(successes, timeouts))}"
            )
        if torch.sum(torch.logical_and(crashes, timeouts)) > 0:
            logger.critical("Crash and timeout are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of timeouts: {torch.count_nonzero(timeouts)}, Timeout envs: {timeout_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, timeouts))}"
            )
        return

    def check_and_update_curriculum_level(self, successes, crashes, timeouts):
        self.success_aggregate += torch.sum(successes)
        self.crashes_aggregate += torch.sum(crashes)
        self.timeouts_aggregate += torch.sum(timeouts)

        instances = self.success_aggregate + self.crashes_aggregate + self.timeouts_aggregate

        if instances >= self.task_config.curriculum.check_after_log_instances:
            success_rate = self.success_aggregate / instances
            crash_rate = self.crashes_aggregate / instances
            timeout_rate = self.timeouts_aggregate / instances

            if success_rate > self.task_config.curriculum.success_rate_for_increase:
                self.curriculum_level += self.task_config.curriculum.increase_step
            elif success_rate < self.task_config.curriculum.success_rate_for_decrease:
                self.curriculum_level -= self.task_config.curriculum.decrease_step

            # clamp curriculum_level
            self.curriculum_level = min(
                max(self.curriculum_level, self.task_config.curriculum.min_level),
                self.task_config.curriculum.max_level,
            )
            self.obs_dict["curriculum_level"] = self.curriculum_level
            self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
            self.curriculum_progress_fraction = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

            logger.warning(
                f"Curriculum Level: {self.curriculum_level}, Curriculum progress fraction: {self.curriculum_progress_fraction}"
            )
            logger.warning(
                f"\nSuccess Rate: {success_rate}\nCrash Rate: {crash_rate}\nTimeout Rate: {timeout_rate}"
            )
            logger.warning(
                f"\nSuccesses: {self.success_aggregate}\nCrashes : {self.crashes_aggregate}\nTimeouts: {self.timeouts_aggregate}"
            )
            self.success_aggregate = 0
            self.crashes_aggregate = 0
            self.timeouts_aggregate = 0

    def process_image_observation(self):
        image_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
        self.image_latents[:] = self.vae_model.encode(image_obs)
        # # comments to make sure the VAE does as expected
        # decoded_image = self.vae_model.decode(self.image_latents[0].unsqueeze(0))
        # image0 = image_obs[0].cpu().numpy()
        # decoded_image0 = decoded_image[0].squeeze(0).cpu().numpy()
        # # save as .png with timestep
        # if not hasattr(self, "img_ctr"):
        #     self.img_ctr = 0
        # self.img_ctr += 1
        # import matplotlib.pyplot as plt
        # plt.imsave(f"image0{self.img_ctr}.png", image0, vmin=0, vmax=1)
        # plt.imsave(f"decoded_image0{self.img_ctr}.png", decoded_image0, vmin=0, vmax=1)
    def process_lidar_observation(self):
        lidar_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
        self.downsampled_lidar[:] = self.lidar_downsampler(lidar_obs)
        # original_img = lidar_obs[0].cpu().numpy()
        # downsampled_img = self.downsampled_lidar[0].reshape(6,16).cpu().numpy()
        # if not hasattr(self, "img_ctr"):
        #     self.img_ctr = 0
        # self.img_ctr += 1
        # import matplotlib.pyplot as plt
        # plt.imsave(f"original{self.img_ctr}.png", original_img, vmin=0, vmax=1)
        # plt.imsave(f"downsampled{self.img_ctr}.png", downsampled_img, vmin=0, vmax=1)
        self.downsampled_lidar_displacements[:] = \
            self.lidar_downsampler.get_displacements(self.downsampled_lidar)
    def step(self, actions):
        # this uses the action, gets observations
        # calculates rewards, returns tuples
        # In this case, the episodes that are terminated need to be
        # first reset, and the first obseration of the new episode
        # needs to be returned.

        transformed_action = self.action_transformation_function(actions)
        logger.debug(f"raw_action: {actions[0]}, transformed action: {transformed_action[0]}")
        self.sim_env.step(actions=transformed_action)

        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        # logger.info(f"Curricluum Level: {self.curriculum_level}")

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        # successes are are the sum of the environments which are to be truncated and have reached the target within a distance threshold
        successes = self.truncations * (
            torch.norm(self.target_position - self.obs_dict["robot_position"], dim=1) < 1.0
        )
        successes = torch.where(self.terminations > 0, torch.zeros_like(successes), successes)
        timeouts = torch.where(
            self.truncations > 0, torch.logical_not(successes), torch.zeros_like(successes)
        )
        timeouts = torch.where(
            self.terminations > 0, torch.zeros_like(timeouts), timeouts
        )  # timeouts are not counted if there is a crash

        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = self.terminations

        self.logging_sanity_check(self.infos)
        self.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )
        # rendering happens at the post-reward calculation step since the newer measurement is required to be
        # sent to the RL algorithm as an observation and it helps if the camera image is updated then
        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)
        self.num_task_steps += 1
        # do stuff with the image observations here
        # self.process_image_observation()
        self.process_lidar_observation()
        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()
        return return_tuple

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        self.task_obs["observations"][:, 0:3] = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )
        self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_vehicle_orientation"]
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 13:17] = self.obs_dict["robot_actions"]
        self.task_obs["observations"][:, 17:] = self.downsampled_lidar
        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def compute_rewards_and_crashes(self, obs_dict):
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        robot_lin_vel_command = self.obs_dict["robot_actions"][:,0:3]
        cbf_values = self.collision_cbf.get_composite_cbf_value(
            robot_position,
            disp = self.downsampled_lidar_displacements
        )
        cbf_derivatives = self.collision_cbf.get_h_derivative(
            robot_position,
            robot_lin_vel_command,
            disp= self.downsampled_lidar_displacements
        )
        target_orientation = torch.zeros_like(robot_orientation, device=self.device)
        target_orientation[:, 3] = 1.0
        self.pos_error_vehicle_frame_prev[:] = self.pos_error_vehicle_frame
        self.pos_error_vehicle_frame[:] = quat_rotate_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        return compute_reward(
            self.pos_error_vehicle_frame,
            self.pos_error_vehicle_frame_prev,
            obs_dict["crashes"],
            obs_dict["robot_actions"],
            obs_dict["robot_prev_actions"],
            cbf_derivatives,
            cbf_values,
            self.curriculum_progress_fraction,
            self.task_config.reward_parameters
        )


@torch.jit.script
def exponential_reward_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward function"""
    return magnitude * torch.exp(-(value * value) * exponent)


@torch.jit.script
def exponential_penalty_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward function"""
    return magnitude * (torch.exp(-(value * value) * exponent) - 1.0)


@torch.jit.script
def compute_reward(
    pos_error,
    prev_pos_error,
    crashes,
    action,
    prev_action,
    cbf_derivative,
    cbf_value,
    curriculum_progress_fraction,
    parameter_dict,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, float, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]
    MULTIPLICATION_FACTOR_REWARD = (1.0 + (2.0) * curriculum_progress_fraction) * 3.0
    dist = torch.norm(pos_error, dim=1)
    prev_dist_to_goal = torch.norm(prev_pos_error, dim=1)
    pos_reward = exponential_reward_function(
        parameter_dict["pos_reward_magnitude"],
        parameter_dict["pos_reward_exponent"],
        dist,
    )
    very_close_to_goal_reward = exponential_reward_function(
        parameter_dict["very_close_to_goal_reward_magnitude"],
        parameter_dict["very_close_to_goal_reward_exponent"],
        dist,
    )
    getting_closer_reward = parameter_dict["getting_closer_reward_multiplier"] * (
        prev_dist_to_goal - dist
    )
    distance_from_goal_reward = (20.0 - dist) / 20.0
    action_diff = action - prev_action
    x_diff_penalty = exponential_penalty_function(
        parameter_dict["x_action_diff_penalty_magnitude"],
        parameter_dict["x_action_diff_penalty_exponent"],
        action_diff[:, 0],
    )
    z_diff_penalty = exponential_penalty_function(
        parameter_dict["z_action_diff_penalty_magnitude"],
        parameter_dict["z_action_diff_penalty_exponent"],
        action_diff[:, 2],
    )
    yawrate_diff_penalty = exponential_penalty_function(
        parameter_dict["yawrate_action_diff_penalty_magnitude"],
        parameter_dict["yawrate_action_diff_penalty_exponent"],
        action_diff[:, 3],
    )
    action_diff_penalty = x_diff_penalty + z_diff_penalty + yawrate_diff_penalty
    # absolute action penalty
    x_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["x_absolute_action_penalty_magnitude"],
        parameter_dict["x_absolute_action_penalty_exponent"],
        action[:, 0],
    )
    z_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["z_absolute_action_penalty_magnitude"],
        parameter_dict["z_absolute_action_penalty_exponent"],
        action[:, 2],
    )
    yawrate_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["yawrate_absolute_action_penalty_magnitude"],
        parameter_dict["yawrate_absolute_action_penalty_exponent"],
        action[:, 3],
    )
    absolute_action_penalty = x_absolute_penalty + z_absolute_penalty + yawrate_absolute_penalty
    # TODO: Tune the cbf invariance penalty, to be
    # a) comparable to the other penalties
    # b) consider exponential penalty
    # c) using high enought value to maybe guarantee that the CBF is satisfied
    cbf_inv_penalty = cbf_derivative + parameter_dict["cbf_kappa_gain"]*cbf_value
    cbf_inv_penalty = torch.clamp(cbf_inv_penalty, max=0.0)
    cbf_inv_penalty *= parameter_dict["cbf_invariance_penalty_magnitude"]
    total_action_penalty = action_diff_penalty + absolute_action_penalty + cbf_inv_penalty
    # combined reward
    reward = (
        MULTIPLICATION_FACTOR_REWARD
        * (
            pos_reward
            + very_close_to_goal_reward
            + getting_closer_reward
            + distance_from_goal_reward
        )
        + total_action_penalty
    )

    reward[:] = torch.where(
        crashes > 0,
        parameter_dict["collision_penalty"] * torch.ones_like(reward),
        reward,
    )
    return reward, crashes
