from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np
from gym.spaces import Dict, Box
from vae_lidar.vae import Vae as VAELidar
from aerial_gym.utils.vae.vae_image_encoder import VAEImageEncoder
from config.sensor_config.lidar_config.CBF_lidar_config import CBFLidarConfig
from aerial_gym.utils.math import *
from aerial_gym.utils.logging import CustomLogger
from exponential_CBF_quadrotor.C_safety_filters.Composite_first_order_CBF import FirstOrderCompositeQuadCollisionCBF
from lidar_downsampler.lidar_downsampler import LiDARDownsampler
import wandb
logger = CustomLogger("CBF_navigation_task")

# Simple RL task with CBF based safety filter
class CBFNavigationTask(BaseTask):
    def __init__(
        self,
        task_config, 
        seed=None, 
        num_envs=None, 
        headless=None, 
        device=None, 
        use_warp=None
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
        self.lidar_cbf_data = LiDARDownsampler(
            sensor_config= CBFLidarConfig,
            width = task_config.range_cbf_img_size["width"], 
            height = task_config.range_cbf_img_size["height"]
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

        if self.task_config.vae_config.use_lidar_vae:
            self.vae_model = VAELidar(size_latent=self.task_config.vae_config.latent_dims,
                                      shape_imgs=self.task_config.vae_config.image_size,
                                      filename=self.task_config.vae_config.model_file,
                                      device=self.device)
            self.range_latents = torch.zeros(
                (self.sim_env.num_envs, self.task_config.vae_config.latent_dims),
                device=self.device,
                requires_grad=False,
            )
        elif self.task_config.vae_config.use_camera_vae:
            self.vae_model = VAEImageEncoder(config=self.task_config.vae_config, 
                                             device=self.device)
            
            self.range_latents = torch.zeros(
                (self.sim_env.num_envs, self.task_config.vae_config.latent_dims),
                device=self.device,
                requires_grad=False,
            )
        else:
            self.vae_model = lambda x: x
        self.downsampled_lidar_displacements = torch.zeros(
            (self.sim_env.num_envs, 
             self.lidar_cbf_data.downsampled_width*self.lidar_cbf_data.downsampled_height,3),
             device=self.device, requires_grad=False
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

        self.terminations = torch.zeros_like(self.obs_dict["crashes"])
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
            num_obstacles=task_config.range_cbf_img_size["width"]\
                *task_config.range_cbf_img_size["height"],
            kappa=1e2,
            gamma=1,
            quadratic=False)
    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self.reset_idx(torch.arange(self.sim_env.num_envs))
        return self.get_return_tuple()
    # TODO: Change the reset_idx function to not spawn the quadotor in positions
    # where the CBF constraint cannot be satisfied for the given maximal speed
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
            if wandb.run is not None:
                wandb.log({"Curriculum Level": self.curriculum_level})
                wandb.log({"Success Rate": success_rate})
                wandb.log({"Crash Rate": crash_rate})
                wandb.log({"Timeout Rate": timeout_rate})

    def action_transformation_function(self,action):
        position = self.obs_dict["robot_position"]
        # Actions are polar decomposition of the velocity in the body frame,
        # and the yaw rate. We need to restrict the bearing angle, so that
        # the direction of the velocity is in a direction observable by the LiDAR.
        # This makes the CBF constraint more meaningful.
        # We have defined the action space to be between -1 and 1 for all the actions
        transformed_action = torch.zeros_like(action)
        theta = action[:, 0]*torch.pi # crab angle
        phi = action[:, 1]*self.task_config.max_angle_of_attack # Angle of attack
        speed = (action[:, 2] + 1.0)/2.0*self.task_config.max_speed # m/s. Don't want negative speeds
        transformed_action[:,0] = torch.cos(theta)*torch.cos(phi)*speed
        transformed_action[:,1] = torch.sin(theta)*torch.cos(phi)*speed
        transformed_action[:,2] = torch.sin(phi)*speed
        transformed_action[:,3] = action[:, 3]*self.task_config.max_yawrate # rad/s

        if self.task_config.plot_cbf_constraint or self.task_config.penalize_cbf_constraint:
            cbf_values = self.collision_cbf.get_composite_cbf_value(
                position,
                disp = self.downsampled_lidar_displacements
            )
            cbf_derivatives = self.collision_cbf.get_h_derivative(
                position,
                transformed_action[:,0:3],
                disp= self.downsampled_lidar_displacements
            )
            cbf_constraint = cbf_derivatives + self.task_config.cbf_kappa_gain*cbf_values
            cbf_constraint = torch.clamp(cbf_constraint, max=0.0)
            if self.task_config.plot_cbf_constraint and wandb.run is not None:
                wandb.log({"CBF constraint(unfiltered)": cbf_constraint.mean()})
                wandb.log({"CBF values": cbf_values.mean()})
        else:
            cbf_constraint = torch.zeros_like(action[:,0])
        if self.task_config.filter_actions:
            safe_action = torch.zeros_like(transformed_action)
            alpha = self.task_config.cbf_kappa_gain
            safe_action[:,0:3] = self.collision_cbf.get_safe_input(transformed_action[:,0:3],
                                                            x = position,
                                                            disp = self.downsampled_lidar_displacements,
                                                            alpha = alpha)
            safe_action[:,3] = transformed_action[:,3]
            # Investigating how we can incorporate input constraints in the CBF is future work
            correction_mag = torch.linalg.vector_norm(safe_action[:,0:3] - transformed_action[:,0:3], dim=1)
            if wandb.run is not None:
                wandb.log({"Correction magnitude": correction_mag.mean()})
        else:
            safe_action = action
            correction_mag = torch.zeros_like(action[:,0])
        return safe_action, correction_mag, cbf_constraint
    def process_image_observation(self):
        image_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
        self.range_latents[:] = self.vae_model.encode(image_obs)
        # # comments to make sure the VAE does as expected
        # decoded_image = self.vae_model.decode(self.range_latents[0].unsqueeze(0))
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
        lidar_range_limits = (CBFLidarConfig.min_range,
                              CBFLidarConfig.max_range)
        lidar_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
        # Normalize the lidar observation(min-max normalization)
        with torch.no_grad():
            if self.task_config.vae_config.use_lidar_vae:
                means, logvar = self.vae_model(lidar_obs.unsqueeze(1))
                self.range_latents[:] = means + torch.randn_like(means)*torch.exp(0.5*logvar)
            else:
                self.range_latents[:] = lidar_obs
        # original_img = lidar_obs[0].cpu().numpy()
        # decoded_img = self.vae_model.decode(self.range_latents[0].unsqueeze(0))
        # decoded_img = decoded_img[0].squeeze(0).cpu().numpy()
        # if not hasattr(self, "img_ctr"):
        #     self.img_ctr = 0
        # self.img_ctr += 1
        # import matplotlib.pyplot as plt
        # plt.imsave(f"original{self.img_ctr}.png", original_img, vmin=0, vmax=1)
        # plt.imsave(f"decoded{self.img_ctr}.png", decoded_img, vmin=0, vmax=1)
        # Scale the lidar data to obtain the real range values
        lidar_obs_scaled = lidar_obs*(lidar_range_limits[1]-lidar_range_limits[0])\
            +lidar_range_limits[0]
        self.downsampled_lidar_displacements[:] = \
            self.lidar_cbf_data.get_displacements(lidar_obs_scaled)
    def step(self, actions):
        # this uses the action, gets observations
        # calculates rewards, returns tuples
        # In this case, the episodes that are terminated need to be
        # first reset, and the first obseration of the new episode
        # needs to be returned.
        transformed_action ,correction_size, cbf_constraint = \
            self.action_transformation_function(actions)
        logger.debug(f"raw_action: {actions[0]}, transformed action: {transformed_action[0]}")
        self.sim_env.step(actions=transformed_action)

        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        crashes = self.obs_dict["crashes"]
        successes = (
            torch.norm(self.target_position - self.obs_dict["robot_position"], dim=1) < 1.0
        )
        successes = torch.where(
            crashes > 0, torch.zeros_like(successes), successes
        )
        self.terminations[:] = torch.logical_or(crashes, successes)
        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        assert(torch.logical_and(crashes > 0, successes > 0).sum() == 0)
        timeouts = torch.where(
            self.truncations > 0, torch.logical_not(successes), torch.zeros_like(successes)
        )
        timeouts = torch.where(
            self.terminations > 0, torch.zeros_like(timeouts), timeouts
        )  # timeouts are not counted if there is a crash or success

        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = crashes
        self.rewards[:] = self.compute_rewards(
            self.obs_dict,
            successes,
            crashes,
            correction_size,
            cbf_constraint)

        # logger.info(f"Curricluum Level: {self.curriculum_level}")

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.logging_sanity_check(self.infos)
        self.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )
        # rendering happens at the post-reward calculation step since the newer measurement is required to be
        # sent to the RL algorithm as an observation and it helps if the camera image is updated then
        reset_envs = self.sim_env.post_reward_calculation_step(successes)
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)
        self.num_task_steps += 1
        # do stuff with the image observations here
        if self.task_config.vae_config.use_lidar_vae:
            self.process_lidar_observation()
        elif self.task_config.vae_config.use_camera_vae:
            self.process_image_observation()
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
        self.task_obs["observations"][:, 3] = self.obs_dict["robot_position"][:, 2]
        self.task_obs["observations"][:, 4:8] = self.obs_dict["robot_vehicle_orientation"]
        self.task_obs["observations"][:, 8:11] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 11:14] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 14:18] = self.obs_dict["robot_actions"]
        self.task_obs["observations"][:, 18:] = self.range_latents
        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def compute_rewards(self, 
                        obs_dict,
                        successes,
                        crashes, 
                        correction_size, 
                        cbf_constraint):
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        target_orientation = torch.zeros_like(robot_orientation, device=self.device)
        target_orientation[:, 3] = 1.0
        self.pos_error_vehicle_frame_prev[:] = self.pos_error_vehicle_frame
        self.pos_error_vehicle_frame[:] = quat_rotate_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        if not self.task_config.penalize_cbf_corrections:
            correction_size = torch.zeros_like(correction_size)
        if not self.task_config.penalize_cbf_constraint:
            cbf_constraint = torch.zeros_like(cbf_constraint)
        return compute_reward(
            self.pos_error_vehicle_frame,
            self.pos_error_vehicle_frame_prev,
            crashes,
            successes,
            obs_dict["robot_actions"],
            obs_dict["robot_prev_actions"],
            correction_size,
            cbf_constraint,
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
    successes,
    action,
    prev_action,
    cbf_correction_size,
    cbf_constraint,
    curriculum_progress_fraction,
    parameter_dict
):
    # type: (Tensor,Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, float, Dict[str, Tensor]) -> Tensor
    curriculum_reward_multiplier = (1.0 + curriculum_progress_fraction*1.0)
    dist = torch.norm(pos_error, dim=1)
    prev_dist_to_goal = torch.norm(prev_pos_error, dim=1)
    getting_closer_reward = parameter_dict["getting_closer_reward_multiplier"] * (
        prev_dist_to_goal*parameter_dict["gamma"] - dist
    )
    action_diff = action - prev_action
    x_diff_penalty = exponential_penalty_function(
        parameter_dict["x_action_diff_penalty_magnitude"],
        parameter_dict["x_action_diff_penalty_exponent"],
        action_diff[:, 0],
    )
    y_diff_penalty = exponential_penalty_function(
        parameter_dict["y_action_diff_penalty_magnitude"],
        parameter_dict["y_action_diff_penalty_exponent"],
        action_diff[:, 1],
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
    action_diff_penalty = x_diff_penalty + z_diff_penalty + y_diff_penalty + yawrate_diff_penalty
    # absolute action penalty
    x_absolute_penalty = exponential_penalty_function(
        parameter_dict["x_absolute_action_penalty_magnitude"],
        parameter_dict["x_absolute_action_penalty_exponent"],
        action[:, 0],
    )
    y_absolute_penalty = exponential_penalty_function(
        parameter_dict["y_absolute_action_penalty_magnitude"],
        parameter_dict["y_absolute_action_penalty_exponent"],
        action[:, 1],
    )
    z_absolute_penalty = exponential_penalty_function(
        parameter_dict["z_absolute_action_penalty_magnitude"],
        parameter_dict["z_absolute_action_penalty_exponent"],
        action[:, 2],
    )
    yawrate_absolute_penalty = exponential_penalty_function(
        parameter_dict["yawrate_absolute_action_penalty_magnitude"],
        parameter_dict["yawrate_absolute_action_penalty_exponent"],
        action[:, 3],
    )
    absolute_action_penalty = x_absolute_penalty +y_absolute_penalty +\
        z_absolute_penalty + yawrate_absolute_penalty
    # TODO: Consider using exponential penalty for the CBF correction size

    cbf_correction_penalty = parameter_dict["cbf_correction_penalty_magnitude"] \
        * cbf_correction_size
    cbf_constraint_penalty = parameter_dict["cbf_invariance_penalty_magnitude"] \
        * cbf_constraint
    total_action_penalty = action_diff_penalty + absolute_action_penalty + cbf_correction_penalty\
        + cbf_constraint_penalty
    # combined reward
    collision_penalty = crashes * parameter_dict["collision_penalty"]
    success_reward = successes * parameter_dict["success_reward"]
    reward =  getting_closer_reward + total_action_penalty + \
        success_reward + collision_penalty
    reward = reward * curriculum_reward_multiplier
    return reward
