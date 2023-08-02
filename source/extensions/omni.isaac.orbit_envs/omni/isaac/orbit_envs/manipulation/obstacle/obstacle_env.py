# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gym.spaces
import math
import torch
import numpy as np

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematics
from omni.isaac.orbit.markers import PointMarker, StaticMarker
from omni.isaac.orbit.objects import RigidObject
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.math import random_orientation, sample_uniform, scale_transform
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager
from omni.isaac.orbit_envs.utils.data_collector.time_it import TimeItData, TimeIt

from pxr import PhysxSchema, PhysicsSchemaTools
from omni.physx import get_physx_simulation_interface, acquire_physx_interface

from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs

from .obstacle_cfg import RandomizationCfg, ReachEnvCfg


class ReachEnv(IsaacEnv):
    """Environment for reaching to desired pose for a single-arm manipulator."""

    def __init__(self, res: int = 128, image_type: str = "rgb", logdir: str = None, video: bool = False, cfg: ReachEnvCfg = None, headless: bool = False):
        # copy configuration
        self.cfg = cfg
        self.logdir = logdir
        self.step_count = 0
        # parse the configuration fosource/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/manipulation/reachm/reachm_env.pyRr controller configuration
        # note: controller decides the robot control mode
        self._pre_process_cfg()
        # create classes (these are called by the function :meth:`_design_scene`
        self.robot = SingleArmManipulator(cfg=self.cfg.robot)
        self.cube = RigidObject(cfg=self.cfg.cube)

        # subscribe to physics contact report event, this callback issued after each simulation step
        self._contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)

        # initialize the base class to setup the scene.
        super().__init__(self.cfg, headless=headless)
        # parse the configuration for information
        self._process_cfg()
        # initialize views for the cloned scenes
        self._initialize_views()

        # prepare the observation manager
        self._observation_manager = ReachObservationManager(class_to_dict(self.cfg.observations), self, self.device)
        # prepare the reward manager
        self._reward_manager = ReachRewardManager(
            class_to_dict(self.cfg.rewards), self, self.num_envs, self.dt, self.device
        )
        # print information about MDP
        print("[INFO] Observation Manager:", self._observation_manager)
        print("[INFO] Reward Manager: ", self._reward_manager)

        # compute the observation space
        num_obs = self._observation_manager._group_obs_dim["policy"][0]
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
        # compute the action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        print("[INFO]: Completed setting up the environment...")
        # Take an initial step to initialize the scene.
        self.sim.step()
        # -- fill up buffers
        self.robot.update_buffers(self.dt)
        self.cube.update_buffers(self.dt)

    """
    Implementation specifics.
    """

    def _design_scene(self):
        # ground plane
        kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-1.05)
        # table
        prim_utils.create_prim(self.template_env_ns + "/Table", usd_path=self.cfg.table.usd_path)
        # Cube
        self.cube.spawn(self.template_env_ns + "/Cube")
        # robot
        self.robot.spawn(self.template_env_ns + "/Robot")

        # setup debug visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            # create point instancer to visualize the goal points
            self._goal_markers = PointMarker("/Visuals/ee_goal", self.num_envs, radius=0.025)
            # create marker for viewing end-effector pose
            self._ee_markers = StaticMarker(
                "/Visuals/ee_current", self.num_envs, usd_path=self.cfg.marker.usd_path, scale=self.cfg.marker.scale
            )
            # create marker for viewing command (if task-space controller is used)
            if self.cfg.control.control_type == "inverse_kinematics":
                self._cmd_markers = StaticMarker(
                    "/Visuals/ik_command", self.num_envs, usd_path=self.cfg.marker.usd_path, scale=self.cfg.marker.scale
                )
        # return list of global prims
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: VecEnvIndices):
        # extract values from buffer
        print("RESET!!")
        print(env_ids)
        self.reset_buf[:] = 0
        # randomize the MDP
        # -- robot DOF state
        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
        # -- Cube DOF state
        # get the default root state
        root_state = self.cube.get_default_root_state(env_ids)
        # transform command from local env to world
        root_state[:, 0:3] += self.envs_positions[env_ids]
        # set the root state
        self.cube.set_root_state(root_state, env_ids=env_ids)
        # -- desired end-effector pose
        self._randomize_ee_desired_pose(env_ids, cfg=self.cfg.randomization.ee_desired_pose)
        # -- Reward logging
        # fill extras with episode information
        self.extras["episode"] = dict()
        # reset
        # -- rewards manager: fills the sums for terminated episodes
        self._reward_manager.reset_idx(env_ids, self.extras["episode"])
        # -- obs manager
        self._observation_manager.reset_idx(env_ids)
        # -- reset history
        self.previous_actions[env_ids] = 0
        # -- MDP reset
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        # controller reset
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller.reset_idx(env_ids)

    def _step_impl(self, actions: torch.Tensor):
        self.step_count += 1
        # construct time class
        time = TimeIt(self.logdir)
        time.data.start_time()
        # pre-step: set actions into buffer
        action = TimeItData("action", time.data.hierarchy_level + 1)
        simulation = TimeItData("simulation", time.data.hierarchy_level + 1)
        reward = TimeItData("reward", time.data.hierarchy_level + 1)

        for child in [action, simulation, reward]:
            time.data.children.append(child)

        # start action timer
        action.start_time()

        # pre-step: set actions into buffer
        self.actions = actions.clone().to(device=self.device)
        # transform actions based on controller
        if self.cfg.control.control_type == "inverse_kinematics":
            # set the controller commands
            self._ik_controller.set_command(self.actions)
            # compute the joint commands
            self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                self.robot.data.ee_state_w[:, 3:7],
                self.robot.data.ee_jacobian,
                self.robot.data.arm_dof_pos,
            )
            # offset actuator command with position offsets
            self.robot_actions[:, : self.robot.arm_num_dof] -= self.robot.data.actuator_pos_offset[
                :, : self.robot.arm_num_dof
            ]
        elif self.cfg.control.control_type == "default":
            self.robot_actions[:, : self.robot.arm_num_dof] = self.actions
        # perform physics stepping
        for x in range(self.cfg.control.decimation):
            # set actions into buffers and time
            action_apply = TimeItData("action_apply_" + str(x), action.hierarchy_level + 1)
            action.children.append(action_apply)
            action_apply.start_time()

            robot_ee = self.robot.data.ee_state_w[:, 0:3].cpu().detach().numpy()
            print("OLD END EFFECTOR {}".format(robot_ee))
            self.robot.apply_action(self.robot_actions)

            action_apply.end_time()
            # simulate and time
            simulate = TimeItData("simulate_" + str(x), simulation.hierarchy_level + 1)
            simulation.children.append(simulate)
            simulate.start_time()

            self.sim.step(render=self.enable_render)

            simulate.end_time()
            # check that simulation is playing
            if self.sim.is_stopped():
                return

        action.end_time()
        # post-step:
        # -- compute common buffers
        self.robot.update_buffers(self.dt)
        self.cube.update_buffers(self.dt)
        # -- compute MDP signals
        # reward
        reward.start_time()
        self.reward_buf = self._reward_manager.compute()
        robot_ee = self.robot.data.ee_state_w[:, 0:3].cpu().detach().numpy()
        print("END EFFECTOR {}".format(robot_ee))
        reward.end_time()
        rew = self.reward_buf.cpu().detach().numpy()
        if any(np.isnan(rew)) or any(np.isinf(rew)):
            print("is INVALID reward {}".format(rew))
            dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=torch.tensor([0]))
            self.robot.set_dof_state(dof_pos, dof_vel, env_ids=torch.tensor([0]))
            # get the default root state
            root_state = self.robot.get_default_root_state(torch.tensor([0]))
            # transform command from local env to world
            root_state[:, 0:3] += self.envs_positions[torch.tensor([0])]
            self.robot.set_root_state(root_state)
            self.robot.update_buffers(self.dt)
            robot_ee = self.robot.data.ee_state_w[:, 0:3].cpu().detach().numpy()
            print("NEW END EFFECTOR {}".format(robot_ee))
        else:
            print("is VALID reward {}".format(rew))
        # terminations
        self._check_termination()
        # -- store history
        self.previous_actions = self.actions.clone()

        # -- add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length
        # -- update USD visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            self._debug_vis()

        time.data.end_time()
        time.printing_data_handler(time.data)
        print("STILL HERE!")

    def _get_observations(self) -> VecEnvObs:
        # compute observations
        return self._observation_manager.compute()

    """
    Helper functions - Scene handling.
    """

    def _pre_process_cfg(self) -> None:
        """Pre processing of configuration parameters."""
        # set configuration for task-space controller
        if self.cfg.control.control_type == "inverse_kinematics":
            print("Using inverse kinematics controller...")
            # enable jacobian computation
            self.cfg.robot.data_info.enable_jacobian = True
            # enable gravity compensation
            self.cfg.robot.rigid_props.disable_gravity = True
            # set the end-effector offsets
            self.cfg.control.inverse_kinematics.position_offset = self.cfg.robot.ee_info.pos_offset
            self.cfg.control.inverse_kinematics.rotation_offset = self.cfg.robot.ee_info.rot_offset
        else:
            print("Using default joint controller...")

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # compute constants for environment
        self.dt = self.cfg.control.decimation * self.physics_dt  # control-dt
        self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / self.dt)

        # convert configuration parameters to torch
        # randomization
        # -- desired pose
        config = self.cfg.randomization.ee_desired_pose
        for attr in ["position_uniform_min", "position_uniform_max", "position_default", "orientation_default"]:
            setattr(config, attr, torch.tensor(getattr(config, attr), device=self.device, requires_grad=False))

    def _initialize_views(self) -> None:
        """Creates views and extract useful quantities from them."""
        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        self.sim.reset()

        # define views over instances
        self.robot.initialize(self.env_ns + "/.*/Robot")
        self.cube.initialize(self.env_ns + "/.*/Cube")

        # create controller
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller = DifferentialInverseKinematics(
                self.cfg.control.inverse_kinematics, self.robot.count, self.device
            )
            # note: we exclude gripper from actions in this env
            self.num_actions = self._ik_controller.num_actions
        elif self.cfg.control.control_type == "default":
            # note: we exclude gripper from actions in this env
            self.num_actions = self.robot.arm_num_dof

        # history
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        # robot joint actions
        self.robot_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        # commands
        self.ee_des_pose_w = torch.zeros((self.num_envs, 7), device=self.device)

        for env_num in range(0, self.num_envs):
            # apply contact report
            path = "/World/envs/env_" + str(env_num) + "/Cube"
            cubePrim = prim_utils.get_prim_at_path(path)
            PhysxSchema.PhysxContactReportAPI.Apply(cubePrim)

    def _debug_vis(self):
        # compute error between end-effector and command
        error = torch.sum(torch.square(self.ee_des_pose_w[:, :3] - self.robot.data.ee_state_w[:, 0:3]), dim=1)
        # set indices of the prim based on error threshold
        goal_indices = torch.where(error < 0.002, 1, 0)
        # apply to instance manager
        # -- goal
        self._goal_markers.set_world_poses(self.ee_des_pose_w[:, :3], self.ee_des_pose_w[:, 3:7])
        self._goal_markers.set_status(goal_indices)
        # -- end-effector
        self._ee_markers.set_world_poses(self.robot.data.ee_state_w[:, 0:3], self.robot.data.ee_state_w[:, 3:7])
        # -- task-space commands
        if self.cfg.control.control_type == "inverse_kinematics":
            # convert to world frame
            ee_positions = self._ik_controller.desired_ee_pos + self.envs_positions
            ee_orientations = self._ik_controller.desired_ee_rot
            # set poses
            self._cmd_markers.set_world_poses(ee_positions, ee_orientations)

    """
    Helper functions - MDP.
    """

    def _check_termination(self) -> None:
        ee_distance = torch.norm(self.robot.data.ee_state_w[:, 0:3] - self.cube.data.root_pos_w, dim=1)
        # extract values from buffer
        # compute resets
        self.reset_buf[:] = 0
        # -- episode length
        if self.cfg.terminations.episode_timeout:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)

        if self.cfg.terminations.object_collision:
            self.reset_buf = torch.where(ee_distance < 0.1, 1, self.reset_buf)

    def _randomize_ee_desired_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.EndEffectorDesiredPoseCfg):
        """Randomize the desired pose of the end-effector."""
        # -- desired object root position
        if cfg.position_cat == "default":
            # constant command for position
            self.ee_des_pose_w[env_ids, 0:3] = cfg.position_default
        elif cfg.position_cat == "uniform":
            # sample uniformly from box
            # note: this should be within in the workspace of the robot
            self.ee_des_pose_w[env_ids, 0:3] = sample_uniform(
                cfg.position_uniform_min, cfg.position_uniform_max, (len(env_ids), 3), device=self.device
            )
        else:
            raise ValueError(f"Invalid category for randomizing the desired object positions '{cfg.position_cat}'.")
        # -- desired object root orientation
        if cfg.orientation_cat == "default":
            # constant position of the object
            self.ee_des_pose_w[env_ids, 3:7] = cfg.orientation_default
        elif cfg.orientation_cat == "uniform":
            self.ee_des_pose_w[env_ids, 3:7] = random_orientation(len(env_ids), self.device)
        else:
            raise ValueError(
                f"Invalid category for randomizing the desired object orientation '{cfg.orientation_cat}'."
            )
        # transform command from local env to world
        self.ee_des_pose_w[env_ids, 0:3] += self.envs_positions[env_ids]

    def _on_contact_report_event(self, contact_headers, contact_data):
        for contact_header in contact_headers:
            print("Got contact header type: " + str(contact_header.type))
            print("Actor0: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0)))
            print("Actor1: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1)))
            print("Collider0: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0)))
            print("Collider1: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.collider1)))
            print("StageId: " + str(contact_header.stage_id))
            print("Number of contacts: " + str(contact_header.num_contact_data))

            contact_data_offset = contact_header.contact_data_offset
            num_contact_data = contact_header.num_contact_data

            for index in range(contact_data_offset, contact_data_offset + num_contact_data, 1):
                print("Contact:")
                print("Contact position: " + str(contact_data[index].position))
                print("Contact normal: " + str(contact_data[index].normal))
                print("Contact impulse: " + str(contact_data[index].impulse))
                print("Contact separation: " + str(contact_data[index].separation))
                print("Contact faceIndex0: " + str(contact_data[index].face_index0))
                print("Contact faceIndex1: " + str(contact_data[index].face_index1))
                print("Contact material0: " + str(PhysicsSchemaTools.intToSdfPath(contact_data[index].material0)))


class ReachObservationManager(ObservationManager):
    """Reward manager for single-arm reaching environment."""

    def arm_dof_pos_normalized(self, env: ReachEnv):
        """DOF positions for the arm normalized to its max and min ranges."""
        return scale_transform(
            env.robot.data.arm_dof_pos,
            env.robot.data.soft_dof_pos_limits[:, :7, 0],
            env.robot.data.soft_dof_pos_limits[:, :7, 1],
        )

    def arm_dof_vel(self, env: ReachEnv):
        """DOF velocity of the arm."""
        return env.robot.data.arm_dof_vel

    def ee_position(self, env: ReachEnv):
        """Current end-effector position of the arm."""
        return env.robot.data.ee_state_w[:, :3] - env.envs_positions

    def ee_position_command(self, env: ReachEnv):
        """Desired end-effector position of the arm."""
        return env.ee_des_pose_w[:, :3] - env.envs_positions

    def actions(self, env: ReachEnv):
        """Last actions provided to env."""
        return env.actions

    def object_position(self, env: ReachEnv):
        return env.cube.data.root_pos_w - env.envs_positions


class ReachRewardManager(RewardManager):
    """Reward manager for single-arm reaching environment."""

    def tracking_robot_position_l2(self, env: ReachEnv):
        """Penalize tracking position error using L2-kernel."""
        # compute error
        return torch.sum(torch.square(env.ee_des_pose_w[:, :3] - env.robot.data.ee_state_w[:, 0:3]), dim=1)

    def tracking_robot_position_exp(self, env: ReachEnv, sigma: float):
        """Penalize tracking position error using exp-kernel."""
        # compute error
        error = torch.sum(torch.square(env.ee_des_pose_w[:, :3] - env.robot.data.ee_state_w[:, 0:3]), dim=1)
        return torch.exp(-error / sigma)

    def penalizing_robot_dof_velocity_l2(self, env: ReachEnv):
        """Penalize large movements of the robot arm."""
        return torch.sum(torch.square(env.robot.data.arm_dof_vel), dim=1)

    def penalizing_robot_dof_acceleration_l2(self, env: ReachEnv):
        """Penalize fast movements of the robot arm."""
        return torch.sum(torch.square(env.robot.data.dof_acc), dim=1)

    def penalizing_action_rate_l2(self, env: ReachEnv):
        """Penalize large variations in action commands."""
        return torch.sum(torch.square(env.actions - env.previous_actions), dim=1)
