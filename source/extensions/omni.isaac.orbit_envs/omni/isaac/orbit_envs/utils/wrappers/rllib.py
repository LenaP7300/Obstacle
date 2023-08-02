# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`IsaacEnv` instance to RLlib vectorized environment.

The following example shows how to wrap an environment for RLlib:

.. code-block:: python

    from omni.isaac.orbit_envs.utils.wrappers.rllib import RLlibVecEnvWrapper

    env = RLlibVecEnvWrapper(env)

"""

import logging
import gym
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Callable, Tuple, Union

# rllib
from ray.rllib.env.vector_env import VectorEnv, VectorEnvWrapper
from ray.rllib.utils.typing import EnvObsType, EnvInfoDict, EnvActionType, EnvType
from ray.util import log_once

from omni.isaac.orbit_envs.isaac_env import IsaacEnv

__all__ = ["RLlibVecEnvWrapper"]
logger = logging.getLogger(__name__)


"""
Vectorized environment wrapper.
"""


class RLlibVecEnvWrapper(VectorEnv):
    """Wraps around Isaac Orbit environment for RLlib.

    Isaac Sim internally implements a vectorized environment. However, since it is
    still considered a single environment instance, RLlib tries to wrap
    around it using the :class:`DummyVecEnv`. This is only done if the environment
    is not inheriting from their :class:`VecEnv`. Thus, this class thinly wraps
    over the environment from :class:`IsaacEnv`.

    We also add monitoring functionality that computes the un-discounted episode
    return and length. This information is added to the info dicts under key `episode`.

    In contrast to Isaac Orbit environment, stable-baselines expect the following:

    1. numpy datatype for MDP signals
    2. a list of info dicts for each sub-environment (instead of a dict)
    3. when environment has terminated, the observations from the environment should correspond
       to the one after reset. The "real" final observation is passed using the info dicts
       under the key ``terminal_observation``.

    Warning:
        By the nature of physics stepping in Isaac Sim, it is not possible to forward the
        simulation buffers without performing a physics step. Thus, reset is performed only
        at the start of :meth:`step()` function before the actual physics step is taken.
        Thus, the returned observations for terminated environments is still the final
        observation and not the ones after the reset.

    Reference:
        https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
        https://stable-baselines3.readthedocs.io/en/master/common/monitor.html
    """

    def __init__(self, env: IsaacEnv):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`IsaacEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, IsaacEnv):
            raise ValueError(f"The environment must be inherited from IsaacEnv. Environment type: {type(env)}")
        # initialize the wrapper
        gym.Wrapper.__init__(self, env)
        # self.env = VectorEnvWrapper(env)
        VectorEnv.__init__(self, self.env.observation_space, self.env.action_space, self.env.num_envs)

        # add buffer for logging episodic information
        self._ep_rew_buf = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)
        self._ep_len_buf = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)

    """
    Properties
    """

    def get_episode_rewards(self) -> List[float]:
        """Returns the rewards of all the episodes."""
        return self._ep_rew_buf.cpu().tolist()

    def get_episode_lengths(self) -> List[int]:
        """Returns the number of time-steps of all the episodes."""
        return self._ep_len_buf.cpu().tolist()

    """
    Operations - MDP
    """

    '''def vector_reset(
        self, *, seeds: Optional[List[int]] = None, options: Optional[List[dict]] = None
    ) -> Tuple[List[EnvObsType], List[EnvInfoDict]]:
        seeds = seeds or [None] * self.num_envs
        options = options or [None] * self.num_envs
        # Use reset_at(index) to restart and retry until
        # we successfully create a new env.
        obs_dict, infos_dict = self.env.reset(seed=seeds, options=options)
        # convert data types to numpy depending on backen
        resetted_obs = self._process_obs(obs_dict)
        resetted_infos = infos_dict.detach().cpu().numpy()
        return resetted_obs, {}

    def reset_at(
        self,
        index: Optional[int] = None,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Union[EnvObsType, Exception], Union[EnvInfoDict, Exception]]:
        if index is None:
            index = 0
        try:
            obs_dict, infos_dict = self.envs[index].reset(seed=seed, options=options)
            obs = obs_dict.detach().cpu().numpy()
            infos = infos_dict.detach().cpu().numpy()

        except Exception as e:
            if self.restart_failed_sub_environments:
                logger.exception(e.args[0])
                self.restart_at(index)
                obs, infos = e, {}
            else:
                raise e

        return obs, infos
    
    def vector_step(
        self, actions: List[EnvActionType]
    ) -> Tuple[
        List[EnvObsType], List[float], List[bool], List[bool], List[EnvInfoDict]
    ]:
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = (
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(self.num_envs):
            try:
                results = self.envs[i].step(actions[i])
            except Exception as e:
                if self.restart_failed_sub_environments:
                    logger.exception(e.args[0])
                    self.restart_at(i)
                    results = e, 0.0, True, True, {}
                else:
                    raise e

            obs, reward, terminated, truncated, info = results

            if not isinstance(info, dict):
                raise ValueError(
                    "Info should be a dict, got {} ({})".format(info, type(info))
                )
            obs_batch.append(obs)
            reward_batch.append(reward)
            terminated_batch.append(terminated)
            truncated_batch.append(truncated)
            info_batch.append(info)
        return obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch'''

    def reset(self):  # noqa: D102
        obs_dict = self.env.reset()
        # convert data types to numpy depending on backend
        return self._process_obs(obs_dict)

    def step(self, actions: np.ndarray):  # noqa: D102
        # convert input to numpy array
        actions = np.asarray(actions)
        # convert to tensor
        actions = torch.from_numpy(actions).to(device=self.env.device)
        # record step information
        obs_dict, rew, dones, extras = self.env.step(actions)
        self.obs_dict = obs_dict
        self.rew = rew

        # update episode un-discounted return and length
        self._ep_rew_buf += rew
        self._ep_len_buf += 1
        reset_ids = (dones > 0).nonzero(as_tuple=False)

        # convert data types to numpy depending on backend
        # Note: IsaacEnv uses torch backend (by default).
        obs = self._process_obs(obs_dict)
        rew = rew.cpu().numpy()
        dones = dones.cpu().numpy()
        # convert extra information to list of dicts
        infos = self._process_extras(obs, dones, extras, reset_ids)

        # reset info for terminated environments
        self._ep_rew_buf[reset_ids] = 0
        self._ep_len_buf[reset_ids] = 0

        return obs, rew, dones, infos

    """
    Unused methods.
    """

    def step_async(self, actions):  # noqa: D102
        self._async_actions = actions

    def step_wait(self):  # noqa: D102
        return self.step(self._async_actions)

    def get_attr(self, attr_name, indices):  # noqa: D102
        return getattr(self, attr_name)

    def set_attr(self, attr_name, value, indices=None):  # noqa: D102
        raise NotImplementedError

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):  # noqa: D102
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):  # noqa: D102
        raise NotImplementedError

    def get_images(self):  # noqa: D102
        raise NotImplementedError

    """
    Helper functions.
    """

    def _process_obs(self, obs_dict) -> np.ndarray:
        """Convert observations into NumPy data type."""
        # Sb3 doesn't support asymmetric observation spaces, so we only use "policy"
        obs = obs_dict["policy"]
        # Note: IsaacEnv uses torch backend (by default).
        if self.env.sim.backend == "torch":
            if isinstance(obs, dict):
                for key, value in obs.items():
                    obs[key] = value.detach().cpu().numpy()
            else:
                obs = obs.detach().cpu().numpy()
        elif self.env.sim.backend == "numpy":
            pass
        else:
            raise NotImplementedError(f"Unsupported backend for simulation: {self.env.sim.backend}")
        return obs

    def _process_extras(self, obs, dones, extras, reset_ids) -> List[Dict[str, Any]]:
        """Convert miscellaneous information into dictionary for each sub-environment."""
        # create empty list of dictionaries to fill
        infos: List[Dict[str, Any]] = [dict.fromkeys(extras.keys()) for _ in range(self.env.num_envs)]
        # fill-in information for each sub-environment
        # Note: This loop becomes slow when number of environments is large.
        for idx in range(self.env.num_envs):
            # fill-in episode monitoring info
            if idx in reset_ids:
                infos[idx]["episode"] = dict()
                infos[idx]["episode"]["r"] = float(self._ep_rew_buf[idx])
                infos[idx]["episode"]["l"] = float(self._ep_len_buf[idx])
            else:
                infos[idx]["episode"] = None
            # fill-in information from extras
            for key, value in extras.items():
                # 1. remap the key for time-outs for what SB3 expects
                # 2. remap extra episodes information safely
                # 3. for others just store their values
                if key == "time_outs":
                    infos[idx]["TimeLimit.truncated"] = bool(value[idx])
                elif key == "episode":
                    # only log this data for episodes that are terminated
                    if infos[idx]["episode"] is not None:
                        for sub_key, sub_value in value.items():
                            infos[idx]["episode"][sub_key] = sub_value
                else:
                    infos[idx][key] = value[idx]
            # add information about terminal observation separately
            if dones[idx] == 1:
                # extract terminal observations
                if isinstance(obs, dict):
                    terminal_obs = dict.fromkeys(obs.keys())
                    for key, value in obs.items():
                        terminal_obs[key] = value[idx]
                else:
                    terminal_obs = obs[idx]
                # add info to dict
                infos[idx]["terminal_observation"] = terminal_obs
            else:
                infos[idx]["terminal_observation"] = None
        # return list of dictionaries
        return infos
