# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3."""

"""Launch Isaac Sim Simulator first."""


import argparse
import numpy as np
import os
import wandb
from wandb.integration.sb3 import WandbCallback
import cv2
import shutil
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([4, 1, 128, 128], dtype=torch.float, device='cuda', requires_grad=True).to(memory_format=torch.channels_last)
net = torch.nn.Conv2d(1, 32, kernel_size=[8, 8], padding=[0, 0], stride=[4, 4], dilation=[1, 1], groups=1)
net = net.cuda().float().to(memory_format=torch.channels_last)
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--agent", type=str, default=None, help="model to load")
parser.add_argument("--image_type", type=str, default="rgb", help="image type of observation")
parser.add_argument("--video", action="store_true", help="record video of observation")
parser.add_argument("--res", type=int, default=0, help="image resolution")
parser.add_argument("--only_image", action="store_true", help="take only image observation for MultiInput-environments")
parser.add_argument("--wandb", action="store_true", help="record progress with wandb")
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
# load cheaper kit config in headless
if args_cli.headless:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
else:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
# launch the simulator
simulation_app = SimulationApp(config, experience=app_experience)

"""Rest everything follows."""


import gym
import os
from datetime import datetime

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import parse_env_cfg
from omni.isaac.orbit_envs.utils.wrappers.sb3 import Sb3VecEnvWrapper

from config import parse_sb3_cfg


def main():
    # one of my environments?
    myenvironment = bool(args_cli.task == "Isaac-MultiReach-Franka-v0"
                         or args_cli.task == "Isaac-MultiLift-Franka-v0"
                         or args_cli.task == "Isaac-CnnReach-Franka-v0"
                         or args_cli.task == "Isaac-CnnLift-Franka-v0"
                         or args_cli.task == "Isaac-CnnCartpole-v0"
                         or args_cli.task == "Isaac-Reach-Franka-v0"
                         or args_cli.task == "Isaac-Lift-Franka-v0"
                         or args_cli.task == "Isaac-ReachM-Franka-v0"
                         or args_cli.task == "Isaac-Obstacle-Franka-v0"
                         or args_cli.task == "Isaac-CnnReachM-Franka-v0")
    multienv = bool(args_cli.task == "Isaac-MultiReach-Franka-v0"
                    or args_cli.task == "Isaac-MultiLift-Franka-v0"
                    or args_cli.task == "Isaac-MultiReachM-Franka-v0")

    """Train with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg = parse_sb3_cfg(args_cli.task)
    # override configuration with command line arguments
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed

    # directory for logging into
    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%b%d_%H-%M-%S"))
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    wandb_config = {
        "policy_type": policy_arch,
        "total_timesteps": n_timesteps,
        "env_name": args_cli.task,
    }
    if args_cli.wandb:
        run = wandb.init(
            project="sb3",
            config=wandb_config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

    # create isaac environment
    if myenvironment:
        env = gym.make(args_cli.task,
                       res=args_cli.res,
                       image_type=args_cli.image_type,
                       logdir=log_dir,
                       video=args_cli.video,
                       cfg=env_cfg,
                       headless=args_cli.headless)
    elif multienv:
        env = gym.make(args_cli.task,
                       only_image=args_cli.only_image,
                       res=args_cli.res,
                       image_type=args_cli.image_type,
                       logdir=log_dir,
                       video=args_cli.video,
                       cfg=env_cfg,
                       headless=args_cli.headless)
    else:
        env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)
    # wrap around environment for stable baselines

    env = Sb3VecEnvWrapper(env)
    # set the seed
    env.seed(seed=agent_cfg["seed"])

    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # create agent from stable baselines
    if args_cli.agent is None:
        agent = PPO(policy_arch, env, verbose=1, **agent_cfg)
    else:
        agent = PPO.load(args_cli.agent, env=env, **agent_cfg)
    # configure the logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # callbacks for agent
    if myenvironment and args_cli.headless is False:
        if args_cli.wandb:
            callback = [
                WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ),
                CheckpointCallback(save_freq=3000, save_path=log_dir, name_prefix="model", verbose=2),
            ]
        else:
            callback = [
                CheckpointCallback(save_freq=3000, save_path=log_dir, name_prefix="model", verbose=2),
            ]
    else:
        if args_cli.wandb:
            callback = [
                WandbCallback(
                    gradient_save_freq=3000,
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ),
                CheckpointCallback(save_freq=3000, save_path=log_dir, name_prefix="model", verbose=2),
            ]
        else:
            callback = [
                CheckpointCallback(save_freq=3000, save_path=log_dir, name_prefix="model", verbose=2),
            ]

    print(agent.policy)

    # train the agent
    agent.learn(total_timesteps=n_timesteps, callback=callback)
    # save the final model
    agent.save(os.path.join(log_dir, "model"))

    # close the simulator
    env.close()

    # make video
    if args_cli.video:
        image_folder = log_dir + '/images'
        video_name = log_dir + '/video.avi'

        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 1, (height, width))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
        shutil.rmtree(image_folder)

    if args_cli.wandb:
        run.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()
