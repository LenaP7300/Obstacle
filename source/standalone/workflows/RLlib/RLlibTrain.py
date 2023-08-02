from omni.isaac.kit import SimulationApp
import argparse

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


from omni.isaac.orbit_envs.utils import load_default_env_cfg
from omni.isaac.orbit_envs.utils.wrappers.rllib import RLlibVecEnvWrapper

import ray
import gym

from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
import ray.rllib.policy.torch_policy as TPolicy


task_name = "Isaac-MyLift-Franka-v0"
cfg = load_default_env_cfg(task_name)


def main():
    # create environment
    register_env("Isaac-MyLift-Franka-v0", lambda _: gym.make(task_name, cfg=cfg))

    worker = RolloutWorker(
        env_creator=lambda _: gym.make(task_name, cfg=cfg),
        default_policy_class=TPolicy)

    config = (  # 1. Configure the algorithm,
        PPOConfig()
        .environment("Isaac-MyLift-Franka-v0")
        .framework("torch")  # I only changed here (tf2 => torch)
        .training(model={"fcnet_hiddens": [64, 64]})
        .evaluation(evaluation_num_workers=1)
        .rollouts(num_rollout_workers=10,
                  num_envs_per_worker=1,
                  enable_connectors=True,
                  remote_worker_envs=True)
    )

    config.environment(disable_env_checking=True)

    ray.init()
    # algo = config.build()  # 2. build the algorithm,

    for _ in range(5):
        # print(algo.train())  # 3. train it,
        print(worker.observation_space.sample())
        print(worker.action_space.sample())
        # print(worker.sample())

    # algo.evaluate()

if __name__ == "__main__":
    # run main function
    main()
    # close simulation
    simulation_app.close()
