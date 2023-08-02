from stable_baselines3.common.callbacks import BaseCallback


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:

        self.logger.record("reward", self.training_env.get_attr("rew", 0)[0].item())
        return True