import os, glob
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.biped_env import BipedEnv
import functools

class LoggingCallback(BaseCallback):
    def __init__(self, model_dir="models", print_freq=5000, smooth_window=20, save_freq=50000):
        super().__init__()
        self.print_freq = print_freq
        self.smooth_window = smooth_window
        self.episode_rewards = []
        self.model_dir = model_dir
        self.save_freq = save_freq
        self.best_mean = -np.inf

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            for info in infos:
                ep = info.get("episode")
                if ep:
                    self.episode_rewards.append(ep["r"])

        if self.episode_rewards and self.num_timesteps % self.print_freq == 0:
            last = self.episode_rewards[-self.smooth_window:]
            avg = np.mean(last)
            print(f"[Step {self.num_timesteps}] Smoothed AvgReward (last {self.smooth_window}) = {avg:.2f}")
            # save best model
            if avg > self.best_mean:
                self.best_mean = avg
                path = os.path.join(self.model_dir, f"ppo_biped_best_{self.num_timesteps}")
                self.model.save(path)
                # try to save VecNormalize stats if present
                try:
                    venv = self.model.get_env()
                    if hasattr(venv, "save"):
                        venv.save(path + "_vecnormalize.pkl")
                except Exception:
                    pass
                print(f"Saved new best model to {path}.zip (mean {avg:.2f})")

        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.model_dir, f"ppo_biped_ckpt_{self.num_timesteps}")
            self.model.save(path)
            try:
                venv = self.model.get_env()
                if hasattr(venv, "save"):
                    venv.save(path + "_vecnormalize.pkl")
            except Exception:
                pass
            print(f"Periodic checkpoint saved: {path}.zip")

        return True

# Headless vectorized environment with normalization
def make_env(seed: int):
    def _init():
        return BipedEnv(mode="direct", debug=False, safe_reset=False, seed=seed)
    return _init

n_envs = 4
env = DummyVecEnv([make_env(i) for i in range(n_envs)])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Detect tensorboard availability for consistent model creation
try:
    import tensorboard  # noqa: F401
    tb_log = "./tensorboard_logs"
except Exception:
    tb_log = None
    print("TensorBoard not found; continuing without tensorboard logging.")

# Load latest model
model_files = sorted(glob.glob(os.path.join(model_dir, "ppo_biped_standing_*.zip")))
if model_files:
    latest = model_files[-1]
    print(f"Resuming from latest: {latest}")
    try:
        model = PPO.load(latest, env=env)
        # try to load VecNormalize stats if present
        vecfile = latest.replace('.zip', '_vecnormalize.pkl')
        if hasattr(env, 'load') and os.path.exists(vecfile):
            try:
                env.load(vecfile)
                print(f"Loaded VecNormalize stats from {vecfile}")
            except Exception:
                pass
    except ValueError as e:
        print("Saved model incompatible with current observation/action spaces:")
        print(e)
        print("Creating a new model instead. If you want to resume, ensure the environment's observation space matches the saved model.")
        model = PPO(
            "MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            tensorboard_log=tb_log,
        )
else:
    print("No existing model found. Creating new model.")
    # Enable tensorboard logging only if tensorboard is installed
    try:
        import tensorboard  # noqa: F401
        tb_log = "./tensorboard_logs"
    except Exception:
        tb_log = None
        print("TensorBoard not found; continuing without tensorboard logging.")

    model = PPO(
        "MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        tensorboard_log=tb_log,
    )

# Train
total_timesteps = 300_000
try:
    model.learn(total_timesteps=total_timesteps, callback=LoggingCallback(model_dir=model_dir, print_freq=5000, smooth_window=40, save_freq=50000))
except KeyboardInterrupt:
    print("Ctrl+C detected, saving model...")
finally:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(model_dir, f"ppo_biped_standing_{ts}")
    model.save(save_path)
    print(f"Model saved: {save_path}.zip")
    # Save normalization stats if using VecNormalize
    try:
        if hasattr(env, "save"):
            env.save(save_path + "_vecnormalize.pkl")
    except Exception:
        pass
    env.close()
