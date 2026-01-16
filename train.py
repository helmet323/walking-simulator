import os
import glob
from datetime import datetime
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from envs.biped_env import BipedEnv

# =====================================================
# Callback for logging & checkpointing
# =====================================================
class LoggingCallback(BaseCallback):
    def __init__(self, model_dir, print_freq=5000, save_freq=50000, smooth_window=20):
        super().__init__()
        self.model_dir = model_dir
        self.print_freq = print_freq
        self.save_freq = save_freq
        self.smooth_window = smooth_window
        self.best_mean = -np.inf
        self.ep_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep:
                self.ep_rewards.append(ep["r"])

        # Logging smoothed reward
        if self.ep_rewards and self.num_timesteps % self.print_freq == 0:
            last_rewards = self.ep_rewards[-self.smooth_window:]
            avg = np.mean(last_rewards)
            print(f"[Step {self.num_timesteps}] Smoothed AvgReward({len(last_rewards)}) = {avg:.2f}")

            # Save best model
            if avg > self.best_mean:
                self.best_mean = avg
                path = os.path.join(self.model_dir, "ppo_biped_best")
                self.model.save(path)
                try:
                    self.training_env.save(path + "_vecnormalize.pkl")
                except Exception:
                    pass
                print(f"✓ Saved new BEST model ({avg:.2f})")

        # Periodic checkpoint
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.model_dir, f"ppo_biped_ckpt_{self.num_timesteps}")
            self.model.save(path)
            try:
                self.training_env.save(path + "_vecnormalize.pkl")
            except Exception:
                pass
            print(f"✓ Periodic checkpoint saved: {path}.zip")

        return True

# =====================================================
# Environment factory
# =====================================================
def make_env(seed: int):
    def _init():
        return BipedEnv(mode="direct", seed=seed)
    return _init

# =====================================================
# Utility: get latest model by modification time
# =====================================================
def get_latest_model(model_dir="models"):
    if not os.path.exists(model_dir):
        return None
    models = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".zip")
    ]
    if not models:
        return None
    models.sort(key=os.path.getmtime)
    return models[-1]

# =====================================================
# Main training
# =====================================================
if __name__ == "__main__":
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Multi-env setup
    n_envs = 4
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Load latest model if available
    latest_model = get_latest_model(model_dir)
    model = None

    if latest_model:
        print(f"Found latest model: {latest_model}")
        try:
            model = PPO.load(latest_model, env=env)
            vec_file = latest_model.replace(".zip", "_vecnormalize.pkl")
            if os.path.exists(vec_file):
                env.load(vec_file)
                print("✓ Loaded VecNormalize stats")
        except Exception as e:
            print(f"⚠ Model incompatible with current environment. Reason: {e}")
            print("Creating new model instead...")
            model = None

    # Create new model if none loaded
    if model is None:
        print("Creating NEW PPO model")
        model = PPO(
            "MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
        )

    # Train with logging callback
    try:
        model.learn(
            total_timesteps=300_000,
            callback=LoggingCallback(model_dir=model_dir, print_freq=5000, save_freq=50_000),
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")

    # Save final model & VecNormalize stats
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(model_dir, f"ppo_biped_{ts}")
    model.save(final_path)
    try:
        env.save(final_path + "_vecnormalize.pkl")
    except Exception:
        pass

    print(f"✓ Final model saved: {final_path}.zip")
    env.close()
