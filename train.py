import os
import sys
from datetime import datetime
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn

from envs.biped_env import BipedEnv

# ========================== Callback ==========================
class LoggingCallback(BaseCallback):
    """Logs rewards, saves best models & periodic checkpoints"""
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

        # ---------- Smoothed reward logging ----------
        if self.ep_rewards and self.num_timesteps % self.print_freq == 0:
            last = self.ep_rewards[-self.smooth_window:]
            avg = float(np.mean(last))
            print(f"[Step {self.num_timesteps}] Smoothed Reward = {avg:.2f}")

            if avg > self.best_mean:
                self.best_mean = avg
                path = os.path.join(self.model_dir, "ppo_biped_best")
                self.model.save(path)
                try:
                    self.training_env.save(path + "_vecnormalize.pkl")
                except Exception:
                    pass
                print(f"✓ Saved NEW BEST model ({avg:.2f})")

        # ---------- Periodic checkpoint ----------
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.model_dir, f"ppo_biped_ckpt_{self.num_timesteps}")
            self.model.save(path)
            try:
                self.training_env.save(path + "_vecnormalize.pkl")
            except Exception:
                pass
            print(f"✓ Checkpoint saved: {path}.zip")

        return True

# ========================== Environment Factory ==========================
def make_env(seed: int):
    def _init():
        env = BipedEnv(mode="direct", seed=seed)
        env.reset(seed=seed)
        return env
    return _init

# ========================== Utility: Latest Model ==========================
def get_latest_model(model_dir="models"):
    if not os.path.exists(model_dir):
        return None
    models = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".zip")
    ]
    return max(models, key=os.path.getmtime) if models else None

# ========================== Main Training ==========================
if __name__ == "__main__":
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # ---------- Fresh training flag ----------
    fresh = "--fresh" in sys.argv

    # ---------- Create vectorized environments ----------
    n_envs = 4
    venv = DummyVecEnv([make_env(i) for i in range(n_envs)])

    # ---------- Load latest model & VecNormalize ----------
    latest_model = get_latest_model(model_dir)
    model = None

    if latest_model and not fresh:
        print(f"Found latest model: {latest_model}")
        vec_file = latest_model.replace(".zip", "_vecnormalize.pkl")
        # ---------- Try loading VecNormalize ----------
        if os.path.exists(vec_file):
            try:
                temp_vec = VecNormalize.load(vec_file, venv)
                if temp_vec.observation_space.shape == venv.observation_space.shape:
                    venv = temp_vec
                    print("✓ Loaded VecNormalize stats")
                else:
                    print("⚠ VecNormalize shape mismatch, creating new")
                    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
            except Exception:
                print("⚠ Failed to load VecNormalize, creating new")
                venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        else:
            print("⚠ No VecNormalize stats found — creating new")
            venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

        # ---------- Try loading PPO model ----------
        try:
            model = PPO.load(latest_model, env=venv)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"⚠ Model incompatible, creating new: {e}")
            model = None
    else:
        print("🚀 Fresh training: ignoring old models & VecNormalize")
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # ---------- Training mode ----------
    venv.training = True
    venv.norm_reward = True

    # ---------- Create new PPO model if needed ----------
    if model is None:
        print("🚀 Creating NEW PPO model")
        lr_schedule = get_linear_fn(start=3e-4, end=1e-5, end_fraction=1.0)

        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        )

        model = PPO(
            policy="MlpPolicy",
            env=venv,
            learning_rate=lr_schedule,
            n_steps=2048,
            batch_size=128,
            n_epochs=15,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.15,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )

    # ---------- Train ----------
    try:
        model.learn(
            total_timesteps=1_000_000,
            callback=LoggingCallback(model_dir=model_dir, print_freq=5000, save_freq=50_000)
        )
    except KeyboardInterrupt:
        print("⚠ Training interrupted by user")

    # ---------- Save final model ----------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(model_dir, f"ppo_biped_{ts}")
    model.save(final_path)
    venv.save(final_path + "_vecnormalize.pkl")
    print(f"✓ Final model saved: {final_path}.zip")
    venv.close()
