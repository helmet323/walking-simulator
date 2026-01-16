import os
import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.biped_env import BipedEnv

# ---------------- Environment factory ----------------
def make_env():
    return BipedEnv(mode="gui")  # GUI mode for visualization

# ---------------- Get latest trained model ----------------
def get_latest_model(model_dir="models"):
    if not os.path.exists(model_dir):
        return None
    models = [os.path.join(model_dir,f) for f in os.listdir(model_dir) if f.endswith(".zip")]
    return max(models, key=os.path.getmtime) if models else None

# ---------------- Main ----------------
if __name__ == "__main__":

    latest = get_latest_model()
    if latest is None:
        raise FileNotFoundError("❌ No trained model found in models/")

    print(f"🎯 Using model: {latest}")

    # ---------------- Create VecEnv ----------------
    venv = DummyVecEnv([make_env])

    # ---------------- Load VecNormalize ----------------
    vec_file = latest.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(vec_file):
        venv = VecNormalize.load(vec_file, venv)
        print("✓ VecNormalize stats loaded")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
        print("⚠ No VecNormalize found — running unnormalized")

    venv.training = False
    venv.norm_reward = False

    # ---------------- Load model ----------------
    model = PPO.load(latest, env=venv)
    print("✓ Model loaded successfully")

    # ---------------- Viewer loop ----------------
    obs = venv.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)

            if np.any(dones):
                obs = venv.reset()

            time.sleep(1/240)  # match PyBullet physics timestep

    except KeyboardInterrupt:
        print("\n👋 Viewer closed by user")

    finally:
        venv.close()
