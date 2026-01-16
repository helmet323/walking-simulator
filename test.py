import os
import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.biped_env import BipedEnv


# --------------------------------------------------
def make_env():
    return BipedEnv(mode="gui", debug=False)


def get_latest_model(model_dir="models"):
    models = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".zip")
    ]
    if not models:
        return None
    models.sort(key=os.path.getmtime)
    return models[-1]


# --------------------------------------------------
if __name__ == "__main__":

    latest = get_latest_model()
    if latest is None:
        raise FileNotFoundError("❌ No trained model found in models/")

    print(f"🎯 Using model: {latest}")

    # --------------------------------------------------
    # Create GUI env FIRST (never closes)
    # --------------------------------------------------
    env = DummyVecEnv([make_env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )

    # --------------------------------------------------
    # Load model SAFELY
    # --------------------------------------------------
    try:
        model = PPO.load(latest, env=env)
    except ValueError as e:
        print("\n❌ MODEL / ENV MISMATCH")
        print(e)
        print("\n⚠ Retrain required. GUI kept open.")

        env.reset()
        while True:
            time.sleep(1)

    # --------------------------------------------------
    # Load VecNormalize stats if compatible
    # --------------------------------------------------
    vec_file = latest.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(vec_file):
        try:
            env.load(vec_file)
            print("✓ VecNormalize loaded")
        except Exception:
            print("⚠ VecNormalize mismatch (ignored)")

    print("🎥 Running visualization — Ctrl+C to exit")

    obs = env.reset()

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            if np.any(dones):
                obs = env.reset()

            time.sleep(1 / 240)

    except KeyboardInterrupt:
        print("\nExiting viewer")

    finally:
        env.close()
