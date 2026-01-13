import os, glob
import time
from envs.biped_env import BipedEnv
from stable_baselines3 import PPO

# GUI environment for testing
env = BipedEnv(mode="gui", debug=False, safe_reset=False)

# Load latest model
model_dir = "models"
model_files = sorted(glob.glob(os.path.join(model_dir, "ppo_biped_standing_*.zip")))
if model_files:
    latest = model_files[-1]
    print(f"Testing latest model: {latest}")
    model = PPO.load(latest, env=env)
else:
    raise FileNotFoundError("No saved model found to test.")

try:
    for ep in range(10):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total += reward
            time.sleep(1/240)
        print(f"Episode {ep+1} finished, reward={total:.2f}")
finally:
    env.close()
