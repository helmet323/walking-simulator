import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import random

class BipedEnv(gym.Env):
    """
    Biped environment for RL:
    - Tracks episode reward & length
    - Safe reset
    - Encourages knee use
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}

    def __init__(self, mode="direct", debug=False, safe_reset=False, seed=None):
        super().__init__()
        self.mode = mode
        self.debug = debug
        self.safe_reset = safe_reset
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if self.mode == "gui":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        p.setTimeStep(1/240)

        self.plane_id = p.loadURDF("plane.urdf")

        self.biped_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "biped.urdf")
        if not os.path.exists(self.biped_path):
            raise FileNotFoundError(f"Cannot find biped URDF: {self.biped_path}")

        self.biped_id = p.loadURDF(self.biped_path, [0,0,1.0], useFixedBase=False)

        self.LEFT_HIP = 0
        self.LEFT_KNEE = 1
        self.RIGHT_HIP = 2
        self.RIGHT_KNEE = 3
        self.joint_indices = [self.LEFT_HIP, self.LEFT_KNEE, self.RIGHT_HIP, self.RIGHT_KNEE]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # observations: 4 joint angles, 4 joint velocities, base linear vel (3), base angular vel (3), pitch, height, left_contact, right_contact
        obs_low = np.array(
            [-np.pi]*4 + [-10]*4 + [-10.0]*3 + [-10.0]*3 + [-np.pi] + [0.0] + [0,0], dtype=np.float32
        )
        # standing detection and success reward
        self.standing_counter = 0
        self.standing_required_steps = 50
        self.standing_bonus = 5.0
        obs_high = np.array(
            [np.pi]*4 + [10]*4 + [10.0]*3 + [10.0]*3 + [np.pi] + [2.0] + [1,1], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Maximum change in joint target per simulation step (radians)
        self.max_joint_change = 0.02
        self.episode_reward = 0.0
        self.episode_length = 0

    # -----------------------------
    def reset(self, seed=None, options=None):

        # Full reset to a deterministic starting state
        p.resetSimulation()
        p.setGravity(0,0,-9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        self.biped_id = p.loadURDF(self.biped_path, [0,0,1.0], useFixedBase=False)

        tilt = random.uniform(-0.03, 0.03)
        p.resetBasePositionAndOrientation(self.biped_id, [0,0,1.0], p.getQuaternionFromEuler([tilt,0,0]))

        for i in self.joint_indices:
            p.resetJointState(self.biped_id, i, 0.0, 0.0)
            p.setJointMotorControl2(self.biped_id, i, p.POSITION_CONTROL, 0.0, force=200)

        self.episode_reward = 0.0
        self.episode_length = 0

        # previous action targets used for smoothing
        self.prev_targets = np.zeros(len(self.joint_indices), dtype=np.float32)

        obs = self._get_obs()
        # reset standing detector and prev targets
        self.standing_counter = 0
        if self.debug:
            print("[RESET] obs:", obs)
        return obs, {}

    # -----------------------------
    def step(self, action):
        # Map actions to joint target deltas and smooth using previous targets
        action = np.array(action, dtype=np.float32)
        # scale how much each action can change joint target per step (radians)
        target_delta = np.clip(action, -1.0, 1.0) * 0.06
        target_positions = self.prev_targets + target_delta
        for i, j in enumerate(self.joint_indices):
            cur = p.getJointState(self.biped_id, j)[0]
            desired = float(np.clip(target_positions[i], -np.pi, np.pi))
            new_target = np.clip(desired, cur - self.max_joint_change, cur + self.max_joint_change)
            # increase force to allow posture corrections
            p.setJointMotorControl2(self.biped_id, j, p.POSITION_CONTROL, new_target, force=350)
            self.prev_targets[i] = new_target

        p.stepSimulation()
        obs = self._get_obs()

        # unpack obs consistently
        angles = obs[0:4]
        vels = obs[4:8]
        base_linvel = obs[8:11]
        base_angvel = obs[11:14]
        torso_pitch = obs[14]
        torso_height = obs[15]

        # Reward shaping: stronger uprightness, height, knee use, and stability penalties
        upright_reward = float(np.clip(np.cos(torso_pitch), 0.0, 1.0)) * 1.5
        height_reward = max(0.0, torso_height - 0.9) * 1.0
        knee_bonus = min(abs(angles[1]) + abs(angles[3]), 0.6) * 0.5
        action_penalty = 0.02 * float(np.sum(np.square(action)))
        vel_penalty = 0.003 * float(np.sum(np.square(vels)))
        linvel_penalty = 0.01 * float(np.sum(np.square(base_linvel)))
        joint_penalty = 0.2 * float(np.sum(np.abs(angles)))

        reward = upright_reward + height_reward + knee_bonus - action_penalty - vel_penalty - linvel_penalty - joint_penalty + 0.02

        self.episode_reward += reward
        self.episode_length += 1

        # Standing detection: require a sustained upright posture near target joint angles
        angle_deviation = float(np.sum(np.abs(angles)))
        is_upright = abs(torso_pitch) < 0.12
        is_high_enough = torso_height > 0.95
        if is_upright and is_high_enough and angle_deviation < 0.6:
            self.standing_counter += 1
        else:
            self.standing_counter = 0

        # If held upright for required steps, give a success bonus and finish episode
        success = False
        if self.standing_counter >= self.standing_required_steps:
            reward += self.standing_bonus
            success = True

        done = torso_height < 0.25 or self.episode_length > 2000 or success

        info = {}
        if done:
            # Keep internal counters reset; do not set 'episode' key (wrappers manage that)
            self.episode_reward = 0.0
            self.episode_length = 0
            self.standing_counter = 0
            if success:
                info["is_success"] = True

        # If safe_reset requested, perform a quiet reset but still return done=True so training loops behave normally
        if self.safe_reset and done:
            for i in self.joint_indices:
                p.resetJointState(self.biped_id, i, 0.0, 0.0)
                p.setJointMotorControl2(self.biped_id, i, p.POSITION_CONTROL, 0.0, force=200)
            p.resetBasePositionAndOrientation(self.biped_id, [0,0,1.0], [0,0,0,1])

        if self.debug:
            print(f"[STEP] pitch={torso_pitch:.3f}, height={torso_height:.3f}, reward={reward:.3f}, done={done}")

        return obs, float(reward), bool(done), False, info

    # -----------------------------
    def _get_obs(self):
        states = [p.getJointState(self.biped_id, i) for i in self.joint_indices]
        angles = [s[0] for s in states]
        velocities = [s[1] for s in states]

        # Use base (torso) position/orientation which is more robust than link 0
        pos, orn = p.getBasePositionAndOrientation(self.biped_id)
        pitch = p.getEulerFromQuaternion(orn)[1]
        height = pos[2]

        # Contact detection: check contact points between biped and plane and inspect link indices
        left_contact = 0
        right_contact = 0
        contacts = p.getContactPoints(self.biped_id, self.plane_id)
        for cp in contacts:
            # contact tuple layout: (bodyA, bodyB, linkIndexA, linkIndexB, ...)
            # depending on pybullet version, indices 3/4 are link indices
            if len(cp) >= 5:
                linkA = cp[3]
                linkB = cp[4]
                if linkA == self.LEFT_KNEE or linkB == self.LEFT_KNEE:
                    left_contact = 1
                if linkA == self.RIGHT_KNEE or linkB == self.RIGHT_KNEE:
                    right_contact = 1

        # get base linear and angular velocity
        lin_vel, ang_vel = p.getBaseVelocity(self.biped_id)
        lin_vel = list(lin_vel)
        ang_vel = list(ang_vel)

        return np.array(angles + velocities + lin_vel + ang_vel + [pitch, height, left_contact, right_contact], dtype=np.float32)

    def render(self):
        # In GUI mode PyBullet already renders; for rgb_array mode we could use getCameraImage.
        return None

    def close(self):
        try:
            if p.isConnected():
                p.disconnect()
                if self.debug:
                    print("[CLOSE] PyBullet disconnected")
        except Exception:
            # ignore disconnect errors
            if self.debug:
                print("[CLOSE] PyBullet disconnect failed or was not connected")
