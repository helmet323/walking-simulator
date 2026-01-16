import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import random


class BipedEnv(gym.Env):
    """
    Biped standing environment with ankles and feet.
    Designed for PPO + VecNormalize.
    """

    metadata = {"render_modes": ["human"], "render_fps": 240}

    def __init__(self, mode="direct", debug=False, seed=None):
        super().__init__()

        self.mode = mode
        self.debug = debug

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # ---------------- Physics ----------------
        if self.mode == "gui":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.setPhysicsEngineParameter(numSolverIterations=50)

        # ---------------- Load plane ----------------
        self.plane_id = p.loadURDF("plane.urdf")

        # ---------------- Load biped ----------------
        self.biped_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models",
            "biped.urdf",
        )
        if not os.path.exists(self.biped_path):
            raise FileNotFoundError(self.biped_path)

        self.biped_id = p.loadURDF(
            self.biped_path, [0, 0, 1.05], useFixedBase=False
        )

        # ---------------- Joint indices ----------------
        self.LEFT_HIP = 0
        self.LEFT_KNEE = 1
        self.LEFT_ANKLE = 2
        self.RIGHT_HIP = 3
        self.RIGHT_KNEE = 4
        self.RIGHT_ANKLE = 5

        self.joint_indices = [
            self.LEFT_HIP,
            self.LEFT_KNEE,
            self.LEFT_ANKLE,
            self.RIGHT_HIP,
            self.RIGHT_KNEE,
            self.RIGHT_ANKLE,
        ]

        self.n_joints = len(self.joint_indices)

        # ---------------- Spaces ----------------
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        obs_low = np.array(
            [-np.pi] * self.n_joints +
            [-10.0] * self.n_joints +
            [-10.0] * 3 +          # base lin vel
            [-10.0] * 3 +          # base ang vel
            [-np.pi, -np.pi] +     # pitch, roll
            [0.0] +                # height
            [0, 0],                # foot contacts
            dtype=np.float32
        )

        obs_high = np.array(
            [np.pi] * self.n_joints +
            [10.0] * self.n_joints +
            [10.0] * 3 +
            [10.0] * 3 +
            [np.pi, np.pi] +
            [2.0] +
            [1, 1],
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # ---------------- Control ----------------
        self.max_joint_delta = 0.03
        self.max_force = 250
        self.prev_targets = np.zeros(self.n_joints, dtype=np.float32)

        # ---------------- Episode ----------------
        self.step_count = 0
        self.max_steps = 2000

    # =========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        self.plane_id = p.loadURDF("plane.urdf")

        self.biped_id = p.loadURDF(
            self.biped_path, [0, 0, 1.05], useFixedBase=False
        )

        tilt = random.uniform(-0.03, 0.03)
        p.resetBasePositionAndOrientation(
            self.biped_id,
            [0, 0, 1.05],
            p.getQuaternionFromEuler([tilt, 0, 0]),
        )

        for j in self.joint_indices:
            p.resetJointState(self.biped_id, j, 0.0, 0.0)
            p.setJointMotorControl2(
                self.biped_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=self.max_force,
            )

        self.prev_targets[:] = 0.0
        self.step_count = 0

        obs = self._get_obs()
        return obs, {}

    # =========================================================
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        for i, j in enumerate(self.joint_indices):
            cur = p.getJointState(self.biped_id, j)[0]
            desired = self.prev_targets[i] + action[i] * 0.08
            target = np.clip(
                desired,
                cur - self.max_joint_delta,
                cur + self.max_joint_delta,
            )

            p.setJointMotorControl2(
                self.biped_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=self.max_force,
            )

            self.prev_targets[i] = target

        p.stepSimulation()
        self.step_count += 1

        obs = self._get_obs()

        # ---------------- Unpack ----------------
        angles = obs[0:self.n_joints]
        vels = obs[self.n_joints:2*self.n_joints]
        pitch = obs[-5]
        roll = obs[-4]
        height = obs[-3]

        # ---------------- Reward ----------------
        upright = np.exp(-3.0 * (pitch**2 + roll**2))
        height_r = np.clip(height - 0.9, 0.0, 0.25)

        knee_use = abs(angles[self.LEFT_KNEE]) + abs(angles[self.RIGHT_KNEE])
        ankle_use = abs(angles[self.LEFT_ANKLE]) + abs(angles[self.RIGHT_ANKLE])

        energy = np.sum(action ** 2)
        joint_vel = np.sum(np.square(vels))

        reward = (
            2.0 * upright +
            1.5 * height_r +
            0.3 * knee_use +
            0.2 * ankle_use -
            0.02 * energy -
            0.01 * joint_vel +
            0.05
        )

        # ---------------- Termination ----------------
        fallen = height < 0.25
        timeout = self.step_count >= self.max_steps

        done = fallen or timeout
        info = {}

        return obs, float(reward), done, False, info

    # =========================================================
    def _get_obs(self):
        states = [p.getJointState(self.biped_id, j) for j in self.joint_indices]
        angles = [s[0] for s in states]
        vels = [s[1] for s in states]

        pos, orn = p.getBasePositionAndOrientation(self.biped_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.biped_id)

        pitch, roll, _ = p.getEulerFromQuaternion(orn)
        height = pos[2]

        left_contact = 0
        right_contact = 0

        contacts = p.getContactPoints(self.biped_id, self.plane_id)
        for c in contacts:
            if c[3] == self.LEFT_ANKLE:
                left_contact = 1
            if c[3] == self.RIGHT_ANKLE:
                right_contact = 1

        return np.array(
            angles +
            vels +
            list(lin_vel) +
            list(ang_vel) +
            [pitch, roll, height, left_contact, right_contact],
            dtype=np.float32,
        )

    # =========================================================
    def close(self):
        if p.isConnected():
            p.disconnect()
