import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import random


class BipedEnv(gym.Env):
    """
    Optimized Biped Environment:
    - Prevents backward/sideways falls
    - Rewards upright posture, foot contacts, and height
    - Limits knee/ankle bending
    - Faster hip response for balance
    - Standing bonus for stable upright stance
    """

    metadata = {"render_modes": ["human"], "render_fps": 240}

    def __init__(self, mode="direct", seed=None):
        super().__init__()
        self.mode = mode
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # ---------------- Physics ----------------
        self.client = p.connect(p.GUI if mode == "gui" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.setPhysicsEngineParameter(numSolverIterations=50)

        # ---------------- Plane ----------------
        self.plane_id = p.loadURDF("plane.urdf")

        # ---------------- Biped ----------------
        self.biped_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "biped.urdf"
        )
        self.biped_path = os.path.abspath(self.biped_path)
        if not os.path.exists(self.biped_path):
            raise FileNotFoundError(f"{self.biped_path} not found. Place biped.urdf in models/")

        self.biped_id = p.loadURDF(self.biped_path, [0, 0, 1.05], useFixedBase=False)

        # ---------------- Joints ----------------
        self.LEFT_HIP, self.LEFT_KNEE, self.LEFT_ANKLE = 0, 1, 2
        self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE = 3, 4, 5
        self.joint_indices = [
            self.LEFT_HIP, self.LEFT_KNEE, self.LEFT_ANKLE,
            self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE
        ]
        self.n_joints = len(self.joint_indices)

        # ---------------- Spaces ----------------
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )
        obs_low = np.array(
            [-np.pi]*self.n_joints + [-10.0]*self.n_joints + [-10.0]*3 + [-10.0]*3 + [-np.pi, -np.pi] + [0.0] + [0,0],
            dtype=np.float32
        )
        obs_high = np.array(
            [np.pi]*self.n_joints + [10.0]*self.n_joints + [10.0]*3 + [10.0]*3 + [np.pi, np.pi] + [2.0] + [1,1],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # ---------------- Control ----------------
        self.max_joint_delta = 0.05
        self.max_force = 600
        self.prev_targets = np.zeros(self.n_joints, dtype=np.float32)

        # ---------------- Episode ----------------
        self.step_count = 0
        self.max_steps = 2000
        self.standing_counter = 0
        self.standing_required_steps = 50
        self.standing_bonus = 5.0

    # =========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        self.plane_id = p.loadURDF("plane.urdf")
        self.biped_id = p.loadURDF(self.biped_path, [0,0,1.05], useFixedBase=False)

        # Small forward + lateral tilt for stability
        side_tilt = random.uniform(-0.02, 0.02)
        forward_tilt = 0.05
        p.resetBasePositionAndOrientation(
            self.biped_id,
            [0, 0, 1.05],
            p.getQuaternionFromEuler([forward_tilt + side_tilt, 0, 0])
        )

        for j in self.joint_indices:
            p.resetJointState(self.biped_id, j, 0.0, 0.0)
            p.setJointMotorControl2(self.biped_id, j, p.POSITION_CONTROL, targetPosition=0.0, force=self.max_force)

        self.prev_targets[:] = 0.0
        self.step_count = 0
        self.standing_counter = 0

        return self._get_obs(), {}

    # =========================================================
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        for i, j in enumerate(self.joint_indices):
            cur = p.getJointState(self.biped_id, j)[0]
            desired = self.prev_targets[i] + action[i]*0.08

            # Joint limits
            if j in [self.LEFT_KNEE, self.RIGHT_KNEE]:
                desired = np.clip(desired, -0.3, 0.5)
            if j in [self.LEFT_ANKLE, self.RIGHT_ANKLE]:
                desired = np.clip(desired, -0.4, 0.3)

            max_delta = 0.08 if j in [self.LEFT_HIP, self.RIGHT_HIP] else self.max_joint_delta
            target = np.clip(desired, cur - max_delta, cur + max_delta)

            p.setJointMotorControl2(self.biped_id, j, p.POSITION_CONTROL, targetPosition=target, force=self.max_force)
            self.prev_targets[i] = target

        p.stepSimulation()
        self.step_count += 1

        obs = self._get_obs()
        angles = obs[0:self.n_joints]
        vels = obs[self.n_joints:2*self.n_joints]
        lin_vel = obs[2*self.n_joints:2*self.n_joints+3]
        pitch, roll, height, left_contact, right_contact = obs[-5:]

        # ---------------- Reward ----------------
        upright = np.exp(-12.0*pitch**2)*np.exp(-12.0*roll**2)
        height_r = np.clip(height-0.9,0.0,0.25)
        energy = np.sum(action**2)
        joint_vel = np.sum(vels**2)
        lateral_penalty = 0.5*abs(lin_vel[1])
        foot_bonus = 0.3*(left_contact + right_contact)

        reward = (
            3.0*upright +
            2.0*height_r -
            0.02*energy -
            0.01*joint_vel -
            lateral_penalty +
            foot_bonus +
            0.05
        )

        # ---------------- Standing bonus ----------------
        if abs(pitch)<0.12 and abs(roll)<0.1 and height>0.95:
            self.standing_counter += 1
            if self.standing_counter>=self.standing_required_steps:
                reward += self.standing_bonus
        else:
            self.standing_counter = 0

        done = height < 0.25 or self.step_count >= self.max_steps

        return obs, float(reward), done, False, {}

    # =========================================================
    def _get_obs(self):
        states = [p.getJointState(self.biped_id,j) for j in self.joint_indices]
        angles = [s[0] for s in states]
        vels = [s[1] for s in states]
        pos, orn = p.getBasePositionAndOrientation(self.biped_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.biped_id)
        pitch, roll, _ = p.getEulerFromQuaternion(orn)
        height = pos[2]

        left_contact = 0
        right_contact = 0
        for c in p.getContactPoints(self.biped_id, self.plane_id):
            if c[3] == self.LEFT_ANKLE: left_contact=1
            if c[3] == self.RIGHT_ANKLE: right_contact=1

        return np.array(
            angles + vels + list(lin_vel) + list(ang_vel) + [pitch, roll, height, left_contact, right_contact],
            dtype=np.float32
        )

    # =========================================================
    def close(self):
        if p.isConnected():
            p.disconnect()
