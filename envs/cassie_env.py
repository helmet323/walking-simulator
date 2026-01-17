import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
from robot_descriptions import cassie_description
from track import create_track

class CassieEnv(gym.Env):
    """Minimal custom Gym-like environment for Cassie walking."""

    metadata = {'render.modes': ['human']}

    def __init__(self, track_length=400):
        super().__init__()

        # ------------------------------
        # PyBullet connection
        # ------------------------------
        self.physicsClient = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # ------------------------------
        # Track
        # ------------------------------
        create_track(track_length=track_length)

        # ------------------------------
        # Load Cassie
        # ------------------------------
        self.robot = p.loadURDF(cassie_description.URDF_PATH, basePosition=[0, 0, 0.5],
                                useFixedBase=False)

        # ------------------------------
        # Define observation & action space
        # ------------------------------
        n_joints = p.getNumJoints(self.robot)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(n_joints * 2 + 6,), dtype=np.float32)
        # Joint positions + velocities + base pos & orientation
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32)

        # Time step
        self.dt = 1/240

    def step(self, action):
        # Scale action to target joint positions
        n_joints = p.getNumJoints(self.robot)
        for i in range(n_joints):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL,
                                    targetPosition=action[i],
                                    positionGain=0.5,
                                    velocityGain=0.5,
                                    force=100)

        p.stepSimulation()

        # Simple observation: joint positions + velocities + base pose
        joint_positions = []
        joint_velocities = []
        for i in range(n_joints):
            info = p.getJointState(self.robot, i)
            joint_positions.append(info[0])
            joint_velocities.append(info[1])

        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot)
        base_vel, base_angular = p.getBaseVelocity(self.robot)

        obs = np.array(joint_positions + joint_velocities +
                       list(base_pos) + list(base_orn), dtype=np.float32)

        # Simple reward: forward velocity in x
        reward = base_vel[0]

        # Done if Cassie falls below z < 0.2
        done = base_pos[2] < 0.2

        return obs, reward, done, {}

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        create_track(track_length=400)
        self.robot = p.loadURDF(cassie_description.URDF_PATH, basePosition=[0, 0, 0.5],
                                useFixedBase=False)
        n_joints = p.getNumJoints(self.robot)
        joint_positions = [0.0] * n_joints

        # Reset joints to zero
        for i in range(n_joints):
            p.resetJointState(self.robot, i, joint_positions[i])

        # Return initial observation
        joint_velocities = [0.0] * n_joints
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot)
        obs = np.array(joint_positions + joint_velocities +
                       list(base_pos) + list(base_orn), dtype=np.float32)
        return obs

    def render(self, mode='human'):
        pass  # GUI is already open

    def close(self):
        p.disconnect()
