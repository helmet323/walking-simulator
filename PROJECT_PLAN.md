# ML-Based Human Walking Simulation — Project Plan

## 🎯 Project Goal
Build a **bipedal walking agent** that learns to walk forward using **reinforcement learning**, running entirely on **free, open-source tools** (PyBullet + Gymnasium + PPO).

The agent should:
- Maintain balance
- Take repeated steps
- Walk forward without scripted trajectories

---

## 🛠 Technology Stack (Free)
- **Language:** Python
- **Physics Engine:** PyBullet
- **RL Framework:** Stable-Baselines3 (PPO)
- **Environment API:** Gymnasium
- **Hardware:** CPU (GPU optional)

---

## ✅ Success Criteria
The project is considered complete when the agent:
- Walks forward at least **3 meters**
- Remains upright for **≥ 10 seconds**
- Uses **learned control** (no hard-coded motion)
- Produces repeatable walking behavior

---

## 📅 Phase 0 — Preparation
**Objective:** Ensure prerequisites and define scope.

### Tasks
- Review basic Python and RL concepts
- Understand biped structure (torso, thighs, shins)
- Decide to start in **2D walking (locked sideways motion)**

### Deliverables
- Clear understanding of walking task
- Defined success criteria

---

## 🗂 Phase 1 — Project Setup & Physics Environment
**Objective:** Create a working simulation environment.

### Tasks
- Create Git repository and folder structure
- Set up Python virtual environment
- Install dependencies
- Load PyBullet plane and biped URDF
- Implement Gym environment (`reset`, `step`)
- Verify gravity, collisions, and falling behavior

### Deliverables
- Environment runs without errors
- Manual joint torques move the biped
- Episode terminates when the biped falls

**Milestone:** Physics simulation is stable

---

## 🤖 Phase 2 — Observation & Action Space Design
**Objective:** Define what the agent can sense and control.

### Observation Space
- Joint angles (hips, knees)
- Joint angular velocities
- Torso pitch angle
- Torso height
- Foot contact indicators

### Action Space
- Joint torques for each joint
- Normalized to range `[-1, 1]`

### Deliverables
- Observation values update correctly
- No NaNs or unstable values

**Milestone:** Agent can observe and act

---

## 🧠 Phase 3 — Standing & Balance Learning
**Objective:** Teach the agent to stand upright.

### Reward Design
- Positive reward for upright torso
- Positive reward for torso height
- Penalty for large torques
- Strong penalty for falling

### Training Strategy
- No forward motion reward
- Short episode lengths
- Strict fall termination

### Expected Outcome
- Reduced falling frequency
- Stable standing behavior

**Milestone:** Agent stands for ≥ 5 seconds

---

## 🚶 Phase 4 — Single-Step Learning
**Objective:** Learn weight shifting and stepping.

### Changes
- Introduce forward velocity reward
- Encourage alternating foot contacts
- Relax balance penalties slightly

### Techniques
- Curriculum learning
  - Start with low target velocity
  - Gradually increase over training

### Expected Outcome
- Shuffling behavior
- Occasional forward steps

**Milestone:** Agent successfully takes at least one step

---

## 🚶‍♂️ Phase 5 — Continuous Walking
**Objective:** Achieve stable forward walking.

### Reward Components
- Primary: forward velocity
- Penalty: energy usage (torque squared)
- Penalty: torso tilt
- Penalty: sideways motion

### Stabilization Techniques
- Lock movement to 2D
- Limit joint torques
- Optional symmetry regularization

### Evaluation Metrics
- Distance traveled per episode
- Time before falling
- Energy per meter

**Milestone:** Walks ≥ 3 meters without falling

---

## 📊 Phase 6 — Evaluation & Visualization
**Objective:** Validate learned behavior.

### Tasks
- Log rewards and episode lengths
- Plot forward velocity over time
- Save trained PPO models
- Replay policies in GUI mode

### Validation Questions
- Does the agent generalize from different starts?
- Can it recover from small disturbances?
- Is the motion smooth and stable?

**Milestone:** Walking is repeatable and explainable

---

## 🧪 Phase 7 — Optional Extensions
Choose one or more:
- Add ankles and feet
- Extend to full 3D walking
- Add robustness to pushes
- Use imitation learning from motion capture
- Improve energy efficiency

---

## ⚠️ Risk Management
| Risk | Mitigation |
|----|----|
| Agent never walks | Curriculum learning |
| Unstable motion | Stronger balance reward |
| Excessive hopping | Increase energy penalty |
| Slow training | Simplify model |

---

## 🏁 Final Deliverables
- Clean GitHub repository
- Trained PPO walking policy
- Documentation explaining:
  - Environment design
  - Reward shaping
  - Training results

---

## 🚀 Next Step
Begin **Phase 1: Project Setup & Physics Environment**  
Focus on making the biped load, move, and fall correctly before training.

