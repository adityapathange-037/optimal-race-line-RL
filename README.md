# Discrete Optimal Trajectory Planning via Bellman Value Iteration

A discrete Reinforcement Learning (RL) approach leveraging Markov Decision Process (MDP) formulation and Bellman Value Iteration to compute optimal velocity profiles and cornering trajectories on an S-shaped racetrack.

📄 **[Download / View Research Paper (PDF)](./RL_for_optimal_race_line_paper.pdf?raw=true)**

---

## Overview
Autonomous vehicle control and motorsport engineering both require calculating optimal path trajectories that minimize time-to-finish while respecting track boundaries and friction constraints. This project implements an offline Value Iteration algorithm to compute optimal joint actions (acceleration and steering) across a continuous-like grid environment.

---

## Key Features & Emergent Behaviors
- **MDP Formulation:** Joint state space $s = (x, y, v)$ combining spatial coordinates and variable velocity bounds ($v \in [1, 6]$).
- **Action-Dependent Friction Penalty:** Dynamic turn penalties penalize high-speed turning, simulating tire traction limits without complex non-linear physics.
- **Apex Cutting:** The agent learns to cut inner corners to minimize total travel time rather than strictly following the center line.
- **Corner Deceleration:** Automatic braking before entering sharp turns to avoid velocity-dependent penalties.
- **Real-Time Pygame Visualization:** Color-coded trajectory rendering displaying acceleration (Green), braking (Red), and speed maintenance (Blue).

---

## Mathematical Summary
The optimal state-value function $V^*(s)$ is computed using the Bellman Optimality Equation over a discount factor $\gamma = 0.95$:

$$V_{k+1}(x,y,v) = \max_{(a_v, a_y)} \{ R(s, a, s') + \gamma V_k(x', y', v') \}$$

### Reward Structure
- **Goal Reach:** $+100.0$
- **Collision / Off-Track:** $-50.0$
- **Step Cost:** $-1.0 - c_{\text{turn}}$ (where $c_{\text{turn}} = 0.5 \cdot |a_y|$ if $v_{t+1} = v_{\text{max}}$)

---

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install pygame numpy
