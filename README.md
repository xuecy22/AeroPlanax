# Planax: A JAX‑Accelerated Fixed‑Wing Multi‑Agent RL Platform

*A lightweight, GPU‑friendly benchmark for high‑fidelity fixed‑wing dynamics, large‑scale MARL, and hierarchical self‑play.*

---

## Features

* **True six‑DOF aircraft & missile dynamics** implemented in pure JAX for just‑in‑time compilation and vectorisation.
* **Gymnax‑style environments** with built‑in support for thousands of parallel roll‑outs on a single GPU.
* **Ready‑to‑use PPO / MAPPO baselines** (single‑agent, multi‑agent, self‑play, hierarchical).
* **Tacview‑compatible replay exporter** for 3‑D debriefing and qualitative analysis.
* **One‑click reproducibility** via the locked `env_min.yml` Conda environment.

---

![fromework](assets/code structure.png)

## Installation

```shell
# 1. Clone repository
git clone https://github.com/xuecy22/AeroPlanax.git
cd AeroPlanax

# 2. Create research environment (CUDA 12 example)
conda env create -f env_min.yml
conda activate NeuralPlanex
```

> `env_min.yml` lists every runtime dependency used in the paper, including `jax‑cuda12‑pjrt`, `flax`, `optax`, `gymnax`, `tacview‑logger`, etc. Swap CUDA versions freely as long as **JAX ≥ 0.4.35** is available.

---

## Directory Layout

| Path           | Purpose                                                       |
| -------------- | ------------------------------------------------------------- |
| `dynamics/`    | Six‑DOF aircraft & missile models                             |
| `interpolate/` | Trilinear & spline lookup for aero tables                     |
| `envs/`        | Gymnax‑compatible tasks (`heading`, `formation`, `combat`, …) |
| `train_*.py`   | PPO / MAPPO baselines                                         |
| `render_*.py`  | Offline & real‑time Tacview exporters                         |
| `assets/`      | Figures & GIFs for the README                                 |

---

## Environments

### 1. Heading (single‑agent)

* **Scenario** – one aircraft receives a desired course and must stabilise its attitude while aligning with that heading as quickly as possible.
* **Observations** – the full 16‑dimensional flight state (body‑rates, Euler angles, quaternion, velocity components, angle‑of‑attack, sideslip, altitude e.t.).
* **Actions** – a 4‑way discrete set `[δ_a, δ_e, δ_t, δ_r]` surface commands.
* **Reward** – negative absolute heading error with small penalties for attitude deviation and control effort.
* **Termination** – episode ends after 300 simulation steps(30 s), or instantly if the aircraft stalls, exceeds load limits, or hits the ground.

![Heading (single‑agent)](assets/Heading.gif)

### 2. Formation

* **Scenario** – form and maintain wedge, line, or diamond spacing while avoiding mid‑air collisions.
* **Observations** – own flight state as above plus relative position/velocity to the virtual slot and the nearest neighbours.
* **Actions** – identical interface to Heading.
* **Reward** – quadratic distance to slot, collision penalty, shape‑keeping bonus, and control cost.
* **Termination** – collision, ground impact, or maximum episode length.

![Formation](assets/Formation.gif)

### 3. End‑to‑End Combat (self‑play / vs‑baseline)

* **Scenario** – symmetric dog‑fight ranging from 1 v 1 to 50 v 50. Each agent runs a single end‑to‑end policy that outputs manoeuvres plus missile‑launch commands.
* **Observations** – ego flight state, bearing/range/closure rate of visible opponents, missile inventory, line‑of‑sight angles, and basic fuel information.
* **Actions** – four continuous control surfaces plus a `fire_msl` Boolean.
* **Reward** – +1 for a kill, −1 for being killed, shaping for nose‑on position, energy management, and weapon economy.
* **Termination** – all aircraft on one side destroyed, self‑crash, or a 20 k‑step timeout.

### 4. Hierarchical Combat (self‑play / vs‑baseline)

* **Scenario** – identical arena to End‑to‑End, but each agent is governed by a two‑level policy: a high‑level planner outputs target heading / altitude / speed, while a shared low‑level controller (pre‑trained on Heading) tracks those commands.
* **Observations (high‑level)** – coarse situational awareness vectors (bandit angles, missile cues, remaining fuel, etc.).
* **Actions (high‑level)** – continuous `[Δψ_cmd, h_cmd, v_cmd]` guidance commands.
* **Reward** – same combat‑outcome terms, plus an imitation bonus favouring smooth, feasible guidance.
* **Advantages** – faster learning, clearer long‑horizon credit assignment, and the ability to swap different guidance laws with minimal retraining.

![Hierarchical Combat (self‑play / vs‑baseline)](assets/5v5_hier.gif)

---

## Quick Start

### Training

```shell
# single‑agent heading task (≈ 3 hours on one GPU)
python train_heading_discrete.py

# wedge / line / diamond formation (≈ 3 hours on one GPU)
python train_reformation.py

# num‑vs‑num self‑play combat task with hierarchical control (≈ 3 hours on one GPU)
python train_combat_selfplay_hierarchy.py

# num‑vs‑num self‑play combat task with end-to-end control (≈ 3 hours on one GPU)
python train_combat_selfplay.py

# num‑vs‑num vs-baseline combat task with hierarchical control (≈ 3 hours on one GPU)
python train_combat_vsbaseline_hierarchy.py

# num‑vs‑num vs-baseline combat task with end-to-end control (≈ 3 hours on one GPU)
python train_combat_vsbaseline.py

# 
```
The meanings of some common modifiable parameters are as follows.
- `NUM_ENVS` The number of parallel environments.
- `NUM_ACTORS` The number of agents in each environment.
- `NUM_STEPS` The number of trajectory steps collected by each environment before each update.
- `TOTAL_TIMESTEPS` The total number of steps in the entire training process.
- `OUTPUTDIR` Output directory, used to save various output files during the training process.
- `LOGDIR` Log directory, specifically designed to store training logs.
- `SAVEDIR` Model save directory, used to save model checkpoints during the training process.
- `LOADDIR` Directory path for loading pre trained models.

### Evaluation & Rendering

```shell
# single‑agent heading task
python render_heading_discrete.py

# wedge / line / diamond formation
python render_reformation.py

# num‑vs‑num self‑play combat task with hierarchical control
python render_combat_selfplay_hierarchy.py

# num‑vs‑num self‑play combat task with end-to-end control
python render_combat_selfplay.py

# num‑vs‑num vs-baseline combat task with hierarchical control
python render_combat_vsbaseline_hierarchy.py

# num‑vs‑num vs-baseline combat task with end-to-end control
python render_combat_vsbaseline.py
```

This will generate a `*.acmi` file. We can use [**TacView**](https://www.tacview.net/), a universal flight analysis tool, to open the file and watch the render videos.


## Citation

```bibtex
@inproceedings{Planax2025,
  title     = {Planax: A JAX-Based Platform for Efficient and Scalable Multi-Agent Reinforcement Learning in Fixed-Wing Aircraft Systems},
  author    = {Qihan Liu and Chuanyi Xue and Qinyu Dong},
  booktitle = {NeurIPS},
  year      = {2025}
}
```

---

## License

Planax is released under the MIT License. See `LICENSE` for details.
