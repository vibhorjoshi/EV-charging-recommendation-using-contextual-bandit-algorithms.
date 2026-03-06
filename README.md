<div align="center">

<!-- Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00b09b,100:96c93d&height=220&section=header&text=EV%20Charging%20Recommender&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=Contextual%20Bandit%20Algorithms%20for%20Real-Time%20Station%20Recommendation&descSize=15&descAlignY=58&animation=fadeIn" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com)
[![Stars](https://img.shields.io/github/stars/vibhorjoshi/EV-charging-recommendation-using-contextual-bandit-algorithms?style=for-the-badge&color=f59e0b)](https://github.com/vibhorjoshi/EV-charging-recommendation-using-contextual-bandit-algorithms/stargazers)

<br/>

> **Benchmarking 5 contextual bandit algorithms on a 1.5M-station global EV dataset —**  
> **balancing cost savings, power adequacy, shock recovery, and explainability.**

<br/>

</div>

---

## 📋 Table of Contents

- [✨ Overview](#-overview)
- [🧠 Algorithms Benchmarked](#-algorithms-benchmarked)
- [📐 Feature Engineering](#-feature-engineering)
- [🔬 Benchmark Design](#-benchmark-design)
- [📊 Results at a Glance](#-results-at-a-glance)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [🛠️ Tech Stack](#️-tech-stack)
- [📚 References](#-references)

---

## ✨ Overview

EV adoption is accelerating globally, but finding the **right** charging station in real time — one that is close, affordable, powerful enough, and actually available — is a surprisingly hard decision problem.

This project frames real-time EV charging station selection as a **contextual bandit problem**: the agent observes a context (user location, battery state, time of day) and must recommend the best station from thousands of candidates, learning from feedback without knowing the true reward distribution in advance.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────┐
│                      SYSTEM OVERVIEW                            │
│                                                                 │
│  User Context ──► Feature Encoding ──► Bandit Algorithm        │
│       │                                       │                 │
│  [Location]      [8-Dim Vector]         [LinUCB / TS /         │
│  [Battery %]     [Normalised]            NeuralBandit /        │
│  [Time of Day]   [Constrained]           DQN / PPO]            │
│                                               │                 │
│                                    Top-K Station Ranked        │
│                                    with Reward & Feedback      │
└─────────────────────────────────────────────────────────────────┘
```

</div>

### 🌍 Dataset Scale

| Metric | Value |
|---|---|
| Total Stations | **1,500,000+** |
| Countries Covered | **Global** |
| Static Benchmark Queries | **100,000** (5 seeds) |
| Dynamic Benchmark Events | Price shocks + forced outages |

---

## 🧠 Algorithms Benchmarked

Five learning algorithms were implemented and evaluated, spanning classical linear bandits to deep reinforcement learning:

<div align="center">

| # | Algorithm | Type | Key Characteristic |
|---|---|---|---|
| 1 | **LinUCB** | Linear Bandit | Upper Confidence Bound exploration; highly explainable |
| 2 | **Thompson Sampling** | Bayesian Bandit | Probability matching; natural uncertainty quantification |
| 3 | **Neural Bandit** | Deep Bandit | Non-linear reward modeling; highest reward ceiling |
| 4 | **DQN** | Deep RL | Q-learning with replay buffer; adapts to delayed feedback |
| 5 | **PPO** | Deep RL (Policy Gradient) | On-policy; stable in non-stationary environments |

</div>

Each algorithm is compared against two simple baselines: **distance-only** and **price-only** greedy selectors.

---

## 📐 Feature Engineering

Every station is represented as an **8-dimensional normalised feature vector**:

```python
feature_vector = [
    haversine_distance,      # Normalised distance from user (Haversine formula)
    power_adequacy,          # Whether station power meets user requirement
    cost_per_kwh,            # Normalised price
    availability_24_7,       # Binary: is station always open?
    station_age_years,       # Infrastructure freshness
    accessibility_score,     # Disability/universal access rating
    connector_compatibility, # Connector type match
    current_queue_estimate,  # Estimated wait time
]
```

### ⚠️ Constraint Enforcement

A **hard constraint layer** filters candidates before any algorithm sees them:

```
VALID station must satisfy ALL of:
  ✅  Distance  ≤  user_max_distance
  ✅  Power     ≥  user_min_power_kw
  ✅  Status    =  OPERATIONAL
```

Any station failing a hard constraint receives **reward = 0** and is never recommended.

---

## 🔬 Benchmark Design

### Static Benchmark

- **100,000 queries** per algorithm
- **5 independent seeds** for variance estimation
- Metrics: relevance score, cost savings, power adequacy rate, constraint violation rate, oracle regret

### Dynamic Benchmark (Stress Test)

Real-world EV charging markets are volatile. Two shock scenarios test adaptation speed:

<div align="center">

```
  TIMELINE ─────────────────────────────────────────────────►
  
  Epoch 0          Epoch 50              Epoch 100
     │                 │                     │
     ▼                 ▼                     ▼
  [Stable]   ──►  [PRICE SHOCK]   ──►  [OUTAGE EVENT]
                  1.5× multiplier        20% of candidates
                  applied to all         forced offline
                  station prices         simultaneously
```

</div>

**Measured:**  time-to-recovery, post-shock reward degradation, constraint violation spike.

---

## 📊 Results at a Glance

| Algorithm | Cost Savings ↑ | Power Adequacy ↑ | Oracle Regret ↓ | Explainability |
|---|:---:|:---:|:---:|:---:|
| **LinUCB** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | 🟢 High |
| **Thompson Sampling** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | 🟢 High |
| **Neural Bandit** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Low | 🟡 Medium |
| **DQN** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Low | 🔴 Low |
| **PPO** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Low-Med | 🔴 Low |
| Distance Baseline | ⭐⭐ | ⭐⭐⭐ | High | 🟢 High |
| Price Baseline | ⭐⭐⭐⭐ | ⭐⭐ | High | 🟢 High |

> **Key insight:** NeuralBandit and DQN yield higher reward but at the cost of lower explainability. LinUCB and Thompson Sampling offer the best reward–transparency tradeoff for deployment in regulated or consumer-facing contexts.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/vibhorjoshi/EV-charging-recommendation-using-contextual-bandit-algorithms.git
cd EV-charging-recommendation-using-contextual-bandit-algorithms
pip install -r requirements.txt
```

### 2. Run Static Benchmark

```bash
python benchmark_static.py \
  --algorithms linucb thompson_sampling neural_bandit dqn ppo \
  --queries 100000 \
  --seeds 5
```

### 3. Run Dynamic (Shock) Benchmark

```bash
python benchmark_dynamic.py \
  --price_shock_multiplier 1.5 \
  --outage_fraction 0.20 \
  --plot
```

### 4. Visualise Results

```bash
python visualise_results.py --output plots/
```

---

## 📁 Project Structure

```
EV-charging-recommendation/
│
├── 📂 algorithms/
│   ├── linucb.py               # LinUCB with confidence bound exploration
│   ├── thompson_sampling.py    # Bayesian Thompson Sampling
│   ├── neural_bandit.py        # Neural network reward estimator
│   ├── dqn.py                  # Deep Q-Network with replay buffer
│   └── ppo.py                  # Proximal Policy Optimisation agent
│
├── 📂 data/
│   ├── loader.py               # Dataset loader (1.5M stations)
│   └── preprocessor.py        # Feature extraction & normalisation
│
├── 📂 environment/
│   ├── ev_env.py               # Bandit environment simulator
│   └── reward.py               # Constrained reward function
│
├── 📂 benchmarks/
│   ├── static_benchmark.py     # 100K query static evaluation
│   └── dynamic_benchmark.py    # Price shock & outage simulation
│
├── 📂 visualisation/
│   └── plots.py                # Cost curve, regret, transparency plots
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

<div align="center">

| Category | Tools |
|---|---|
| **Core ML** | PyTorch, NumPy, Scikit-learn |
| **Geospatial** | Haversine, Pandas |
| **Visualisation** | Matplotlib, Seaborn |
| **Data** | Kaggle API, Parquet |
| **Environment** | Python 3.10+, CUDA (optional) |

</div>

---

## 📚 References

- Chu, W. et al. (2011). *Contextual Bandits with Linear Payoff Functions.* AISTATS.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Thompson, W.R. (1933). *On the likelihood that one unknown probability exceeds another.* Biometrika.

---

<div align="center">

**Made with ⚡ by [Vibhor Joshi](https://github.com/vibhorjoshi)**  
M.Tech CSE · IIIT Guwahati · Research: Edge AI, LLM Quantisation, RL

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:96c93d,100:00b09b&height=100&section=footer" width="100%"/>

</div>
