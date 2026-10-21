# CreditQ: Autonomous RL Credit Decision Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0.0-green.svg)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Live Web Simulator](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-brightgreen.svg)](https://saksh2024.github.io/CreditQ/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Autonomous reinforcement learning decision system dynamically tuning loan approval cutoffs against macro default shocks on 100K+ loans to beat static baselines.**

🌐 **Live Interactive Web Simulator:** [https://saksh2024.github.io/CreditQ/](https://saksh2024.github.io/CreditQ/)  
📄 **Full Engineering Changelog (v1.0 – v9.0):** [CHANGELOG.md](CHANGELOG.md)

---

##  Key Highlights & Breakthroughs

* **Solved State Aliasing (1D $\rightarrow$ 4D):** Evolved observation space from 1D mean FICO to a 4D state vector `[acceptance_rate, incoming_mean_fico, rejection_rate, default_rate]`, preventing the agent from misinterpreting borrower score drops as strict cutoffs and isolating trailing default surges before capital insolvency.
* **Vectorized Counterfactual Updating:** Engineered a NumPy counterfactual updater evaluating all 34 cutoff actions simultaneously across each application batch, recovering **97% of cohort feedback signals** and preventing policy divergence under selection bias.
* **Life-of-Loan Calibration:** Expanded simulation horizon from 24 to **78 weeks at biweekly cadence** with empirical **55% recovery rate ($LGD = 0.45$)**, matching LendingClub's 17.2-month charge-off lifecycle.
* **Emergency Circuit-Breaker (`STOP_LENDING`):** Action 33 enforces zero origination during severe macro default surges, preserving **$493,987 (+60.3% loss reduction)** compared to static baselines.
* **Empirical Profit Surge:** Delivered **+45.9% profit lift** under population drift tests ($p < 10^{-8}$) across 100 Monte Carlo runs.

---

##  Table 1: Empirical Benchmark Results (78-Week Biweekly Horizon)

Benchmarked on **100,000+ LendingClub loans** (80K train / 20K test) across 100 Monte Carlo iterations ($N=100$):

| Scenario | Market Stress Condition | Static Baseline Profit | CreditQ (v9.0 Counterfactual RL) | Net Capital Preserved / Lift | Statistical Significance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Clean Economy** | Stationary distribution ($\Delta\mu = 0, \Delta PD = 0$) | **$663,799** | **$652,374** | *Near-optimal parity ($\sim98.3\%$)* | $p = 0.28$ (n.s.) |
| **2. Population Drift** | FICO degradation ($\mu - 20$ pts) | **$318,891** | **$465,302** | **+$146,411 (+45.9\%)$** | **$p = 3.2 \times 10^{-9}$** |
| **3. Selection Bias** | Hidden risk surge ($\Delta PD = +10\%$) | **-$819,057** | **-$325,070** | **+$493,987 (+60.3\%$ loss cut)** | **$p = 1.1 \times 10^{-14}$** |
| **4. Severe Crisis** | Drift + Default Surge ($\mu - 25, +15\%$ PD) | **-$1,241,500** | **-$412,180** | **+$829,320 (+66.8\%$ loss cut)** | **$p = 4.7 \times 10^{-18}$** |

---

## Architecture Overview

```
       Incoming Loan Stream (50 apps/week)
                       │
                       ▼
         ┌───────────────────────────┐
         │     Gymnasium CreditEnv   │
         │  State s_t = [acc, fico,  │
         │             rej, def]     │
         └─────────────┬─────────────┘
                       │
                       ▼
         ┌───────────────────────────┐
         │     CreditQ Policy Engine │
         │   (Dense Counterfactual   │
         │    Boltzmann Q-Learning)  │
         └─────────────┬─────────────┘
                       │ Action a_t (Cutoff 660..820 | STOP_LENDING)
                       ▼
         ┌───────────────────────────┐
         │ Portfolio Cash Flow Unit  │
         │  + Interest from Repaid   │
         │  - LGD (0.45) on Defaults │
         └───────────────────────────┘
```

### Observation Space (4D)
$$\mathbf{s}_t = \begin{bmatrix} \text{acceptance\_rate}_{t-1} \\ \text{norm\_mean\_fico}_t \\ \text{rejection\_rate}_{t-1} \\ \text{default\_rate}_{t-1} \end{bmatrix} \in [0, 1]^4$$

### Action Space (34 Discrete Cutoffs)
$$\mathcal{A} = \{660, 665, 670, \dots, 820\} \cup \{\text{STOP\_LENDING (9999)}\}$$

### Reward Formulation (Amortized Cash Flow with Recovery)
$$R_t = \sum_{i \in \text{Approved}} \left( \mathbb{I}(\text{repaid}_i) \cdot L_i \cdot r_i - \mathbb{I}(\text{default}_i) \cdot L_i \cdot (1 - \text{recovery\_rate}) \right)$$

---

##  Quickstart & Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Saksh2024/CreditQ.git
cd CreditQ
pip install -r requirements.txt
```

*(Core dependencies: `gymnasium`, `torch`, `scikit-learn`, `pandas`, `numpy`, `streamlit`, `matplotlib`)*

### 2. Run the Interactive Web Dashboard
The web simulator runs natively in your browser:
* **Live Online:** [https://saksh2024.github.io/CreditQ/](https://saksh2024.github.io/CreditQ/)
* **Locally:** Simply open `docs/index.html` in Chrome or Edge.

### 3. Run the Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```

### 4. Run the 78-Week Monte Carlo Benchmark
```bash
python Source/test_biweekly.py
```

---

##  Repository Structure

```
CreditQ/
├── docs/                      # GitHub Pages static deployment
│   ├── index.html             # Interactive Web Simulator & Charts
│   └── Figures/               # Visualizations & Architecture SVGs
├── Source/
│   ├── gymnasium_env.py       # Custom Gymnasium CreditCutoffEnv (4D state, 34 actions)
│   ├── rl_agent.py            # v9.0 Dense Counterfactual Q-Learning Engine
│   ├── test_biweekly.py       # 78-Week Monte Carlo Benchmark Suite
│   ├── prepare_lending_club.py # LendingClub 100K data pipeline
│   ├── prepare_short_term.py  # Short-term BNPL data pipeline (Phase 2)
│   └── lending_club_scored.csv # Pre-processed LendingClub benchmark cohort
├── streamlit_app.py           # Streamlit Web Application
├── CHANGELOG.md               # Version evolution log (v1.0 - v9.0)
├── index.html                 # Root simulator dashboard
└── README.md                  # Project documentation
```

---

##  Citation & License

Developed by **Sakshi Tandel** ([GitHub](https://github.com/Saksh2024) | [LinkedIn](https://www.linkedin.com/in/sakshitandel/)).  
Released under the [MIT License](LICENSE).
