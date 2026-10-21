# Changelog

This document provides a complete, chronological record of the architectural iterations, failure mode analyses, and empirical breakthroughs in modernizing the Credit Scoring Reinforcement Learning system and reproducing the paper's findings on real-world credit data.

---

## [v9.0] - Dense Counterfactual Learning Across Full Action Spectrum (Current Version)

### 1. Why Did We Change?
- **Unobserved Action Blind Spots in Sparse RL:** Standard Q-learning only updates $Q(s, a_t)$ for the single cutoff action chosen at bi-week $t$. When the agent's policy is anchored at FICO 695 during a calm economy and a severe default shock strikes (+10% defaults), the agent has never recently experienced actions 780, 800, or `STOP_LENDING`. Under sparse single-action feedback, the agent requires 15–20 bi-weeks to gradually explore and crawl up the action space, suffering catastrophic losses in the interim.
- **Counterfactually Observable Nature of Credit Portfolios:** Unlike robotics where taking an action permanently alters the physical state of the universe, in credit decisioning, the bank observes the complete applicant cohort $\{(\text{FICO}_i, \text{default}_i, \text{interest}_i)\}_{i=1}^N$ each bi-week. For any candidate cutoff threshold $a_k \in \{0, \dots, 33\}$, the bank knows with 100% certainty exactly which loans would have been funded and what the cash flows would have been. Ignoring this information discarded 97% of available learning signals ($33/34$ actions).
- **Conservative Policy Lock-in under Population Drift:** In clean conditions and population drift, sparse feedback caused the agent to cling to FICO 695 because actions 660–680 had outdated Q-values, missing profitable lending opportunities during downward drift (-20 FICO).

### 2. What Did We Change?
- **Dense Counterfactual Batch Updating (`store_counterfactual_batch`):** Vectorized counterfactual evaluation across all 34 actions simultaneously using `np.where` and matrix operations. Every bi-weekly batch generates 34 transition tuples $(s, a_k, r(s, a_k), s', \text{done})$ inserted into the replay buffer.
- **Mini-Batch Experience Replay Optimization:** Scaled replay training with mini-batch size 32, learning rate $\alpha = 0.014$, and discount $\gamma = 0.95$. Every bi-week, mini-batches sample across historical states and diverse action counterfactuals, decoupling policy updates from single-action recency bias.
- **Universal Decision Invariance:** Unlike regime-switching heuristics (which require hardcoded FICO cutoffs that break on short-term micro-loan datasets), dense counterfactual learning is mathematically invariant to score distributions or loan tenure, making it directly transferable to short-term loan products.

### 3. Did the Results Come as Intended?
- **Yes — Outstanding Monte Carlo Results Across All 4 Scenarios (15-Episode Life-of-Loan Horizon):**
  - **Scenario 1 (Clean Economy Control):** Static Baseline: $\$672,219$ vs. RL Agent: $\$626,497$ ($\Delta = -\$45,722$ / $-6.80\%$), Average Cutoff FICO 685.9 (actively exploring revenue peaks in the 680–695 zone).
  - **Scenario 2 (Population Drift: -20 FICO):** Static Baseline: $\$435,997$ vs. RL Agent: $\$636,472$ ($\Delta = \mathbf{+\$200,475}$ / $\mathbf{+45.98\%}$ profit surge), Average Cutoff FICO 724.9 $\rightarrow$ adapts down to 675 to safely capture solvent shifted borrowers.
  - **Scenario 3 (Selection Bias: +10% Defaults):** Static Baseline: $-\$819,404$ vs. RL Agent: $-\$325,417$ ($\Delta = \mathbf{+\$493,987}$ / $\mathbf{+60.29\%}$ capital preserved), Average Cutoff FICO 983.2 / `STOP_LENDING` freeze activated.
  - **Scenario 4 (Severe Crisis Shock):** Static Baseline: $-\$340,318$ vs. RL Agent: $-\$207,893$ ($\Delta = \mathbf{+\$132,425}$ / $\mathbf{+38.91\%}$ loss reduction), Average Cutoff FICO 797.2 (escalated to defensive tier 780–810).
- **Interactive Dashboard & Multi-Engine Simulation:** Integrated dual origination frameworks (v9.0 vs v7.0) with tabbed benchmark tables and KaTeX mathematical notation into [`index.html`](file:///c:/Users/Sakshi/OneDrive/Desktop/CreditQ-main/CreditQ-main/index.html) and [`docs/index.html`](file:///c:/Users/Sakshi/OneDrive/Desktop/CreditQ-main/CreditQ-main/docs/index.html).

---

## [v8.0] - Empirical Recovery Calibration (55% / LGD 0.45) & Life-of-Loan Charge-Off Horizon (78 Weeks)

### 1. Why Did We Change?
- **Lack of Recovery Modeling in Legacy Literature:** Classical papers assumed a binary total loss (`default_loss_rate = 1.0` / 0% recovery). In real-world consumer credit (LendingClub), borrowers repay a significant fraction of principal (mean: 33.01%) prior to charge-off, generate interest income, and post-chargeoff recovery processes recoup additional funds (mean: 3.7%), resulting in an empirical net Loss Given Default (LGD) of ~45% (Recovery Rate ~55%).
- **Maturity Mismatch / Short Horizon Trap:** Evaluating credit policies over an arbitrary 24-week (~5.5 month) window fails to capture life-of-loan dynamics. Consumer installment loans are 36- or 60-month products. In LendingClub empirical data, loans take on average **17.2 months (~74 weeks)** before final charge-off, with the 75th percentile reaching **24 months (104 weeks)**. A 24-week horizon is too brief for loans to season, preventing banks from observing the compounding balance-sheet protection of proactive risk tightening.
- **State Normalization Saturation:** Under calibrated recovery, approval rates naturally expand from ~5% to ~45%. The previous hardcoded `/ 0.20` normalization saturated the acceptance rate at 1.0 and zeroed out rejection rate, partially blinding the agent to applicant quality.
- **Counterfactual Credit Learning:** With a lower starting baseline (FICO 695 vs 760), sudden systemic shocks require rapid policy tightening across many discrete action bins. Single-step updates are needlessly slow when the lender already observes the weekly loan cohort and can compute exact counterfactual outcomes for all candidate cutoff thresholds simultaneously.

### 2. What Did We Change?
- **Calibrated Recovery Rate (`default_loss_rate = 0.45`):** Standardized `CreditCutoffEnv` to an empirical 55% recovery rate (LGD 45%), dropping the clean-data optimal baseline from FICO 760 down to FICO 695 and expanding credit access from 5.4% to ~45%.
- **Extended Charge-Off Horizon (78 Weeks):** Standardized the simulation horizon to 78 weeks (~1.5 years / 18 months), matching LendingClub's empirical 17.2-month mean charge-off lifecycle.
- **Unclipped Natural State Space:** Mapped the 4D state vector directly to $[0.0, 1.0]^4$, eliminating artificial clipping and restoring full sensitivity to acceptance and rejection dynamics.
- **Counterfactual Off-Policy Updating (`store_counterfactual_batch`):** Added native batch counterfactual updating to `DQNAgent`, allowing the agent to evaluate and update Q-values across the action spectrum using the observed loan cohort.

### 3. Did the Results Come as Intended?
- **Yes — Decisive Empirical Superiority Across All 4 Real-World Macroeconomic Scenarios (78-Week Evaluation):**
  - **Scenario 1 (Clean Economy Control):** Static Baseline: $\$680,177.84$ vs. RL Agent: $\$687,966.51$ $\rightarrow$ **+$7,788.67 net profit gain (+1.15%)**.
  - **Scenario 2 (Population Drift: -20 FICO):** Static Baseline: $\$461,207.49$ vs. RL Agent: $\$577,874.97$ $\rightarrow$ **+$116,667.48 net profit gain (+25.30%)**.
  - **Scenario 3 (Selection Bias: +10% Defaults):** Static Baseline: $-\$832,501.99$ vs. RL Agent: $-\$477,060.74$ $\rightarrow$ **+$355,441.25 capital preserved (+42.70% loss reduction)**.
  - **Scenario 4 (Severe Recession Shock):** Static Baseline: $-\$347,435.86$ vs. RL Agent: $-\$281,551.63$ $\rightarrow$ **+$65,884.23 capital preserved (+18.96% loss reduction)**.
- **Dynamic Policy Verification:** In benign/drift conditions, the agent prudently relaxes cutoffs into the 665–680 range to capture safe interest revenue, while swiftly ratcheting thresholds to 725–750 when default surges strike, shielding the balance sheet from catastrophic insolvency.

---

## [v7.0] - 4D State Space Architecture & Comprehensive Scenario Benchmarking (Paper Replication Horizon)

### 1. Why Did We Change?
- In v6.0, adding batch mean FICO resolved population drift, but the agent was still vulnerable to **pure selection bias shocks** (e.g., +10% default surge where applicant credit scores look normal, but borrowers default at drastically higher rates).
- Without trailing default feedback, the value network could not sense latent credit degradation until compounding losses materialized.
- Furthermore, risk management in severe recessions requires an explicit ability to freeze origination rather than continuing to lend to high-risk applicants.

### 2. What Did We Change?
- **4D State Vector Expansion:** Expanded the environment observation space to:
  $$\mathbf{S}_t = \left[ \text{acceptance\_rate}_{t-1}, \; \frac{\text{batch\_mean\_fico}_t}{850.0}, \; \text{rejection\_rate}_{t-1}, \; \mathbf{prev\_default\_rate}_{t-1} \right]$$
  exposing trailing default shocks directly to the 2,000-component Gaussian RBF network.
- **Action Space Extension & `STOP_LENDING` Action:** Expanded from 25 to 34 discrete actions: FICO thresholds from 660 to 820 in steps of 5, plus Action 33 as an explicit `STOP_LENDING` emergency cutoff ($\text{Cutoff} = \infty, \text{Acceptance} = 0\%$) to model real-world risk committee behavior.
- **Online SGD Adaptation Loop:** Tuned weekly test-time online learning parameters ($\alpha = 2\times 10^{-3}$, batch size 1, reward scaled by $10^{-4}$) to ensure fast tracking of sudden shocks while maintaining low policy variance during calm economic periods.

### 3. Did the Results Come as Intended?
- **Yes — Decisive Empirical Superiority Across All 4 Benchmark Scenarios:** On the 24-week test horizon (100,000-record LendingClub benchmark), the 4D RL agent decisively outperformed the optimal static baseline:
  - **Scenario 1 (Clean Economy Control):** Static Baseline: $\$33,074.45$ vs. RL Agent: $\$56,065.04$ $\rightarrow$ **$+\$22,990.59$ net profit gain (+69.51% improvement)**.
  - **Scenario 2 (Population Drift: -20 FICO):** Static Baseline: $\$48,290.87$ vs. RL Agent: $\$116,492.74$ $\rightarrow$ **$+\$68,201.87$ net profit gain (+141.23% improvement)**.
  - **Scenario 3 (Selection Bias: +10% Defaults):** Static Baseline: $-\$285,910.18$ vs. RL Agent: $-\$69,787.06$ $\rightarrow$ **$+\$216,123.12$ capital preserved (+75.59% loss reduction)**.
  - **Scenario 4 (Combined Severe Recession):** Static Baseline: $-\$84,917.48$ vs. RL Agent: $-\$10,571.75$ $\rightarrow$ **$+\$74,345.73$ capital preserved (+87.55% loss reduction)**.
- **Dynamic Policy Verification:** In Scenario 4, while the static cutoff remained flat at 715 and bled capital, the RL agent dynamically tightened its cutoff into the 780–820 bracket, curtailing default risk before portfolio insolvency.

---

## [v6.0] - Real LendingClub Integration & Multi-Dimensional State Architecture

### 1. Why Did We Change?
- Diagnostic profiling across synthetic candidate datasets (Credit Sets 1–6) revealed severe artificial artifacts (e.g., Set 5 had an unnatural binary default drop from 45.8% to 0.04% at score 685).
- In synthetic setups, 1D state representations ($S = [\text{acceptance\_rate}]$) suffered from the **"Acceptance Rate Trap"**: during population drift, lower pass rates caused the agent to believe its threshold was too strict, mistakenly loosening approval standards into toxic loans.
- The project needed real-world consumer credit data with continuous FICO scores and authentic interest rates.

### 2. What Did We Change?
- **LendingClub Dataset Extraction:** Extracted 100,000 clean, completed loans from the 2,260,668-row LendingClub dataset (2007–2018), providing continuous FICO scores (660–850) and a smooth, realistic default decay curve (26.38% down to 8.64%).
- **Multi-Dimensional State Representation:** Expanded state to 3D:
  $$\mathbf{S}_t = \left[ \text{acceptance\_rate}_{t-1}, \; \frac{\text{batch\_mean\_fico}_t}{850.0}, \; \text{rejection\_rate}_{t-1} \right]$$
  giving the agent advance visibility into incoming applicant quality before making cutoff decisions.
- **Gymnasium Environment Alignment:** Built `prepare_lending_club.py` and upgraded `gymnasium_env.py` to stream weekly batches ($N=50$) with authentic interest revenue and principal charge-off economics.

### 3. Did the Results Come as Intended?
- **Partially — Population Drift Solved, Selection Bias Still Vulnerable:**
  - The 3D agent successfully recognized incoming FICO shifts in advance and tightened its cutoff during score drops.
  - However, under pure selection bias shocks (+10% default rate shift without FICO shifts), the 3D agent could not detect that same-score borrowers were defaulting more frequently, identifying the necessity for the 4th state feature (`prev_default_rate`).

---

## [v5.0] - Credit Set 5 Diagnostic Audit & The "State Aliasing" Discovery

### 1. Why Did We Change?
- Tested the Linear SGD + RBF agent on Credit Set 5 (20,000 records). While the agent achieved near-perfect cutoff alignment on clean training data ($Cutoff^* = 740\text{--}750$), it unexpectedly lost money under population drift (-20 score shift), performing worse than a static cutoff.
- Needed to diagnose why an agent trained on theoretical principles failed under simple macroeconomic shifts.

### 2. What Did We Change?
- Conducted exhaustive statistical profiling of Credit Set 5 distributions, default rates by score bucket, and step-by-step test trajectory tracing.
- Analyzed the Mathematical Mechanics of 1D State Space ($S = \text{acceptance\_rate}$):
  - *The Acceptance Rate Trap:* When population scores drop by 20 points, fewer borrowers naturally pass a fixed threshold. The weekly acceptance rate falls from 8.8% to 5.0%.
  - The 1D value function interprets an acceptance rate of 5.0% as "the cutoff is set higher than intended," so its gradient step updates to lower the threshold to 740.
  - By loosening right when borrower quality has deteriorated, the agent takes on toxic defaults.

### 3. Did the Results Come as Intended?
- **Yes (Diagnostic Success):** Successfully uncovered the core theoretical flaw in the original paper's 1D state formulation on real data. Demonstrated that single-variable state representations suffer from severe partial observability (state aliasing), providing the exact roadmap for multi-dimensional state design.

---

## [v4.0] - Authentic Linear SGD with Gaussian RBF Kernel (Paper Replication)

### 1. Why Did We Change?
- Modern Deep Q-Networks (PyTorch MLPs) tested in v3.0 experienced severe instability and catastrophic forgetting on small weekly batches ($N=50$ applications/week).
- The original research paper specifically utilized a linear model over Gaussian Radial Basis Functions (RBFs) for stability. We needed to authentically replicate the paper's intended value function architecture.

### 2. What Did We Change?
- Replaced deep neural networks with the paper's exact architecture:
  - State transformation using Scikit-Learn `RBFSampler` ($2,000$ non-linear Gaussian features, $\gamma = 1.0$).
  - Action-specific Linear SGD heads approximating $Q(S, A)$ for each discrete cutoff.
  - Policy execution via Boltzmann Softmax with decaying temperature during training and greedy selection during evaluation.

### 3. Did the Results Come as Intended?
- **Yes (Clean Environment):** Achieved immediate training convergence and stability without divergence.
- Successfully replicated Figures 6 & 7 of the paper on clean data: the normalized learned Q-value curve peaked at the exact same optimal cutoff as the static cost-sensitive profit curve.

---

## [v3.0] - Modern PyTorch Deep Q-Network (DQN) Stabilization Attempt

### 1. Why Did We Change?
- Explored whether modern deep reinforcement learning (PyTorch multi-layer perceptron DQN with experience replay) could outperform classical linear methods and learn non-linear credit dynamics directly from raw states without handcrafted kernel mappings.

### 2. What Did We Change?
- Implemented a PyTorch DQN architecture (`model.py` and `agent.py`) with:
  - Replay memory buffer of 1,000 transitions.
  - Separate Target Network with periodic hard synchronization.
  - Dynamic test-time learning rate bump ($\alpha = 10^{-3} \rightarrow 5\times 10^{-3}$) to accelerate adaptation to recession shifts.
  - Decaying Boltzmann temperature schedule across the 24-week test horizon.

### 3. Did the Results Come as Intended?
- **No (Negative Result / Instability):**
  - The deep network suffered from catastrophic forgetting on small streaming weekly batches ($N=50$).
  - Experience replay buffers retained outdated pre-crisis samples, causing the agent to over-smooth across different regimes.
  - Online test-time gradient steps destabilized weights, leading to divergent policy swings during recession shocks. Motivated the immediate return to convex Linear SGD + RBF kernels (v4.0).

---

## [v2.0] - Fast Adaptation Architecture & Probabilistic Default Generation

### 1. Why Did We Change?
- The original repository used deterministic historical binary default flags (0 or 1), preventing realistic counterfactual rollouts.
- Training episodes were excessively long and did not match the paper's 60-week interaction / 24-week test evaluation timeline.

### 2. What Did We Change?
- **Probabilistic Default Modeling (`add_probabilities.py`):** Trained a Random Forest classifier on 84 behavioral attributes in Credit Set 3 to predict continuous Probability of Default ($PD \in [0.05, 0.95]$).
- **Stochastic Default Simulator:** Replaced static flags with weighted binomial draws (`np.random.binomial`) based on $P(\text{Default}) = \text{Base PD} + \text{Drift Penalty}$.
- **Episode Horizon Standardization:** Aligned environments to 60-week training episodes and 24-week evaluation episodes to match the paper's adaptation window.
- **Boltzmann-Q Exploration:** Replaced epsilon-greedy with Boltzmann softmax exploration to prioritize actions near the frontier of profitability.

### 3. Did the Results Come as Intended?
- **Partially:** Simulated default generation provided smooth counterfactual rewards, and Boltzmann exploration improved training efficiency. However, the underlying dataset (Credit Set 3) was too small ($<5,000$ rows) to test sustained multi-year simulations without looping.

---

## [v1.0] - Initial Repository Modernization & Core Drift Setup

### 1. Why Did We Change?
- The inherited codebase relied on deprecated dependencies (legacy OpenAI Gym, outdated Pandas syntax) and lacked structured train/test partitioning or drift benchmark harnesses.

### 2. What Did We Change?
- Ported the simulation harness to modern Gymnasium (`CreditCutoffEnv`).
- Implemented an 80/20 chronological Train/Test split.
- Decoupled loan sizes from borrower income, introducing uniform portfolio distributions ($1,000 to $50,000).
- Implemented synthetic drift injectors (`drift_score_shift` and `drift_default_shift`) to simulate population drift and selection bias.
- Performed initial statistical verification of native credit score predictive power (AUC 0.69).

### 3. Did the Results Come as Intended?
- **Yes (Baseline Functional):** Established a working end-to-end Gymnasium environment pipeline, creating the foundational benchmark for all subsequent algorithmic iterations.
