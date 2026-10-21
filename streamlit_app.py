import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import sys

# Append Source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "Source"))

from gymnasium_env import CreditCutoffEnv
from rl_agent import DQNAgent

# Page Configuration
st.set_page_config(
    page_title="CreditQ — Autonomous RL Credit Decisioning",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark/Modern Fintech Theme)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #2563EB, #0284C7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.2rem;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #10B981;
        font-weight: 600;
    }
    .metric-sub-neg {
        font-size: 0.85rem;
        color: #EF4444;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">CreditQ: Autonomous RL Credit Decisioning System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Dynamic Credit Scoring Cutoff Optimization under Population Drift & Macroeconomic Shocks '
    '• 100K LendingClub Benchmark • 4D Observation Space • Linear SGD with Gaussian RBF Kernel'
    '</div>', 
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Sidebar: Simulation Controls
# -----------------------------------------------------------------------------
st.sidebar.header("🕹️ Simulation Controls")

preset_options = [
    "Severe Recession Shock (-20 FICO & +10% Defaults)",
    "Systemic Default Surge (+10% Defaults)",
    "Population Drift (-20 FICO Migration)",
    "Clean Economy Control (Baseline Distribution)",
    "Custom Stress Parameters"
]

selected_preset = st.sidebar.selectbox("Select Macroeconomic Scenario", preset_options)

if selected_preset == "Severe Recession Shock (-20 FICO & +10% Defaults)":
    drift_score = -20
    drift_default = 0.10
    sc_desc = "Compound crisis: downward borrower credit migration combined with severe systemic default rate escalation."
elif selected_preset == "Systemic Default Surge (+10% Defaults)":
    drift_score = 0
    drift_default = 0.10
    sc_desc = "Credit degradation shock: identical applicant credit scores experience a +10% default surge due to macroeconomic hardship."
elif selected_preset == "Population Drift (-20 FICO Migration)":
    drift_score = -20
    drift_default = 0.0
    sc_desc = "Macroeconomic credit migration: applicant scores systematically shift downward by 20 FICO points."
elif selected_preset == "Clean Economy Control (Baseline Distribution)":
    drift_score = 0
    drift_default = 0.0
    sc_desc = "Control environment: stable historical distributions with zero artificial macroeconomic distortions."
else:
    st.sidebar.markdown("---")
    drift_score = st.sidebar.slider("FICO Score Shift (Points)", min_value=-50, max_value=50, value=-20, step=5)
    drift_default = st.sidebar.slider("Default Probability Shift (%)", min_value=-0.05, max_value=0.20, value=0.10, step=0.01)
    sc_desc = f"Custom Stress Experiment: FICO shift = {drift_score:+d} pts, Default shift = {drift_default*100:+.1f}%."

sim_weeks = st.sidebar.slider("Simulation Horizon (Weeks)", min_value=12, max_value=48, value=24, step=4)
online_lr = st.sidebar.selectbox("Online Test Adaptation Rate (α)", [0.002, 0.005, 0.008, 0.010], index=2)
random_seed = st.sidebar.number_input("Cohort Seed", min_value=1, max_value=999999, value=42)

st.sidebar.markdown("---")
st.sidebar.markdown("**Static Baseline:** Fixed FICO 760 Cutoff")
st.sidebar.markdown("**RL Agent:** 4D State Adaptive Policy")

# -----------------------------------------------------------------------------
# Main Tabs
# -----------------------------------------------------------------------------
tab_sim, tab_results, tab_figures, tab_architecture = st.tabs([
    "🚀 Live Interactive Simulation", 
    "📊 Research Benchmarks & Table 1", 
    "🖼️ Publication Figures Gallery", 
    "🧠 4D Architecture & Methodology"
])

# -----------------------------------------------------------------------------
# TAB 1: Live Interactive Simulation
# -----------------------------------------------------------------------------
with tab_sim:
    st.info(f"**Current Scenario:** {selected_preset} — {sc_desc}")
    
    thresholds = [660 + a * 5 for a in range(33)] + [9999]
    action_dim = len(thresholds)
    optimal_static_cutoff = 760
    best_action_static = 20  # Action 20 corresponds to FICO 760
    
    # Run Simulation
    @st.cache_data(show_spinner=False)
    def execute_simulation(score_shift, default_shift, weeks, lr, seed):
        env = CreditCutoffEnv(
            mode='test',
            drift_score_shift=score_shift,
            drift_default_shift=default_shift,
            max_weeks=weeks
        )
        
        # Load or initialize base agent
        base_agent = DQNAgent(state_dim=4, action_dim=action_dim, lr=2e-3, gamma=0.0)
        q_weights_path = os.path.join(os.path.dirname(__file__), "Source", "q_network.pth")
        if os.path.exists(q_weights_path):
            base_agent.q_network.load_state_dict(torch.load(q_weights_path, map_location=torch.device('cpu')))
            base_agent.target_network.load_state_dict(base_agent.q_network.state_dict())
            
        # 1. Static Baseline Run
        state_b, _ = env.reset(seed=seed, options={'start_idx': (seed * 181) % 6000})
        b_rewards, b_cutoffs, b_acc, b_defaults, b_accepted = [], [], [], 0, 0
        truncated = False
        while not truncated:
            state_b, r, term, truncated, info = env.step(best_action_static)
            b_rewards.append(r)
            b_cutoffs.append(optimal_static_cutoff)
            b_acc.append(info['acceptance_rate'])
            b_accepted += info['num_accepted']
            b_defaults += info['num_defaults']
            
        # 2. Adaptive RL Run with Online Updates
        agent = DQNAgent(state_dim=4, action_dim=action_dim, lr=lr, gamma=0.0)
        agent.rbf = base_agent.rbf
        agent.q_network.load_state_dict(base_agent.q_network.state_dict())
        agent.target_network.load_state_dict(base_agent.target_network.state_dict())
        
        state_rl, _ = env.reset(seed=seed, options={'start_idx': (seed * 181) % 6000})
        rl_rewards, rl_cutoffs, rl_acc, rl_defaults, rl_accepted = [], [], [], 0, 0
        truncated = False
        while not truncated:
            act = agent.select_action(state_rl, evaluate=True)
            cutoff_val = thresholds[act]
            rl_cutoffs.append(cutoff_val)
            next_state, r, term, truncated, info = env.step(act)
            
            # Online test-time adaptation
            agent.store_transition(state_rl, act, r / 10000.0, next_state, truncated)
            agent.update(batch_size=1)
            
            state_rl = next_state
            rl_rewards.append(r)
            rl_acc.append(info['acceptance_rate'])
            rl_accepted += info['num_accepted']
            rl_defaults += info['num_defaults']
            
        return {
            "b_rewards": b_rewards,
            "rl_rewards": rl_rewards,
            "b_cutoffs": b_cutoffs,
            "rl_cutoffs": rl_cutoffs,
            "b_acc": b_acc,
            "rl_acc": rl_acc,
            "b_def_rate": (b_defaults / b_accepted * 100.0) if b_accepted > 0 else 0.0,
            "rl_def_rate": (rl_defaults / rl_accepted * 100.0) if rl_accepted > 0 else 0.0,
            "b_total_profit": sum(b_rewards),
            "rl_total_profit": sum(rl_rewards),
            "b_accepted": b_accepted,
            "rl_accepted": rl_accepted
        }

    sim_res = execute_simulation(drift_score, drift_default, sim_weeks, online_lr, random_seed)
    
    profit_diff = sim_res["rl_total_profit"] - sim_res["b_total_profit"]
    pct_imp = (profit_diff / abs(sim_res["b_total_profit"])) * 100.0 if sim_res["b_total_profit"] != 0 else 0.0
    
    # Metric KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Static Baseline Profit</div>
            <div class="metric-value">${sim_res['b_total_profit']:,.0f}</div>
            <div style="font-size:0.8rem; color:#64748B;">Fixed FICO {optimal_static_cutoff} Cutoff</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">RL Agent Profit</div>
            <div class="metric-value">${sim_res['rl_total_profit']:,.0f}</div>
            <div class="{'metric-sub' if profit_diff >= 0 else 'metric-sub-neg'}">{profit_diff:+,.0f} ({pct_imp:+.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Portfolio Default Rate</div>
            <div class="metric-value">{sim_res['rl_def_rate']:.1f}%</div>
            <div style="font-size:0.8rem; color:#64748B;">Baseline: {sim_res['b_def_rate']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        avg_cutoff = np.mean(sim_res['rl_cutoffs'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Cutoff Used</div>
            <div class="metric-value">FICO {avg_cutoff:.0f}</div>
            <div style="font-size:0.8rem; color:#64748B;">Baseline: FICO 760</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Simulation Charts (Matplotlib High-DPI Plot)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, dpi=180)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    weeks_arr = np.arange(1, len(sim_res['b_rewards']) + 1)
    cum_b = np.cumsum(sim_res['b_rewards']) / 1000.0
    cum_rl = np.cumsum(sim_res['rl_rewards']) / 1000.0
    
    # Cumulative Profit Trajectory
    ax1.plot(weeks_arr, cum_b, color='#1E40AF', linestyle='--', linewidth=2.5, label=f'Static Baseline (Fixed FICO {optimal_static_cutoff})')
    ax1.plot(weeks_arr, cum_rl, color='#EA580C', linewidth=2.8, label='Adaptive 4D RL Agent (Online SGD)')
    ax1.set_title('Cumulative Portfolio Profit Dynamics (in Thousands $)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Profit ($k)', fontsize=10, fontweight='bold')
    ax1.legend(loc='lower left' if np.mean(cum_b) < 0 else 'upper left', fontsize=9.5)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Policy Cutoff Dynamics
    ax2.axhline(optimal_static_cutoff, color='#1E40AF', linestyle='--', linewidth=2.2, label=f'Static Baseline (Fixed FICO {optimal_static_cutoff})')
    ax2.plot(weeks_arr, sim_res['rl_cutoffs'], color='#EA580C', marker='o', linewidth=2.4, label='Adaptive Policy Threshold (Cutoff Used)')
    ax2.set_title('Weekly Credit Approval Cutoff Policy Adaptation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Simulation Weeks Horizon', fontsize=10, fontweight='bold')
    ax2.set_ylabel('FICO Cutoff Threshold', fontsize=10, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9.5)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# TAB 2: Research Benchmarks & Table 1
# -----------------------------------------------------------------------------
with tab_results:
    st.subheader("📋 Table 1: Formal Research Publication Benchmark (100 Monte Carlo Runs)")
    st.markdown("""
    This table presents the rigorous statistical hypothesis testing conducted over **100 independent simulation runs** 
    on the 100,000-record LendingClub dataset across 4 macroeconomic distortion scenarios.
    """)
    
    csv_path = os.path.join(os.path.dirname(__file__), "Source", "research_benchmark_results.csv")
    if os.path.exists(csv_path):
        df_bench = pd.read_csv(csv_path)
        
        display_df = pd.DataFrame({
            "Scenario": df_bench["Scenario_ID"] + ": " + df_bench["Scenario_Name"],
            "Description": df_bench["Scenario_Sub"],
            "Static Profit": df_bench["Mean_Baseline_Profit"].apply(lambda x: f"${x:,.0f}"),
            "RL Profit": df_bench["Mean_RL_Profit"].apply(lambda x: f"${x:,.0f}"),
            "Net Advantage (Δ)": df_bench["Mean_Profit_Difference"].apply(lambda x: f"{x:+,.0f}"),
            "Paired t-Stat": df_bench["t_statistic"].apply(lambda x: f"{x:.2f}"),
            "p-Value": df_bench["p_value"].apply(lambda x: f"{x:.2e}"),
            "Cohen's d": df_bench["Cohens_d"].apply(lambda x: f"{x:.2f}"),
            "95% Confidence Interval": df_bench.apply(lambda r: f"[${r['CI_95_Lower']:,.0f}, ${r['CI_95_Upper']:,.0f}]", axis=1),
            "Significance": df_bench["p_value"].apply(lambda p: "p < 0.001 ***" if p < 0.001 else ("p < 0.05 *" if p < 0.05 else "Not Sig."))
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Benchmark CSV not found. Please run Source/research_benchmark.py to generate results.")
        
    st.markdown("""
    #### Key Statistical Findings:
    1. **Systemic Default Shock (Scenario 3):** Under a +10% default surge, the RL agent achieves a **+$26,619 capital preservation advantage** over the static cutoff ($t = 6.08, p = 1.14 \times 10^{-8}$), establishing overwhelming statistical significance ($p < 0.001$).
    2. **Severe Compound Recession (Scenario 4):** Under dual credit rating migration (-20 FICO) and default shocks (+10%), the RL agent saves **+$21,410 in portfolio losses** ($t = 5.90, p = 2.51 \times 10^{-8}$), demonstrating robust crisis resilience.
    """)

# -----------------------------------------------------------------------------
# TAB 3: Publication Figures Gallery
# -----------------------------------------------------------------------------
with tab_figures:
    st.subheader("🖼️ Project Figures Gallery (Sequential Numbering)")
    st.markdown("All 6 publication-grade figures generated by the CreditQ benchmarking pipeline:")
    
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        f1_path = os.path.join(os.path.dirname(__file__), "Figures", "fig_01_baseline_threshold_optimization.png")
        if os.path.exists(f1_path):
            st.image(f1_path, caption="Figure 1: Baseline Static Cutoff Optimization across FICO Thresholds on Train Data")
            
        f3_path = os.path.join(os.path.dirname(__file__), "Figures", "fig_03_training_reward_convergence.png")
        if os.path.exists(f3_path):
            st.image(f3_path, caption="Figure 3: Reinforcement Learning Training Reward Convergence (100 Monte Carlo Episodes)")
            
        f5_path = os.path.join(os.path.dirname(__file__), "Figures", "fig_05_dynamic_policy_cutoff_tightening.png")
        if os.path.exists(f5_path):
            st.image(f5_path, caption="Figure 5: Policy Action Dynamics & Marginal Threshold Shift during Recession Shock")

    with fig_col2:
        f2_path = os.path.join(os.path.dirname(__file__), "Figures", "fig_02_learned_value_function_shape.png")
        if os.path.exists(f2_path):
            st.image(f2_path, caption="Figure 2: Learned Q-Value Function Shape Alignment with Static Potential Profit Curve")
            
        f4_path = os.path.join(os.path.dirname(__file__), "Figures", "fig_04_monte_carlo_statistical_benchmarks.png")
        if os.path.exists(f4_path):
            st.image(f4_path, caption="Figure 4: 100-Run Monte Carlo Benchmark Trajectories & Empirical Advantage Distributions")
            
        f6_path = os.path.join(os.path.dirname(__file__), "Figures", "fig_06_recession_capital_preservation.png")
        if os.path.exists(f6_path):
            st.image(f6_path, caption="Figure 6: Cumulative P&L Comparison and Policy Tightening in Severe Crisis Shock")

# -----------------------------------------------------------------------------
# TAB 4: Architecture & Methodology
# -----------------------------------------------------------------------------
with tab_architecture:
    st.subheader("🧠 Mathematical Formulation & 4D State Representation")
    
    st.markdown("""
    ### 1. The 1D State Aliasing Failure ("The Acceptance Rate Trap")
    In the original research paper, the environment state was modeled as a 1D scalar:
    $$S_t = [\\text{acceptance\\_rate}_{t-1}]$$
    
    **The Discovery:** On authentic, continuous credit distributions (LendingClub), this 1D representation suffers from **severe partial observability**. 
    When population drift occurs (-20 FICO drop), fewer applicants pass the cutoff, lowering the weekly acceptance rate. 
    The naive 1D agent misinterprets this drop as *"my threshold is set too strictly"* and paradoxically **loosens its cutoff into toxic default territory** right when macro risk is escalating!

    ### 2. The 4D State Space Solution
    We resolved state aliasing by engineering a multi-dimensional state vector with active-range scaling into $[0.0, 1.0]^4$:
    $$\\mathbf{S}_t = \\left[ \\text{norm\\_acc\\_rate}_{t-1}, \\; \\frac{\\text{batch\\_mean\\_fico}_t - 650}{150}, \\; \\text{norm\\_rej\\_rate}_{t-1}, \\; \\frac{\\text{prev\\_default\\_rate}_{t-1}}{0.25} \\right]$$
    
    * **Batch Mean FICO:** Grants the agent proactive foresight of incoming applicant credit migration before loans are issued.
    * **Previous Default Rate (Smooth EMA):** Preserves latent default shock memory even when lending volume is temporarily frozen, eliminating ping-pong policy oscillations.

    ### 3. Model Architecture & Value Function Approximation
    * **Feature Projection:** 2,000-dimensional Gaussian Radial Basis Function (RBF) kernel via Scikit-Learn `RBFSampler` ($\gamma = 2.0$).
    * **Policy Heads:** Action-specific Linear SGD layers over 34 discrete actions (FICO thresholds 660–820 in steps of 5, plus an emergency `STOP_LENDING` action).
    * **Online Test-Time Adaptation:** Weekly online SGD update step ($\alpha = 0.008$, batch size 1) enabling continuous tracking of macroeconomic regimes.
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748B; font-size: 0.85rem;'>CreditQ • Autonomous RL Credit Decisioning System • MIT License</div>", unsafe_allow_html=True)
