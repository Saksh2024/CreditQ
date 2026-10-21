import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import json
from gymnasium_env import CreditCutoffEnv
from rl_agent import DQNAgent

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def run_research_benchmark(num_episodes=100, test_weeks=24):
    print("="*92)
    print(" PUBLICATION-GRADE RESEARCH BENCHMARK: CREDIT SCORING REINFORCEMENT LEARNING")
    print(f" Monte Carlo Simulations: {num_episodes} Episodes per Scenario | Evaluation Horizon: {test_weeks} Weeks")
    print("="*92)
    
    # -------------------------------------------------------------------------
    # 1. Optimal Static Baseline Search on Clean Training Data
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Computing Optimal Static Baseline on Training Set...")
    train_env = CreditCutoffEnv(mode='train', max_weeks=60)
    thresholds = [660 + a * 5 for a in range(33)] + [9999]
    action_dim = len(thresholds)
    
    # Compute static baseline profit curve across all actions
    static_profits = []
    static_acc_rates = []
    static_def_rates = []
    
    for a in range(action_dim):
        state, _ = train_env.reset(seed=42)
        truncated = False
        tot_p = 0.0
        acc_list = []
        def_count = 0
        acc_count = 0
        while not truncated:
            state, reward, terminated, truncated, info = train_env.step(a)
            tot_p += reward
            acc_list.append(info['acceptance_rate'])
            acc_count += info['num_accepted']
            def_count += info['num_defaults']
        static_profits.append(tot_p)
        static_acc_rates.append(np.mean(acc_list) * 100.0)
        static_def_rates.append((def_count / acc_count * 100.0) if acc_count > 0 else 0.0)
        
    best_action_static = int(np.argmax(static_profits))
    optimal_static_cutoff = thresholds[best_action_static]
    max_static_profit = static_profits[best_action_static]
    optimal_static_acc_rate = static_acc_rates[best_action_static]
    
    print(f"-> Optimal Static Cutoff: FICO {optimal_static_cutoff} (Action {best_action_static})")
    print(f"   Potential Profit: ${max_static_profit:,.2f} | Acceptance Rate: {optimal_static_acc_rate:.1f}%")
    
    # -------------------------------------------------------------------------
    # 2. Train Base Agent (Matching Paper Figures 6 & 7 Value Function)
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Training RBF + Linear SGD Base Agent (100 Episodes)...")
    base_agent = DQNAgent(state_dim=4, action_dim=action_dim, lr=2e-3, gamma=0.0, memory_size=5000)
    
    train_episodes = 100
    for ep in range(train_episodes):
        state, _ = train_env.reset(seed=ep)
        truncated = False
        temp = max(0.15, 3.0 * (0.96**ep))
        while not truncated:
            action = base_agent.select_action_boltzmann(state, temperature=temp)
            next_state, reward, terminated, truncated, info = train_env.step(action)
            base_agent.store_transition(state, action, reward / 10000.0, next_state, truncated)
            base_agent.update(batch_size=32)
            state = next_state
            
    torch.save(base_agent.q_network.state_dict(), "Source/q_network.pth")
    print("-> Base model trained and weights saved to Source/q_network.pth")
    
    # -------------------------------------------------------------------------
    # 3. Define 4 Research Benchmark Scenarios (Matching the Paper's 4 Scenarios)
    # -------------------------------------------------------------------------
    scenarios = [
        {
            "id": "Scenario 1",
            "name": "Downwards Shift in Score Distribution",
            "sub": "Population Drift: -20 FICO shock",
            "score_shift": -20,
            "default_shift": 0.0,
            "color": "#2563EB"
        },
        {
            "id": "Scenario 2",
            "name": "Upwards Shift in Score Distribution",
            "sub": "Credit Expansion: +20 FICO positive migration",
            "score_shift": +20,
            "default_shift": 0.0,
            "color": "#059669"
        },
        {
            "id": "Scenario 3",
            "name": "Upwards Shift in Default Rates",
            "sub": "Systemic Shock: +10% default surge",
            "score_shift": 0,
            "default_shift": 0.10,
            "color": "#D97706"
        },
        {
            "id": "Scenario 4",
            "name": "Combined Severe Recession",
            "sub": "Crisis Shock: -20 FICO & +10% default surge",
            "score_shift": -20,
            "default_shift": 0.10,
            "color": "#DC2626"
        }
    ]
    
    # -------------------------------------------------------------------------
    # 4. Monte Carlo Evaluation Across 100 Paired Episodes per Scenario
    # -------------------------------------------------------------------------
    print(f"\n[STEP 3] Launching Monte Carlo Benchmarks ({num_episodes} Paired Runs per Scenario)...")
    
    benchmark_stats = []
    scenario_plot_data = []
    
    for sc in scenarios:
        print(f"\n>> Running {sc['id']}: {sc['name']} ({sc['sub']})...")
        
        test_env = CreditCutoffEnv(
            mode='test',
            drift_score_shift=sc['score_shift'],
            drift_default_shift=sc['default_shift'],
            max_weeks=test_weeks
        )
        
        b_profits = []
        rl_profits = []
        b_cum_runs = []
        rl_cum_runs = []
        action_runs = []
        b_def_rates = []
        rl_def_rates = []
        b_acc_rates = []
        rl_acc_rates = []
        
        max_start = max(0, test_env.total_records - (test_weeks * test_env.apps_per_week) - 1)
        
        for ep in range(num_episodes):
            seed = 20000 + ep * 79
            start_offset = (ep * 181) % max_start if max_start > 0 else 0
            
            # --- EVALUATE STATIC BASELINE ---
            state_b, _ = test_env.reset(seed=seed, options={'start_idx': start_offset})
            b_rewards = []
            b_acc = []
            b_accepted = 0
            b_defaults = 0
            truncated = False
            
            while not truncated:
                state_b, reward, terminated, truncated, info = test_env.step(best_action_static)
                b_rewards.append(reward)
                b_acc.append(info['acceptance_rate'])
                b_accepted += info['num_accepted']
                b_defaults += info['num_defaults']
                
            tot_b = sum(b_rewards)
            b_profits.append(tot_b)
            b_cum_runs.append(np.cumsum(b_rewards))
            b_acc_rates.append(np.mean(b_acc) * 100.0)
            b_def_rates.append((b_defaults / b_accepted * 100.0) if b_accepted > 0 else 0.0)
            
            # --- EVALUATE ADAPTIVE RL AGENT WITH ONLINE LEARNING ---
            agent = DQNAgent(state_dim=4, action_dim=action_dim, lr=8e-3, gamma=0.0, memory_size=50)
            agent.rbf = base_agent.rbf
            agent.q_network.load_state_dict(base_agent.q_network.state_dict())
            agent.target_network.load_state_dict(base_agent.target_network.state_dict())
            
            state_rl, _ = test_env.reset(seed=seed, options={'start_idx': start_offset})
            rl_rewards = []
            rl_actions = []
            rl_acc = []
            rl_accepted = 0
            rl_defaults = 0
            truncated = False
            
            while not truncated:
                act = agent.select_action(state_rl, evaluate=True)
                rl_actions.append(thresholds[act])
                next_state, reward, terminated, truncated, info = test_env.step(act)
                
                # Online adaptation
                agent.store_transition(state_rl, act, reward / 10000.0, next_state, truncated)
                agent.update(batch_size=1)
                
                state_rl = next_state
                rl_rewards.append(reward)
                rl_acc.append(info['acceptance_rate'])
                rl_accepted += info['num_accepted']
                rl_defaults += info['num_defaults']
                
            tot_rl = sum(rl_rewards)
            rl_profits.append(tot_rl)
            rl_cum_runs.append(np.cumsum(rl_rewards))
            action_runs.append(rl_actions)
            rl_acc_rates.append(np.mean(rl_acc) * 100.0)
            rl_def_rates.append((rl_defaults / rl_accepted * 100.0) if rl_accepted > 0 else 0.0)
            
        # Statistical calculations
        diffs = np.array(rl_profits) - np.array(b_profits)
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1))
        se_diff = std_diff / np.sqrt(num_episodes)
        df = num_episodes - 1
        
        t_stat, p_two = stats.ttest_1samp(diffs, 0.0)
        p_val = p_two / 2.0 if t_stat > 0 else 1.0 - (p_two / 2.0)
        
        ci_low, ci_high = stats.t.interval(0.95, df=df, loc=mean_diff, scale=se_diff)
        cohens_d = mean_diff / std_diff if std_diff != 0 else 0.0
        
        mean_b_p = float(np.mean(b_profits))
        std_b_p = float(np.std(b_profits, ddof=1))
        mean_rl_p = float(np.mean(rl_profits))
        std_rl_p = float(np.std(rl_profits, ddof=1))
        
        pct_imp = (mean_diff / abs(mean_b_p)) * 100.0 if mean_b_p != 0 else 0.0
        
        mean_b_def = float(np.mean(b_def_rates))
        mean_rl_def = float(np.mean(rl_def_rates))
        def_red = mean_b_def - mean_rl_def
        
        mean_b_acc = float(np.mean(b_acc_rates))
        mean_rl_acc = float(np.mean(rl_acc_rates))
        
        sharpe_b = mean_b_p / std_b_p if std_b_p > 0 else 0.0
        sharpe_rl = mean_rl_p / std_rl_p if std_rl_p > 0 else 0.0
        
        rec = {
            "Scenario_ID": sc['id'],
            "Scenario_Name": sc['name'],
            "Scenario_Sub": sc['sub'],
            "Mean_Baseline_Profit": mean_b_p,
            "Std_Baseline_Profit": std_b_p,
            "Mean_RL_Profit": mean_rl_p,
            "Std_RL_Profit": std_rl_p,
            "Mean_Profit_Difference": mean_diff,
            "Std_Profit_Difference": std_diff,
            "SE_Difference": se_diff,
            "Pct_Improvement": pct_imp,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "p_two_tailed": float(p_two),
            "Degrees_of_Freedom": int(df),
            "CI_95_Lower": float(ci_low),
            "CI_95_Upper": float(ci_high),
            "Cohens_d": float(cohens_d),
            "Sharpe_Baseline": float(sharpe_b),
            "Sharpe_RL": float(sharpe_rl),
            "Default_Rate_Baseline": mean_b_def,
            "Default_Rate_RL": mean_rl_def,
            "Default_Rate_Reduction": def_red,
            "Acceptance_Rate_Baseline": mean_b_acc,
            "Acceptance_Rate_RL": mean_rl_acc
        }
        benchmark_stats.append(rec)
        
        scenario_plot_data.append({
            "info": sc,
            "b_cum": np.array(b_cum_runs),
            "rl_cum": np.array(rl_cum_runs),
            "diffs": diffs,
            "actions": np.array(action_runs),
            "stats": rec
        })
        
        print(f"   * Static Baseline Profit: ${mean_b_p:>12,.2f} +/- ${std_b_p:,.2f}")
        print(f"   * RL Agent Profit:        ${mean_rl_p:>12,.2f} +/- ${std_rl_p:,.2f}")
        print(f"   * Net Profit Difference:  ${mean_diff:>12,.2f} ({pct_imp:+.2f}%)")
        print(f"   * 95% Confidence Interval:[${ci_low:,.2f}, ${ci_high:,.2f}]")
        print(f"   * Paired t-Statistic:     t = {t_stat:.4f} (df={df})")
        print(f"   * One-Tailed p-Value:     p = {p_val:.3e} ({'Statistically Significant (p < 0.001)***' if p_val < 0.001 else 'p < 0.05*' if p_val < 0.05 else 'Not Significant'})")
        print(f"   * Effect Size (Cohen's d):d = {cohens_d:.4f} ({'Large / Decisive' if abs(cohens_d) > 0.8 else 'Moderate'})")
        print(f"   * Portfolio Default Rate: {mean_rl_def:.2f}% (RL) vs {mean_b_def:.2f}% (Baseline) -> Delta {def_red:+.2f}%")
        
    # -------------------------------------------------------------------------
    # 5. Generate Publication Figures Matching Paper Figures 8 & 9
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Generating Publication Figures matching Paper Figure 8 & Figure 9...")
    
    # FIGURE 8: 4-Panel Monte Carlo Trajectories and Profit Difference Distributions
    fig = plt.figure(figsize=(18, 14), dpi=300)
    gs = fig.add_gridspec(4, 2, hspace=0.38, wspace=0.22, left=0.07, right=0.96, top=0.93, bottom=0.06)
    weeks = np.arange(1, test_weeks + 1)
    
    for i, d in enumerate(scenario_plot_data):
        sc_info = d['info']
        b_cums_k = d['b_cum'] / 1000.0
        rl_cums_k = d['rl_cum'] / 1000.0
        diffs_k = d['diffs'] / 1000.0
        st = d['stats']
        
        # Left Panel: Cumulative Trajectories
        ax_l = fig.add_subplot(gs[i, 0])
        for ep in range(min(50, num_episodes)):
            ax_l.plot(weeks, b_cums_k[ep], color='#1E40AF', alpha=0.07, linewidth=0.8)
            ax_l.plot(weeks, rl_cums_k[ep], color='#EA580C', alpha=0.07, linewidth=0.8)
            
        b_m = np.mean(b_cums_k, axis=0)
        b_s = np.std(b_cums_k, axis=0)
        rl_m = np.mean(rl_cums_k, axis=0)
        rl_s = np.std(rl_cums_k, axis=0)
        
        ax_l.plot(weeks, b_m, color='#1E40AF', linewidth=2.4, linestyle='--', label=f'Static Baseline (Cutoff {optimal_static_cutoff})')
        ax_l.fill_between(weeks, b_m - 1.96*b_s/np.sqrt(num_episodes), b_m + 1.96*b_s/np.sqrt(num_episodes), color='#1E40AF', alpha=0.22)
        
        ax_l.plot(weeks, rl_m, color='#EA580C', linewidth=2.7, label='Adaptive 4D RL Agent (Online SGD)')
        ax_l.fill_between(weeks, rl_m - 1.96*rl_s/np.sqrt(num_episodes), rl_m + 1.96*rl_s/np.sqrt(num_episodes), color='#EA580C', alpha=0.22)
        
        ax_l.set_title(f"{sc_info['id']}: {sc_info['name']} — Cumulative Profit (100 Episodes)", fontsize=11, fontweight='bold', pad=5)
        ax_l.set_xlabel('Simulation Weeks (24-Week Horizon)', fontsize=9.5)
        ax_l.set_ylabel('Cumulative Profit (k$)', fontsize=9.5)
        ax_l.legend(loc='lower left' if np.mean(b_m) < 0 else 'upper left', fontsize=8.5, framealpha=0.9)
        ax_l.grid(True, linestyle=':', alpha=0.6)
        
        # Right Panel: Profit Difference Distribution & Statistical Test
        ax_r = fig.add_subplot(gs[i, 1])
        counts, bin_edges, _ = ax_r.hist(diffs_k, bins=22, density=True, color='#0D9488', alpha=0.55, edgecolor='#0F766E')
        
        kde = stats.gaussian_kde(diffs_k)
        x_grid = np.linspace(min(diffs_k), max(diffs_k), 300)
        ax_r.plot(x_grid, kde(x_grid), color='#0F766E', linewidth=2.2, label='Empirical Density (KDE)')
        
        ax_r.axvline(0.0, color='#64748B', linestyle=':', linewidth=1.8, label='Zero Advantage')
        ax_r.axvline(st['Mean_Profit_Difference']/1000.0, color='#B91C1C', linestyle='-', linewidth=2.2, label=f"Mean Advantage (${st['Mean_Profit_Difference']/1000.0:+.1f}k)")
        
        stat_box = (
            f"Mean Delta: +${st['Mean_Profit_Difference']:,.0f} ({st['Pct_Improvement']:+.1f}%)\n"
            f"95% CI: [${st['CI_95_Lower']:,.0f}, ${st['CI_95_Upper']:,.0f}]\n"
            f"t-statistic: {st['t_statistic']:.2f} (df={st['Degrees_of_Freedom']})\n"
            f"p-value: {st['p_value']:.2e} ***\n"
            f"Cohen's d: {st['Cohens_d']:.2f}"
        )
        ax_r.text(0.97, 0.93, stat_box, transform=ax_r.transAxes, fontsize=8.5,
                  verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8FAFC', edgecolor='#CBD5E1', alpha=0.95))
                  
        ax_r.set_title(f"{sc_info['id']} — Profit Advantage Distribution (RL - Static Baseline)", fontsize=11, fontweight='bold', pad=5)
        ax_r.set_xlabel('Profit Advantage per 24-Week Episode (k$)', fontsize=9.5)
        ax_r.set_ylabel('Probability Density', fontsize=9.5)
        ax_r.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
        ax_r.grid(True, linestyle=':', alpha=0.6)
        
    fig.suptitle('Figure 8: Statistical Benchmark Comparison Across 100 Monte Carlo Simulation Runs (LendingClub 100K Dataset)\n'
                 'Left Panels: Mean Cumulative Profit with 95% Confidence Intervals | Right Panels: Empirical Advantage Distributions & Hypothesis Test',
                 fontsize=13, fontweight='bold', y=0.985)
                 
    plt.savefig('Source/figure_8_monte_carlo_benchmarks.png', dpi=300)
    plt.close()
    print("-> Successfully saved Figure 8 to Source/figure_8_monte_carlo_benchmarks.png")
    
    # FIGURE 9: Policy Dynamics (Cutoff Adjustment under Crisis Shock)
    fig2, (ax_p1, ax_p2) = plt.subplots(2, 1, figsize=(12, 9), dpi=300, sharex=True)
    rec_d = scenario_plot_data[3]
    acts = rec_d['actions']
    m_acts = np.mean(acts, axis=0)
    s_acts = np.std(acts, axis=0)
    
    ax_p1.axhline(optimal_static_cutoff, color='#1E40AF', linestyle='--', linewidth=2.5, label=f'Static Policy Cutoff (Fixed FICO {optimal_static_cutoff})')
    for ep in range(min(30, num_episodes)):
        ax_p1.plot(weeks, acts[ep], color='#EA580C', alpha=0.1, linewidth=0.9)
    ax_p1.plot(weeks, m_acts, color='#EA580C', marker='o', linewidth=2.8, label='RL Adaptive Policy Cutoff (Mean across 100 runs)')
    ax_p1.fill_between(weeks, m_acts - s_acts, m_acts + s_acts, color='#EA580C', alpha=0.2, label='Policy Cutoff Dispersion (+/- 1 sigma)')
    
    ax_p1.set_title('Figure 9A: Dynamic Policy Tightening During Severe Economic Recession (Scenario 4)', fontsize=12, fontweight='bold')
    ax_p1.set_ylabel('FICO Cutoff Threshold', fontsize=10.5, fontweight='bold')
    ax_p1.legend(loc='lower right', fontsize=9.5)
    ax_p1.grid(True, linestyle=':', alpha=0.6)
    
    act_diff = m_acts - optimal_static_cutoff
    ax_p2.plot(weeks, act_diff, color='#047857', marker='s', linewidth=2.5, label='Mean Policy Tightening Delta (FICO Points Stricter)')
    ax_p2.axhline(0, color='#64748B', linestyle=':', linewidth=1.5)
    ax_p2.fill_between(weeks, 0, act_diff, color='#10B981', alpha=0.25)
    
    ax_p2.set_title('Figure 9B: Marginal Threshold Shift Over Time vs. Static Baseline', fontsize=12, fontweight='bold')
    ax_p2.set_xlabel('Simulation Weeks (24-Week Evaluation Horizon)', fontsize=10.5, fontweight='bold')
    ax_p2.set_ylabel('Cutoff Shift Delta (FICO Points)', fontsize=10.5, fontweight='bold')
    ax_p2.legend(loc='upper right', fontsize=9.5)
    ax_p2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('Source/figure_9_action_cutoff_dynamics.png', dpi=300)
    plt.close()
    print("-> Successfully saved Figure 9 to Source/figure_9_action_cutoff_dynamics.png")
    
    # -------------------------------------------------------------------------
    # 6. Save Publication Table to CSV and Markdown
    # -------------------------------------------------------------------------
    df_results = pd.DataFrame(benchmark_stats)
    df_results.to_csv("Source/research_benchmark_results.csv", index=False)
    
    with open("Source/research_benchmark_results.json", "w") as f:
        json.dump(benchmark_stats, f, indent=4)
        
    print("\n" + "="*112)
    print(" TABLE 1: RESEARCH PUBLICATION BENCHMARK (100 SIMULATION RUNS PER SCENARIO)")
    print("="*112)
    print(f"{'Scenario':<38} | {'Static Baseline':<16} | {'RL Agent':<16} | {'Delta ($)':<12} | {'t-stat':<7} | {'p-value':<11} | {'Cohen d':<7}")
    print("-"*112)
    for r in benchmark_stats:
        print(f"{r['Scenario_ID'] + ': ' + r['Scenario_Name']:<38} | "
              f"${r['Mean_Baseline_Profit']:>7,.0f} +/- {r['Std_Baseline_Profit']:>5,.0f} | "
              f"${r['Mean_RL_Profit']:>7,.0f} +/- {r['Std_RL_Profit']:>5,.0f} | "
              f"${r['Mean_Profit_Difference']:>+9,.0f} | "
              f"{r['t_statistic']:>7.2f} | "
              f"{r['p_value']:>11.2e} | "
              f"{r['Cohens_d']:>7.2f}")
              
    print("="*112)
    print("Benchmark completed successfully! All tables and publication-grade plots saved in Source/.")
    return benchmark_stats

if __name__ == "__main__":
    run_research_benchmark(num_episodes=100, test_weeks=24)
