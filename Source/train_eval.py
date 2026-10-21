import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from gymnasium_env import CreditCutoffEnv
from rl_agent import DQNAgent

# Set style for publication-quality charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def run_experiment():
    print("="*80)
    print(" CREDIT SCORING REINFORCEMENT LEARNING EXPERIMENT (LENDINGCLUB - 100,000 RECORDS)")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # PHASE 1: Baseline Static Threshold Optimization on Train Data
    # -------------------------------------------------------------------------
    print("\n[PHASE 1] Running Static Baseline Optimization across all 34 Actions (FICO 660..820 + STOP_LENDING)...")
    train_env = CreditCutoffEnv(mode='train', max_weeks=60)
    
    thresholds = [660 + a * 5 for a in range(33)] + [9999]
    baseline_profits = []
    baseline_acceptance_rates = []
    
    for a in range(34):
        state, _ = train_env.reset()
        truncated = False
        profit = 0.0
        acc_rates = []
        while not truncated:
            state, reward, terminated, truncated, info = train_env.step(a)
            profit += reward
            acc_rates.append(info['acceptance_rate'])
        baseline_profits.append(profit)
        baseline_acceptance_rates.append(np.mean(acc_rates) * 100.0)
        
    best_action_static = int(np.argmax(baseline_profits))
    optimal_static_cutoff = thresholds[best_action_static]
    max_static_profit = baseline_profits[best_action_static]
    optimal_static_acc_rate = baseline_acceptance_rates[best_action_static]
    
    print(f"Optimal Static Cutoff: FICO {optimal_static_cutoff} | Potential Profit: ${max_static_profit:,.2f} | Acceptance Rate: {optimal_static_acc_rate:.1f}%")
    
    # Plot Phase 1: Baseline Potential Profit & Acceptance Rate
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    l1 = ax1.plot(thresholds, np.array(baseline_profits)/1000.0, 'b-', linewidth=2.5, label='Potential Profit (k$)')
    l2 = ax2.plot(thresholds, baseline_acceptance_rates, 'r--', linewidth=2.0, label='Acceptance Rate (%)')
    ax1.axvline(x=optimal_static_cutoff, color='green', linestyle=':', linewidth=2, label=f'Optimal Cutoff ({optimal_static_cutoff})')
    
    ax1.set_xlabel('Acceptance Threshold (FICO Score)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Potential Profit ($ in Thousands)', color='b', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Acceptance Rate (%)', color='r', fontsize=12, fontweight='bold')
    plt.title('Figure 1: Baseline Acceptance Threshold Optimization on Train Dataset (LendingClub)', fontsize=14, fontweight='bold')
    
    lines = l1 + l2 + [plt.Line2D([0], [0], color='green', linestyle=':', linewidth=2, label=f'Optimum ({optimal_static_cutoff})')]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Source/1_baseline_optimization.png', dpi=300)
    plt.close()
    print("Saved Phase 1 plot to Source/1_baseline_optimization.png")
    
    # -------------------------------------------------------------------------
    # PHASE 2: Train RL Agent (Linear SGD + RBF) on Clean Train Data
    # -------------------------------------------------------------------------
    print("\n[PHASE 2] Training Reinforcement Learning Agent (Linear SGD with 2,000 RBF Features & 4D State)...")
    agent = DQNAgent(state_dim=4, action_dim=34, lr=1e-3, memory_size=1)
    
    episodes = 100
    current_train_temp = 5.0
    episode_rewards = []
    
    for ep in range(episodes):
        state, _ = train_env.reset()
        ep_reward = 0.0
        truncated = False
        while not truncated:
            action = agent.select_action_boltzmann(state, temperature=current_train_temp)
            next_state, reward, terminated, truncated, info = train_env.step(action)
            # Scale reward to stabilize linear SGD gradient step
            agent.store_transition(state, action, reward / 10000.0, next_state, truncated)
            agent.update(batch_size=1)
            state = next_state
            ep_reward += reward
            
        agent.update_target_network()
        episode_rewards.append(ep_reward)
        current_train_temp = max(0.1, current_train_temp * 0.95)
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1:3d}/{episodes} - Reward: ${ep_reward:,.2f} | Temp: {current_train_temp:.2f}")
            
    torch.save(agent.q_network.state_dict(), "Source/q_network.pth")
    print("Trained Q-Network saved to Source/q_network.pth")
    
    # Extract learned Q-values on typical operating acceptance rate
    ref_state = np.array([optimal_static_acc_rate / 100.0, 0.8, 1.0 - (optimal_static_acc_rate / 100.0), 0.10], dtype=np.float32)
    learned_q_values = agent.get_q_values(ref_state)
    best_action_rl = int(np.argmax(learned_q_values))
    optimal_rl_cutoff = thresholds[best_action_rl]
    
    print(f"\n--- Clean Train Comparison ---")
    print(f"Static Baseline Optimal Cutoff: FICO {optimal_static_cutoff}")
    print(f"RL Agent Learned Optimal Cutoff: FICO {optimal_rl_cutoff}")
    print(f"Cutoff Alignment: {'PERFECT MATCH' if optimal_static_cutoff == optimal_rl_cutoff else f'Near Match (Delta: {abs(optimal_static_cutoff - optimal_rl_cutoff)} pts)'}")
    
    # Plot Phase 2: Q-Value Function Shape vs. Baseline Profit
    norm_baseline = (np.array(baseline_profits) - np.min(baseline_profits)) / (np.max(baseline_profits) - np.min(baseline_profits))
    norm_q = (learned_q_values - np.min(learned_q_values)) / (np.max(learned_q_values) - np.min(learned_q_values))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, norm_baseline, 'b-', linewidth=2, label='Static Potential Profit (Normalized)')
    plt.plot(thresholds, norm_q, 'r--', linewidth=2.5, label='RL Learned Q-Value Function (Normalized)')
    plt.axvline(x=optimal_static_cutoff, color='b', linestyle=':', label=f'Static Optimum ({optimal_static_cutoff})')
    plt.axvline(x=optimal_rl_cutoff, color='r', linestyle=':', label=f'RL Optimum ({optimal_rl_cutoff})')
    plt.title('Figure 2: Learned Value Function Shape vs. Static Potential Profit Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Acceptance Threshold (FICO Score)', fontsize=12, fontweight='bold')
    plt.ylabel('Normalized Value / Expected Profit', fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Source/2_q_value_vs_baseline.png', dpi=300)
    plt.close()
    print("Saved Phase 2 value curve to Source/2_q_value_vs_baseline.png")
    
    # Plot Phase 2: Training Convergence
    plt.figure(figsize=(10, 5))
    plt.plot(np.array(episode_rewards)/1000.0, color='teal', linewidth=2)
    plt.title('Figure 3: Reinforcement Learning Training Convergence (100 Episodes)', fontsize=14, fontweight='bold')
    plt.xlabel('Training Episode (60 Weeks each)', fontsize=12, fontweight='bold')
    plt.ylabel('Episode Profit ($ in Thousands)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Source/3_training_convergence.png', dpi=300)
    plt.close()
    print("Saved Phase 2 training convergence to Source/3_training_convergence.png")
    
    # -------------------------------------------------------------------------
    # PHASE 3: 4 Experimental Scenarios on 24-Week Test Horizon
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print(" [PHASE 3] 24-WEEK TEST EVALUATION ACROSS 4 DRIFT & SHOCK SCENARIOS")
    print("="*80)
    
    scenarios = [
        {"name": "Scenario 1: Control (Clean Economy)", "score_shift": 0, "default_shift": 0.0},
        {"name": "Scenario 2: Population Drift (-20 FICO Shift)", "score_shift": -20, "default_shift": 0.0},
        {"name": "Scenario 3: Selection Bias (+10% Default Surge)", "score_shift": 0, "default_shift": 0.10},
        {"name": "Scenario 4: Combined Severe Recession (-20 FICO & +10% Defaults)", "score_shift": -20, "default_shift": 0.10}
    ]
    
    results = []
    fig_multi, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, sc in enumerate(scenarios):
        # Fresh test environment for 24 weeks
        test_env = CreditCutoffEnv(mode='test', drift_score_shift=sc['score_shift'], 
                                   drift_default_shift=sc['default_shift'], max_weeks=24)
        
        # 1. Evaluate Static Baseline on Test
        baseline_rewards = []
        state, _ = test_env.reset()
        truncated = False
        while not truncated:
            state, reward, terminated, truncated, info = test_env.step(best_action_static)
            baseline_rewards.append(reward)
            
        # 2. Evaluate RL Agent with Online Learning on Test
        test_agent = DQNAgent(state_dim=4, action_dim=34, lr=2e-3, memory_size=2000)
        test_agent.q_network.load_state_dict(agent.q_network.state_dict())
        test_agent.target_network.load_state_dict(agent.target_network.state_dict())
        
        rl_rewards = []
        rl_cutoffs = []
        state, _ = test_env.reset()
        truncated = False
        
        while not truncated:
            action = test_agent.select_action(state, evaluate=True)
            cutoff_val = thresholds[action]
            rl_cutoffs.append(cutoff_val)
            next_state, reward, terminated, truncated, info = test_env.step(action)
            
            # Online adaptation update every week
            test_agent.store_transition(state, action, reward / 10000.0, next_state, truncated)
            test_agent.update(batch_size=1)
            
            state = next_state
            rl_rewards.append(reward)
            
        total_b_profit = sum(baseline_rewards)
        total_rl_profit = sum(rl_rewards)
        profit_diff = total_rl_profit - total_b_profit
        pct_imp = (profit_diff / abs(total_b_profit)) * 100.0 if total_b_profit != 0 else 0.0
        mean_cutoff = np.mean(rl_cutoffs)
        
        results.append({
            "Scenario": sc['name'],
            "Baseline Profit": total_b_profit,
            "RL Agent Profit": total_rl_profit,
            "Difference ($)": profit_diff,
            "Improvement (%)": pct_imp,
            "Avg RL Cutoff": mean_cutoff
        })
        
        # Plot Scenario Subplot
        cum_b = np.cumsum(baseline_rewards) / 1000.0
        cum_rl = np.cumsum(rl_rewards) / 1000.0
        
        ax = axes[i]
        ax.plot(cum_b, 'b--', linewidth=2.2, label=f'Static Baseline (Cutoff {optimal_static_cutoff})')
        ax.plot(cum_rl, 'orange', linewidth=2.5, label='RL Agent (Adaptive Cutoff)')
        ax.set_title(sc['name'], fontsize=12, fontweight='bold')
        ax.set_xlabel('Simulation Weeks', fontsize=10)
        ax.set_ylabel('Cumulative Profit (k$)', fontsize=10)
        ax.legend(loc='lower left' if total_b_profit < 0 else 'upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Save scenario 4 as master plot
        if i == 3:
            fig_master, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Cumulative profit
            ax_top.plot(cum_b, 'b--', linewidth=2.5, label=f'Static Baseline (Cutoff {optimal_static_cutoff})')
            ax_top.plot(cum_rl, 'tab:orange', linewidth=2.8, label='RL Agent (Dynamic Cutoff)')
            ax_top.set_title('Figure 4A: Cumulative Profit under Severe Shock (-20 FICO & +10% Defaults)', fontsize=13, fontweight='bold')
            ax_top.set_ylabel('Cumulative Profit (k$)', fontsize=11, fontweight='bold')
            ax_top.legend(loc='best', fontsize=11)
            ax_top.grid(True, alpha=0.3)
            
            # Action dynamics (Cutoff over weeks)
            ax_bot.axhline(y=optimal_static_cutoff, color='b', linestyle='--', linewidth=2, label=f'Static Cutoff ({optimal_static_cutoff})')
            ax_bot.plot(rl_cutoffs, color='tab:orange', marker='o', linewidth=2.2, label='RL Dynamic Cutoff')
            ax_bot.set_title('Figure 4B: Weekly Credit Cutoff Dynamics (Policy Tightening)', fontsize=13, fontweight='bold')
            ax_bot.set_xlabel('Simulation Weeks (24-Week Test Period)', fontsize=11, fontweight='bold')
            ax_bot.set_ylabel('FICO Cutoff Threshold', fontsize=11, fontweight='bold')
            ax_bot.legend(loc='best', fontsize=11)
            ax_bot.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('Source/test_profit_comparison.png', dpi=300)
            plt.close()
            
    fig_multi.tight_layout()
    fig_multi.savefig('Source/4_drift_scenarios_comparison.png', dpi=300)
    plt.close()
    print("Saved 4-scenario multi-plot to Source/4_drift_scenarios_comparison.png")
    print("Saved Master test comparison to Source/test_profit_comparison.png")
    
    # -------------------------------------------------------------------------
    # PRINT QUANTIFIED SUMMARY TABLE
    # -------------------------------------------------------------------------
    df_res = pd.DataFrame(results)
    print("\n" + "="*80)
    print(" FINAL QUANTITATIVE EXPERIMENTAL RESULTS SUMMARY (24 WEEKS)")
    print("="*80)
    for res in results:
        print(f"\n>> {res['Scenario']}")
        print(f"   • Static Baseline Profit: ${res['Baseline Profit']:>12,.2f}")
        print(f"   • RL Agent Profit:        ${res['RL Agent Profit']:>12,.2f}")
        print(f"   • Net Improvement:        ${res['Difference ($)']:>12,.2f} ({res['Improvement (%)']:+.2f}%)")
        print(f"   • Avg RL Cutoff Used:      {res['Avg RL Cutoff']:>12.1f} (vs {optimal_static_cutoff} static)")

if __name__ == "__main__":
    run_experiment()


