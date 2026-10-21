import numpy as np
import pandas as pd
import torch
import sys
import os

sys.path.append(r"c:\Users\Sakshi\OneDrive\Desktop\CreditQ-main\CreditQ-main\Source")
from gymnasium_env import CreditCutoffEnv
from rl_agent import DQNAgent

def test_biweekly_evaluation():
    print("=" * 85)
    print("BI-WEEKLY EVALUATION BENCHMARK (39 BI-WEEKS = 78 WEEKS, 100 APPS / BI-WEEK)")
    print("Calibrated Loss Rate: 0.45 (Empirical LendingClub Recovery: 55%)")
    print("=" * 85)
    
    thresholds = [660 + a * 5 for a in range(33)] + [9999]
    action_dim = len(thresholds)
    
    # 1. Base training on clean data with bi-weekly batches (100 apps/step, 50 bi-weeks/episode = 100 weeks)
    train_env = CreditCutoffEnv(default_loss_rate=0.45, apps_per_week=100, mode='train', max_weeks=50)
    base_agent = DQNAgent(state_dim=4, action_dim=action_dim, lr=2e-3, gamma=0.0, memory_size=5000)
    
    print("Training Base RL Agent on clean data (bi-weekly cadence)...")
    for ep in range(50):
        state, _ = train_env.reset(seed=ep)
        truncated = False
        temp = max(0.15, 3.0 * (0.96**ep))
        while not truncated:
            action = base_agent.select_action_boltzmann(state, temperature=temp)
            next_state, reward, _, truncated, info = train_env.step(action)
            base_agent.store_counterfactual_batch(state, action, reward, next_state, truncated, info, default_loss_rate=0.45)
            base_agent.update(batch_size=32)
            state = next_state
            
    # Compute optimal static cutoff on clean training data
    static_profits = []
    for a in range(action_dim):
        train_env.reset(seed=42)
        tot_p = 0.0
        truncated = False
        while not truncated:
            _, r, _, truncated, _ = train_env.step(a)
            tot_p += r
        static_profits.append(tot_p)
    best_static_action = int(np.argmax(static_profits))
    print(f"Optimal Static Cutoff (Clean Training): FICO {thresholds[best_static_action]} (Action {best_static_action})")
    print(f"Static Training Profit: ${static_profits[best_static_action]:,.2f}")
    
    # Test across the 4 Scenarios (39 bi-weeks = 78 weeks)
    scenarios = [
        {"name": "Scenario 1 (Clean Control)", "score_shift": 0, "default_shift": 0.0},
        {"name": "Scenario 2 (Pop Drift: -20 FICO)", "score_shift": -20, "default_shift": 0.0},
        {"name": "Scenario 3 (Selection Bias: +10% Def)", "score_shift": 0, "default_shift": 0.10},
        {"name": "Scenario 4 (Severe Recession)", "score_shift": -20, "default_shift": 0.10},
    ]
    
    for sc in scenarios:
        test_env = CreditCutoffEnv(default_loss_rate=0.45, apps_per_week=100, mode='test', max_weeks=39,
                                  drift_score_shift=sc["score_shift"], drift_default_shift=sc["default_shift"])
        
        static_profits = []
        rl_profits = []
        rl_cutoffs = []
        sample_path = []
        
        max_start = max(0, test_env.total_records - (39 * 100) - 1)
        
        for ep in range(15):
            seed = 20000 + ep * 53
            offset = (ep * 163) % max_start if max_start > 0 else 0
            
            # Static Run
            test_env.reset(seed=seed, options={'start_idx': offset})
            tot_s = 0.0
            truncated = False
            while not truncated:
                _, r, _, truncated, _ = test_env.step(best_static_action)
                tot_s += r
            static_profits.append(tot_s)
            
            # Adaptive RL Run
            agent = DQNAgent(state_dim=4, action_dim=action_dim, lr=0.014, gamma=0.0, memory_size=500)
            agent.rbf = base_agent.rbf
            agent.q_network.load_state_dict(base_agent.q_network.state_dict())
            
            state, _ = test_env.reset(seed=seed, options={'start_idx': offset})
            tot_rl = 0.0
            truncated = False
            ep_cuts = []
            
            while not truncated:
                a = agent.select_action(state, evaluate=True)
                ep_cuts.append(thresholds[a])
                next_state, r, _, truncated, info = test_env.step(a)
                tot_rl += r
                
                # Bi-weekly dense counterfactual batch update across all actions
                agent.store_counterfactual_batch(state, a, r, next_state, truncated, info, default_loss_rate=0.45)
                agent.update(batch_size=min(len(agent.memory), 32))
                
                state = next_state
                
            rl_profits.append(tot_rl)
            rl_cutoffs.append(np.mean(ep_cuts))
            if ep == 0:
                sample_path = ep_cuts
                
        mean_s = np.mean(static_profits)
        mean_rl = np.mean(rl_profits)
        diff = mean_rl - mean_s
        pct = (diff / abs(mean_s)) * 100 if abs(mean_s) > 0 else 0.0
        
        print(f"\n[{sc['name']}] (39 Bi-Weeks / 78 Weeks)")
        print(f"  Static Mean Profit: ${mean_s:,.2f}")
        print(f"  RL Mean Profit:     ${mean_rl:,.2f}")
        print(f"  Advantage:          ${diff:,.2f} ({pct:+.2f}%)")
        print(f"  Avg RL Cutoff:      FICO {np.mean(rl_cutoffs):.1f} vs Static FICO {thresholds[best_static_action]}")
        print(f"  Bi-weekly Path:     {sample_path[:8]} ... {sample_path[-8:]}")

if __name__ == "__main__":
    test_biweekly_evaluation()
