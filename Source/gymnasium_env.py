import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

class CreditCutoffEnv(gym.Env):
    """
    Custom Gymnasium Environment for Credit Score Cutoff Optimization using LendingClub Data (100,000 records).
    Simulates a stream of weekly consumer credit applications.
    State: [acceptance_rate_{t-1}, incoming_batch_mean_fico / 850.0, batch_rejection_rate_{t-1}]
    Action: 25 discrete FICO score cutoffs mapping to scores [660, 665, 670, ..., 780].
    Reward: Weekly portfolio profit = sum(repaid loan value * interest_rate) - sum(defaulted loan value * default_loss_rate).
    """
    def __init__(self, default_loss_rate=0.45, apps_per_week=50, mode='train', 
                 drift_score_shift=0, drift_default_shift=0.0, max_weeks=78, seed=42):
        super(CreditCutoffEnv, self).__init__()
        
        # Load LendingClub Scored Dataset (100,000 records)
        data_path = os.path.join(os.path.dirname(__file__), "lending_club_scored.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"LendingClub scored dataset not found at {data_path}. Run prepare_lending_club.py first.")
            
        self.df = pd.read_csv(data_path)
        total_rows = len(self.df)
        split_idx = int(total_rows * 0.8)  # 80,000 train (1,600 weeks) / 20,000 test (400 weeks)
        
        if mode == 'train':
            self.df = self.df.iloc[:split_idx].reset_index(drop=True)
        else:
            self.df = self.df.iloc[split_idx:].reset_index(drop=True)
            
        # Extract features
        self.scores = self.df['CreditScore'].values.astype(np.float32)
        self.default_probs = self.df['PD'].values.astype(np.float32)
        self.loan_amounts = self.df['LoanAmount'].values.astype(np.float32)
        self.interest_rates = self.df['InterestRate'].values.astype(np.float32)
        
        self.total_records = len(self.scores)
        self.default_loss_rate = default_loss_rate
        self.apps_per_week = apps_per_week
        self.mode = mode
        self.drift_score_shift = drift_score_shift
        self.drift_default_shift = drift_default_shift
        self.max_weeks = max_weeks
        self.weeks_passed = 0
        
        # Action space: 34 discrete actions (0..32 -> FICO 660..820, 33 -> STOP_LENDING)
        self.action_space = spaces.Discrete(34)
        
        # Observation space: 4D vector [prev_acceptance_rate, batch_mean_fico/850.0, prev_rejection_rate, prev_default_rate]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self.current_idx = 0
        self.prev_acceptance_rate = 0.45 if default_loss_rate < 0.8 else 0.11
        self.prev_default_rate = 0.12 if default_loss_rate < 0.8 else 0.10
        
    def _get_cutoff_for_action(self, action):
        if action == 33:
            return 9999  # STOP_LENDING cutoff
        return 660 + action * 5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        start_idx = 0
        if options and 'start_idx' in options:
            start_idx = options['start_idx']
            
        self.current_idx = start_idx
        self.prev_acceptance_rate = 0.45 if self.default_loss_rate < 0.8 else 0.11
        self.prev_default_rate = 0.12 if self.default_loss_rate < 0.8 else 0.10
        self.weeks_passed = 0
        
        # Peek at initial batch score
        end_idx = min(self.current_idx + self.apps_per_week, self.total_records)
        init_scores = self.scores[self.current_idx:end_idx] + self.drift_score_shift
        norm_mean_fico = float(np.clip((np.mean(init_scores) - 650.0) / 150.0, 0.0, 1.0)) if len(init_scores) > 0 else 0.5
        norm_acc_rate = float(np.clip(self.prev_acceptance_rate, 0.0, 1.0))
        norm_rej_rate = 1.0 - norm_acc_rate
        norm_def_rate = float(np.clip(self.prev_default_rate, 0.0, 1.0))
        
        state = np.array([norm_acc_rate, norm_mean_fico, norm_rej_rate, norm_def_rate], dtype=np.float32)
        return state, {}

    def step(self, action):
        cutoff = self._get_cutoff_for_action(action)
        
        # Sample apps_per_week loan applications
        end_idx = min(self.current_idx + self.apps_per_week, self.total_records)
        if self.current_idx >= self.total_records:
            # Wrap around if data is exhausted in long training
            self.current_idx = 0
            end_idx = self.apps_per_week
            
        week_scores = self.scores[self.current_idx:end_idx].copy()
        week_probs = self.default_probs[self.current_idx:end_idx].copy()
        week_amounts = self.loan_amounts[self.current_idx:end_idx]
        week_rates = self.interest_rates[self.current_idx:end_idx]
        
        # Apply Population Drift (Shift in FICO score distribution)
        if self.drift_score_shift != 0:
            week_scores = np.clip(week_scores + self.drift_score_shift, 300, 850)
            
        # Apply Selection Bias / Default Shock (Shift in default probabilities)
        final_probs = np.clip(week_probs + self.drift_default_shift, 0.0, 1.0)
        week_defaults = np.random.binomial(n=1, p=final_probs)
        
        self.current_idx = end_idx
        total_apps = len(week_scores)
        
        if total_apps == 0:
            state = np.array([self.prev_acceptance_rate, 0.8, 1.0 - self.prev_acceptance_rate, self.prev_default_rate], dtype=np.float32)
            return state, 0.0, True, False, {}
            
        # Decision: accept if credit score >= cutoff
        accepted_mask = week_scores >= cutoff
        num_accepted = np.sum(accepted_mask)
        acceptance_rate = float(num_accepted) / total_apps if total_apps > 0 else 0.0
        
        # Calculate Reward (portfolio profit)
        reward = 0.0
        accepted_defaults = 0
        for i in range(total_apps):
            if accepted_mask[i]:
                loan_val = week_amounts[i]
                loan_rate = week_rates[i]
                if week_defaults[i] == 1:
                    # Default loss
                    reward -= loan_val * self.default_loss_rate
                    accepted_defaults += 1
                else:
                    # Interest earned
                    reward += loan_val * loan_rate
                    
        # Update state with incoming batch score mean and default rate
        norm_mean_fico = float(np.clip((np.mean(week_scores) - 650.0) / 150.0, 0.0, 1.0)) if total_apps > 0 else 0.5
        if num_accepted > 0:
            batch_default_rate = float(accepted_defaults) / float(num_accepted)
            self.prev_default_rate = 0.6 * self.prev_default_rate + 0.4 * batch_default_rate
        # If num_accepted == 0, preserve self.prev_default_rate so risk information is not lost
        
        self.prev_acceptance_rate = 0.6 * self.prev_acceptance_rate + 0.4 * acceptance_rate
        norm_acc_rate = float(np.clip(self.prev_acceptance_rate, 0.0, 1.0))
        norm_rej_rate = 1.0 - norm_acc_rate
        norm_def_rate = float(np.clip(self.prev_default_rate, 0.0, 1.0))
        
        state = np.array([norm_acc_rate, norm_mean_fico, norm_rej_rate, norm_def_rate], dtype=np.float32)
        
        self.weeks_passed += 1
        terminated = False
        truncated = (self.current_idx >= self.total_records) or (self.max_weeks is not None and self.weeks_passed >= self.max_weeks)
        
        info = {
            "cutoff_used": cutoff,
            "acceptance_rate": acceptance_rate,
            "num_accepted": num_accepted,
            "total_apps": total_apps,
            "num_defaults": accepted_defaults,
            "week_scores": week_scores,
            "week_defaults": week_defaults,
            "week_amounts": week_amounts,
            "week_rates": week_rates
        }
        
        return state, float(reward), terminated, truncated, info


