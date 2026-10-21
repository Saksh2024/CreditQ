import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from sklearn.kernel_approximation import RBFSampler

class DQNAgent:
    def __init__(self, state_dim=3, action_dim=25, lr=1e-3, gamma=0.95, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1, memory_size=2000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Fit RBFSampler on 3D state vectors in [0.0, 1.0]^3
        self.rbf = RBFSampler(gamma=1.0, n_components=2000, random_state=42)
        rng = np.random.RandomState(42)
        dummy_states = rng.uniform(0.0, 1.0, size=(500, state_dim))
        self.rbf.fit(dummy_states)
        
        # Linear SGD Architecture: 2000 RBF features -> action_dim
        self.q_network = nn.Linear(2000, action_dim)
        self.target_network = nn.Linear(2000, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)
        
    def _transform_state(self, state):
        state_arr = np.array(state, dtype=np.float32)
        if state_arr.ndim == 1:
            state_arr = state_arr.reshape(1, -1)
        return self.rbf.transform(state_arr)

    def select_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        rbf_state = self._transform_state(state)
        state_t = torch.FloatTensor(rbf_state)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(torch.argmax(q_values).item())
        
    def get_q_values(self, state):
        rbf_state = self._transform_state(state)
        state_t = torch.FloatTensor(rbf_state)
        with torch.no_grad():
            q_values = self.q_network(state_t).squeeze(0).numpy()
        return q_values
        
    def select_action_boltzmann(self, state, temperature=10000.0):
        rbf_state = self._transform_state(state)
        state_t = torch.FloatTensor(rbf_state)
        with torch.no_grad():
            q_values = self.q_network(state_t).squeeze(0).numpy()
            
        # Subtract max for numerical stability before exp
        q_values_shifted = q_values - np.max(q_values)
        exp_q = np.exp(q_values_shifted / temperature)
        probs = exp_q / np.sum(exp_q)
        
        action = np.random.choice(self.action_dim, p=probs)
        return action
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def store_counterfactual_batch(self, state, action, reward, next_state, done, info, default_loss_rate=0.45):
        """
        Dense Counterfactual Off-Policy Updating:
        Since the lender observes the full applicant cohort for the period, it computes
        exact counterfactual portfolio rewards across the full action spectrum simultaneously.
        """
        self.store_transition(state, action, reward / 10000.0, next_state, done)
        if "week_scores" in info:
            scores = info["week_scores"]
            defaults = info["week_defaults"]
            amounts = info["week_amounts"]
            rates = info["week_rates"]
            
            # Precompute net individual loan cash flows once (positive interest or negative default loss)
            loan_cash_flows = np.where(defaults == 1, -amounts * default_loss_rate, amounts * rates)

            # Vectorized dense counterfactual update across ALL actions
            for act_cf in range(self.action_dim):
                if act_cf == action:
                    continue
                cut_cf = 9999 if act_cf == (self.action_dim - 1) else (660 + act_cf * 5)
                mask_cf = scores >= cut_cf
                cf_reward = float(np.sum(loan_cash_flows[mask_cf])) if np.any(mask_cf) else 0.0
                self.store_transition(state, act_cf, cf_reward / 10000.0, next_state, done)
        
    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        rbf_states = self._transform_state(states)
        rbf_next_states = self._transform_state(next_states)
        
        states_t = torch.FloatTensor(rbf_states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(rbf_next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        
        curr_q = self.q_network(states_t).gather(1, actions_t)
        
        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q
            
        loss = nn.MSELoss()(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

