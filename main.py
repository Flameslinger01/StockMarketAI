import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf

data = np.genfromtxt('/workspaces/StockMarketAI/data/testticker.csv', delimiter=',', dtype=None)

print(float(data[1][1]))

# neural network
class PolicyNetwork(tf.keras.Model):
    def __init__(self, n_actions):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.logits = tf.keras.layers.Dense(n_actions)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        logits = self.logits(x)
        return logits
    
# PPO algorithm
class PPOAgent:
    def __init__(self, n_actions, lr_actor=1e-4, clip_ratio=0.2):
        self.policy_network = PolicyNetwork(n_actions)
        self.optimizer = tf.keras.optimizers.Adam(lr_actor)
        self.clip_ratio = clip_ratio

    def get_action(self, state):
        logits = self.policy_network(state)
        action_probs = tf.nn.softmax(logits)
        action = np.random.choice(range(len(action_probs[0])), p=action_probs[0])
        return action

    def train_step(self, states, actions, advantages, old_probs):
        with tf.GradientTape() as tape:
            logits = self.policy_network(states)
            probs = tf.nn.softmax(logits)
            action_masks = tf.one_hot(actions, depth=len(probs[0]))

            ratios = tf.reduce_sum(action_masks * probs, axis=1) / old_probs
            clipped_ratios = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)

            surrogate1 = ratios * advantages
            surrogate2 = clipped_ratios * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        gradients = tape.gradient(actor_loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))

# Main training loop
state_dim = len(data.columns)  # Number of features in the stock data
n_actions = 3  # Example: 3 actions (buy, sell, hold)

ppo_agent = PPOAgent(n_actions)

epochs = 100
for epoch in range(epochs):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    total_reward = 0

    for i in range(len(data) - 1):
        state = data.iloc[i].values
        next_state = data.iloc[i + 1].values

        # Example: Simple action selection based on price change
        action = 0  # Hold
        if next_state[3] > state[3]:  # Close price increased
            action = 1  # Buy
        elif next_state[3] < state[3]:  # Close price decreased
            action = 2  # Sell

        states.append(state)
        actions.append(action)
        rewards.append(next_state[3] - state[3])  # Reward based on close price change
        next_states.append(next_state)

    # Compute advantages using a simple reward-to-go scheme
    advantages = []
    adv = 0
    for r in reversed(rewards):
        adv = adv * 0.99 + r
        advantages.append(adv)
    advantages.reverse()
    advantages = np.array(advantages)
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    # Perform PPO training step
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
    old_probs = tf.nn.softmax(ppo_agent.policy_network(states)).numpy()

    ppo_agent.train_step(states, actions, advantages, old_probs)

    total_reward = np.sum(rewards)
    print(f"Epoch: {epoch + 1}, Total Reward: {total_reward}")


