# Reinforcement-Learning
Contains implementations of foundation RL algorithms such as UCB, KL-UCB, Thompson Sampling etc.

The file UCB, KL-UCB contains implementations for the algorithms Upper Confidence Bounds and KL Divergence Upper Confidence Bounds. A brief description for the same is given below - 
# 1. UCB (Upper Confidence Bound) -> 
UCB is a strategy used in decision-making scenarios where you need to balance exploration and exploitation.
It works by giving each option a "score" that combines how well the option has performed in the past (exploitation) with a "bonus" for uncertainty (exploration). The idea is that if an option hasn't been tried much, the bonus is higher, encouraging you to explore it. Over time, as you gather more information, the bonus decreases, and you start focusing more on the options that have performed the best.
This approach helps you make smart choices that balance trying new things with sticking to what works, which is crucial when you don't have much information upfront.


# 2. KL-UCB (Kullback-Leibler Upper Confidence Bound) ->
KL-UCB is a more advanced version of UCB. It uses a concept from information theory called the Kullback-Leibler (KL) divergence to create a more precise exploration bonus.
Instead of just adding a fixed bonus like in UCB, KL-UCB calculates the bonus based on how different the expected performance of an option could be from what you already know. It considers both the average performance and how confident you are about that performance. The KL divergence helps to measure the difference between what you expect and what might happen.
KL-UCB is usually more accurate than regular UCB because it adapts the exploration bonus more carefully based on the specific data you have. This can lead to better decisions, especially in complex situations where the outcomes of different options vary widely.

# 3. Thompson Sampling ->
Thompson Sampling is a decision-making strategy used in problems like the multi-armed bandit, where you need to balance exploration (trying new options) and exploitation (choosing the best-known option). It works by maintaining a belief about the success rate of each option, represented by probability distributions. In each round, Thompson Sampling randomly samples a success rate from these distributions and picks the option with the highest sampled rate. Afterward, it updates its belief based on the observed outcome, gradually refining its choices. This method effectively balances exploration and exploitation, helping to identify the best option over time.

# 4. SARSA ->

SARSA (State-Action-Reward-State-Action) is a reinforcement learning algorithm used for learning a policy in a Markov decision process. It works by updating the value of the current state-action pair based on the expected reward and the estimated value of the next state-action pair. The name SARSA comes from the sequence of events it uses to update the Q-value: State, Action, Reward, (next) State, (next) Action





