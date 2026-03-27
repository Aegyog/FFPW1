Model Card: Personalized Recommendation with Contextual Bandits (NLP + RL)

1. Model Details

Developed by: Jaycen John C. Carreon and Justin Errol L. Priniel

Model Type: Hybrid 1D Text CNN and LinUCB Contextual Bandit

Architecture: The NLP component uses pre-trained embeddings combined with a 1D Text CNN to capture local n-gram patterns in news titles. The RL component utilizes a Linear Upper Confidence Bound (LinUCB) agent.

2. Intended Use

Primary Use: Academic research for the 6INTELSY course to demonstrate the integration of NLP and Reinforcement Learning.

Out of Scope: This model is not intended for live production environments or commercial news distribution.

3. Factors and Biases

Demographics: The training data (MIND) reflects the user base of the Microsoft News platform, which may contain inherent demographic biases.

Mitigation: We use an $\epsilon$-greedy / UCB exploration strategy to actively counter "Filter Bubbles" and ideological echo chambers.

Class Imbalance: EDA identified a low Click-Through Rate (CTR) of ~4.8%, justifying the use of Contextual Bandits to handle sparsity and exploration.

4. Training and Evaluation Data

Task: Personalized News Recommendation.

Dataset: Microsoft News Dataset (MIND), specifically the MIND-small version for rapid iteration.

Data Processing: Tab-separated behavior logs were parsed to extract individual user click events.

Splits: An 80/20 train and validation split was implemented based on impression timestamps to maintain chronological integrity.

Anonymization: All user IDs are hashed and personally identifiable information has been removed by the provider.

5. RL Logic & Initial Metrics

RL Logic: The environment is managed via a Replay Simulator for offline evaluation using historical logs.

Reward Design: Binary reward structure where 1 represents a click and 0 represents a non-click.

Initial Benchmarks (nDCG@5):

Random Baseline: 0.182

Popularity Baseline: 0.224

LinUCB (Initial Run): 0.251
