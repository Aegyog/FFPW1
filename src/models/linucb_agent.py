import numpy as np

class LinUCBAgent:
    def __init__(self, feature_dim, alpha=0.1):
        self.alpha = alpha
        self.feature_dim = feature_dim
        # A and b are the internal "memory" of the bandit
        self.A = np.identity(feature_dim)
        self.b = np.zeros((feature_dim, 1))

    def select_article(self, article_features):
        # Calculate scores for all candidate articles using the UCB formula 
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        
        best_score = -np.inf
        best_idx = -1
        
        for i, x in enumerate(article_features):
            x = x.reshape(-1, 1)
            # Expected reward + uncertainty bound (exploration) 
            p = theta.T @ x + self.alpha * np.sqrt(x.T @ A_inv @ x)
            if p > best_score:
                best_score = p
                best_idx = i
                
        return best_idx

    def update(self, selected_features, reward):
        # Update policy based on click (1) or skip (0)
        x = selected_features.reshape(-1, 1)
        self.A += x @ x.T
        self.b += reward * x
