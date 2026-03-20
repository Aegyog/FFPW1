import numpy as np

def run_offline_simulation(agent, behaviors_data, news_features):
    cumulative_reward = 0
    total_steps = 0
    rewards_history = []

    for impression in behaviors_data:
        # Get candidate articles from the log [cite: 22, 53]
        candidates = impression['candidates']
        true_click_idx = impression['clicked_idx']
        
        # Get dense vectors for these candidates from the CNN/NLP pipeline [cite: 51]
        candidate_vectors = [news_features[id] for id in candidates]
        
        # Agent makes a choice
        chosen_idx = agent.select_article(candidate_vectors)
        
        # Binary reward design: Only reward if choice matches historical truth [cite: 17, 54, 66]
        reward = 1 if chosen_idx == true_click_idx else 0
        
        # Only update the agent if we have ground truth for that specific action [cite: 20]
        agent.update(candidate_vectors[chosen_idx], reward)
        
        cumulative_reward += reward
        total_steps += 1
        rewards_history.append(cumulative_reward / total_steps)

    return rewards_history
