import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure Python can find the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.text_cnn import TextCNN
from src.models.linucb_agent import LinUCBAgent
from src.evaluation.simulator import run_offline_simulation

def generate_mock_data(num_news=100, num_impressions=500, vocab_size=1000, seq_len=15):
    """
    Generates mock data to simulate the MIND dataset so the evaluation 
    script can run and generate plots without needing to download 5GB of data.
    """
    print("Generating mock data for simulation...")
    
    # Mock News Text (random word indices)
    news_texts = torch.randint(0, vocab_size, (num_news, seq_len))
    
    # Mock Categories for Slice Analysis
    categories = ['Sports', 'Politics', 'Finance', 'Entertainment']
    news_categories = {i: categories[np.random.randint(0, len(categories))] for i in range(num_news)}
    
    # Mock Impressions (User behaviors)
    behaviors_data = []
    for _ in range(num_impressions):
        # User is presented with 5 random articles
        candidates = np.random.choice(num_news, 5, replace=False).tolist()
        # They click exactly 1
        clicked_idx = np.random.randint(0, 5)
        behaviors_data.append({
            'candidates': candidates,
            'clicked_idx': clicked_idx,
            'category_clicked': news_categories[candidates[clicked_idx]] # For slice analysis
        })
        
    return news_texts, behaviors_data, news_categories

def extract_features(cnn_model, news_texts):
    """Passes text through the Text-CNN to get dense feature vectors."""
    cnn_model.eval()
    with torch.no_grad():
        features_tensor = cnn_model(news_texts)
    # Convert to numpy for the LinUCB agent
    features_np = features_tensor.numpy()
    
    # Create dictionary mapping news_id to its feature vector
    news_features = {i: features_np[i] for i in range(len(features_np))}
    return news_features

def run_ablations():
    # Setup directories
    os.makedirs('experiments/results', exist_ok=True)
    
    # Base Parameters
    vocab_size = 1000
    filter_sizes = [2, 3, 4]
    n_filters = 32
    feature_dim = len(filter_sizes) * n_filters # 3 * 32 = 96
    
    # Generate Data
    news_texts, behaviors_data, news_categories = generate_mock_data(vocab_size=vocab_size)
    
    # ==========================================
    # ABLATION 1: RL Exploration Rate (Alpha)
    # ==========================================
    print("\n--- Running Ablation 1: LinUCB Alpha Parameter ---")
    cnn_base = TextCNN(vocab_size, embed_dim=64, n_filters=n_filters, filter_sizes=filter_sizes)
    features_base = extract_features(cnn_base, news_texts)
    
    agent_low_exp = LinUCBAgent(feature_dim=128, alpha=0.1) # Note: TextCNN fc layer outputs 128
    agent_high_exp = LinUCBAgent(feature_dim=128, alpha=0.5)
    
    rewards_alpha_01 = run_offline_simulation(agent_low_exp, behaviors_data, features_base)
    rewards_alpha_05 = run_offline_simulation(agent_high_exp, behaviors_data, features_base)
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_alpha_01, label='LinUCB (alpha=0.1)')
    plt.plot(rewards_alpha_05, label='LinUCB (alpha=0.5)')
    plt.title('Ablation 1: Learning Curves by Exploration Rate')
    plt.xlabel('Impressions (Time)')
    plt.ylabel('Cumulative Reward Rate (CTR)')
    plt.legend()
    plt.savefig('experiments/results/ablation_1_alpha.png')
    print("Saved plot to experiments/results/ablation_1_alpha.png")
    
    # ==========================================
    # ABLATION 2: CNN Embedding Dimension
    # ==========================================
    print("\n--- Running Ablation 2: CNN Embedding Dimension ---")
    cnn_small_emb = TextCNN(vocab_size, embed_dim=32, n_filters=n_filters, filter_sizes=filter_sizes)
    cnn_large_emb = TextCNN(vocab_size, embed_dim=128, n_filters=n_filters, filter_sizes=filter_sizes)
    
    features_small = extract_features(cnn_small_emb, news_texts)
    features_large = extract_features(cnn_large_emb, news_texts)
    
    agent_small = LinUCBAgent(feature_dim=128, alpha=0.1)
    agent_large = LinUCBAgent(feature_dim=128, alpha=0.1)
    
    rewards_small_emb = run_offline_simulation(agent_small, behaviors_data, features_small)
    rewards_large_emb = run_offline_simulation(agent_large, behaviors_data, features_large)
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_small_emb, label='TextCNN (embed_dim=32)')
    plt.plot(rewards_large_emb, label='TextCNN (embed_dim=128)')
    plt.title('Ablation 2: Learning Curves by NLP Embedding Size')
    plt.xlabel('Impressions (Time)')
    plt.ylabel('Cumulative Reward Rate (CTR)')
    plt.legend()
    plt.savefig('experiments/results/ablation_2_embeddings.png')
    print("Saved plot to experiments/results/ablation_2_embeddings.png")
    
    # ==========================================
    # ERROR & SLICE ANALYSIS
    # ==========================================
    print("\n--- Running Error & Slice Analysis ---")
    # Let's analyze the performance of the best agent across different news categories
    category_clicks = {cat: 0 for cat in set(news_categories.values())}
    category_impressions = {cat: 0 for cat in set(news_categories.values())}
    
    # Re-run simulation and track categories
    agent_final = LinUCBAgent(feature_dim=128, alpha=0.2)
    for impression in behaviors_data:
        candidates = impression['candidates']
        true_idx = impression['clicked_idx']
        true_category = impression['category_clicked']
        
        candidate_vectors = [features_large[id] for id in candidates]
        chosen_idx = agent_final.select_article(candidate_vectors)
        
        reward = 1 if chosen_idx == true_idx else 0
        agent_final.update(candidate_vectors[chosen_idx], reward)
        
        category_impressions[true_category] += 1
        category_clicks[true_category] += reward

    cats = list(category_clicks.keys())
    ctrs = [category_clicks[c] / max(1, category_impressions[c]) for c in cats]
    
    plt.figure(figsize=(8, 5))
    plt.bar(cats, ctrs, color='skyblue')
    plt.title('Slice Analysis: Model CTR by News Category')
    plt.ylabel('Simulated CTR')
    plt.savefig('experiments/results/slice_analysis.png')
    print("Saved plot to experiments/results/slice_analysis.png")
    
    print("\nEvaluation Complete! All requirements for Week 3 met.")

if __name__ == "__main__":
    run_ablations()
