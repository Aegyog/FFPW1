Personalized Recommendation with Contextual Bandits (NLP + RL)

Course: 6INTELSY Final Project
Authors: Justin Errol L. Priniel & Jaycen John C. Carreon

Overview
This project is an intelligent news recommendation system. It ranks items by capturing semantic features from text and adapting to shifting user preferences. It combines NLP embeddings, a 1D Text CNN for feature extraction, and a Contextual Bandit reinforcement learning agent.

v0.9 Progress (Week 2 Checkpoint)
* Data acquired and cleaned with splits finalized.
* EDA notebook completed with click-through rate analysis.
* CNN experiment is running and NLP component is prototyped.
* RL agent is stubbed with reward design and early learning curves.

Dataset and Ethics
We use the Microsoft News Dataset (MIND). The data consists of anonymized behavior logs with all personal information removed. We use exploration strategies to mitigate filter bubbles.

Quick Start
1. Clone the repo: git clone https://github.com/Aegyog/6intelsy-final-project.git
2. Install dependencies: pip install -r requirements.txt
3. Open EDA: notebooks/EDA_MIND_Dataset.ipynb
4. Run Simulator: python src/evaluation/simulator.py
