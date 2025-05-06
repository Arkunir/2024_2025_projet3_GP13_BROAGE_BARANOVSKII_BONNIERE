# train_reinforcement.py
import os
import numpy as np
import matplotlib.pyplot as plt
from reinforcement_env import GeometryDashEnv
from reinforcement_agent import DQNAgent

def plot_scores(scores, avg_scores, filename="training_progress.png"):
    """Affiche et sauvegarde un graphique des scores pendant l'entraînement"""
    plt.figure(figsize=(12, 8))
    plt.plot(scores, label='Score par épisode', alpha=0.5)
    plt.plot(avg_scores, label='Score moyen (100 épisodes)', linewidth=2)
    plt.xlabel('Épisode')
    plt.ylabel('Score')
    plt.title('Progrès de l\'entraînement de l\'IA')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def train(episodes=1000, render_every=50, max_steps=5000, save_dir="models"):
    """Entraîne l'agent par renforcement sur l'environnement Geometry Dash"""
    
    # Créer le répertoire de sauvegarde s'il n'existe pas
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialiser l'environnement et l'agent
    env = GeometryDashEnv(render_mode=None)  # Sans rendu par défaut
    state_size = env.observation_space
    action_size = env.action_space
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=50000,
        batch_size=64,
        update_target_every=500
    )
    
    # Charger un modèle existant s'il existe
    model_path = os.path.join(save_dir, "geometry_dash_dqn.pt")
    agent.load(model_path)
    
    # Suivi des performances
    scores = []
    avg_scores = []
    best_score = -float('inf')
    
    for episode in range(1, episodes + 1):
        # Afficher le rendu pour certains épisodes
        render = (episode % render_every == 0)
        if render:
            env.close()  # Fermer l'ancien environnement
            env = GeometryDashEnv(render_mode="human")
        else:
            env.close()
            env = GeometryDashEnv(render_mode=None)
        
        # Réinitialiser l'environnement
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        # Boucle d'un épisode
        while not done and step_count < max_steps:
            # Sélectionner une action
            action = agent.act(state)
            
            # Exécuter l'action
            next_state, reward, done, info = env.step(action)
            
            # Stocker la transition
            agent.remember(state, action, reward, next_state, done)
            
            # Apprentissage
            agent.replay()
            
            # Passage à l'état suivant
            state = next_state
            total_reward += reward
            step_count += 1
        
        # Enregistrer le score
        score = info["score"]
        scores.append(score)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        avg_scores.append(avg_score)
        
        # Afficher les progrès
        print(f"Épisode {episode}/{episodes} | Score: {score} | Avg Score: {avg_score:.2f} | Epsilon: {agent.epsilon:.4f} | Steps: {step_count}")
        
        # Sauvegarder le modèle s'il s'améliore
        if avg_score > best_score and episode > 100:
            best_score = avg_score
            agent.save(model_path)
            print(f"Nouveau meilleur score moyen: {best_score:.2f} - Modèle sauvegardé!")
        
        # Sauvegarder périodiquement
        if episode % 100 == 0:
            agent.save(os.path.join(save_dir, f"geometry_dash_dqn_episode_{episode}.pt"))
            # Tracer et sauvegarder le graphique
            plot_scores(scores, avg_scores, filename=os.path.join(save_dir, f"progress_episode_{episode}.png"))
    
    # Sauvegarder le modèle final et le graphique
    agent.save(os.path.join(save_dir, "geometry_dash_dqn_final.pt"))
    plot_scores(scores, avg_scores, filename=os.path.join(save_dir, "final_progress.png"))
    
    env.close()
    print("Entraînement terminé!")

if __name__ == "__main__":
    train(episodes=1000, render_every=50)
