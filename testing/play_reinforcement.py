# play_reinforcement.py
import pygame
import sys
import os
import torch
import numpy as np
from reinforcement_env import GeometryDashEnv
from reinforcement_agent import DQNAgent

def play_ai(model_path="models/geometry_dash_dqn.pt", max_steps=10000):
    """Fait jouer l'IA entraînée"""
    
    # Initialiser l'environnement avec rendu
    env = GeometryDashEnv(render_mode="human")
    state_size = env.observation_space
    action_size = env.action_space
    
    # Créer l'agent et charger le modèle entraîné
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size
    )
    agent.load(model_path)
    
    # Désactiver l'exploration pour l'évaluation
    agent.epsilon = 0
    
    # Réinitialiser l'environnement
    state = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    
    # Boucle principale du jeu
    while not done and step_count < max_steps:
        # Gérer les événements pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    env.close()
                    return
        
        # Prédire l'action
        action = agent.act(state, evaluate=True)
        
        # Exécuter l'action
        next_state, reward, done, info = env.step(action)
        
        # Passer à l'état suivant
        state = next_state
        total_reward += reward
        step_count += 1
        
        # Afficher des infos sur l'écran
        font = pygame.font.SysFont(None, 24)
        ai_text = font.render(f"IA en action - Appuyer sur ESC pour quitter", True, (255, 0, 0))
        action_text = font.render(f"Action: {'SAUTER' if action == 1 else 'ATTENDRE'}", True, (0, 0, 255))
        
        env.screen.blit(ai_text, (400, 20))
        env.screen.blit(action_text, (400, 50))
        pygame.display.flip()
    
    print(f"Partie terminée! Score: {info['score']}")
    env.close()

def main():
    """Menu principal pour choisir et exécuter un modèle"""
    
    # Chercher les modèles disponibles
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("Aucun modèle trouvé! Veuillez d'abord entraîner l'IA.")
        return
    
    models = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
    
    if not models:
        print("Aucun modèle trouvé! Veuillez d'abord entraîner l'IA.")
        return
    
    # Afficher les modèles disponibles
    print("Modèles disponibles:")
    for i, model in enumerate(models):
        print(f"{i+1}. {model}")
    
    # Demander à l'utilisateur de choisir un modèle
    choice = -1
    while choice < 1 or choice > len(models):
        try:
            choice = int(input(f"Choisissez un modèle (1-{len(models)}): "))
        except ValueError:
            print("Veuillez entrer un nombre valide.")
    
    selected_model = os.path.join(models_dir, models[choice-1])
    print(f"Modèle sélectionné: {selected_model}")
    
    # Lancer le jeu avec l'IA
    play_ai(model_path=selected_model)

if __name__ == "__main__":
    main()
