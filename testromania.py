import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import pygame
import sys
import os
from collections import deque
import matplotlib.pyplot as plt

# Suppression des avertissements PyTorch
import warnings
warnings.filterwarnings("ignore")

# Constantes globales
GROUND_HEIGHT = 500
WIDTH, HEIGHT = 800, 600
CUBE_SIZE = 50

# Configuration de l'IA
STATE_SIZE = 10  # Nombre de caractéristiques d'état
ACTION_SIZE = 2  # [Ne pas sauter, Sauter]
BATCH_SIZE = 64  # Taille du lot d'expériences pour l'apprentissage
GAMMA = 0.99     # Facteur de réduction pour les récompenses futures
LEARNING_RATE = 0.0005  # Taux d'apprentissage
MEMORY_SIZE = 100000  # Taille maximale de la mémoire d'expérience
EXPLORATION_MAX = 1.0  # Exploration initiale
EXPLORATION_MIN = 0.01  # Exploration minimale
EXPLORATION_DECAY = 0.9995  # Taux de décroissance de l'exploration

# Configuration pour l'entraînement
EPISODES = 1000
PRINT_EVERY = 10
SAVE_MODEL_EVERY = 100

# Réseau de neurones pour Deep Q-Learning
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Agent d'apprentissage par renforcement
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EXPLORATION_MAX
        self.epsilon_min = EXPLORATION_MIN
        self.epsilon_decay = EXPLORATION_DECAY
        self.learning_rate = LEARNING_RATE
        
        # Vérifier si CUDA (GPU) est disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation de: {self.device}")
        
        # Réseaux principal et cible
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()
        
        # Optimiseur
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Pour suivre les statistiques d'entraînement
        self.scores = []
        self.epsilon_history = []
        self.avg_scores = []
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # Exploration vs exploitation
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Convertir l'état en tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Passer à travers le réseau
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state)
        self.model.train()
        
        return torch.argmax(q_values).item()
    
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Échantillonner une expérience aléatoire de la mémoire
        minibatch = random.sample(self.memory, BATCH_SIZE)
        
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            
            # Calculer la cible Q
            target = self.model(state.unsqueeze(0)).detach().squeeze()
            
            if done:
                target[action] = reward
            else:
                next_q_values = self.target_model(next_state.unsqueeze(0)).detach().squeeze()
                target[action] = reward + self.gamma * torch.max(next_q_values).item()
            
            states.append(state)
            targets.append(target)
        
        # Convertir en tenseurs
        states = torch.stack(states)
        targets = torch.stack(targets)
        
        # Mettre à jour le modèle
        self.optimizer.zero_grad()
        predictions = self.model(states)
        loss = F.mse_loss(predictions, targets)
        loss.backward()
        
        # Ajout d'un gradient clipping pour stabiliser l'apprentissage
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Réduire l'exploration au fil du temps
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'scores': self.scores,
            'epsilon_history': self.epsilon_history,
            'avg_scores': self.avg_scores
        }, filename)
        print(f"Modèle sauvegardé: {filename}")
    
    def load_model(self, filename):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.scores = checkpoint.get('scores', [])
            self.epsilon_history = checkpoint.get('epsilon_history', [])
            self.avg_scores = checkpoint.get('avg_scores', [])
            print(f"Modèle chargé: {filename}")
        else:
            print(f"Fichier non trouvé: {filename}")

    def plot_training(self):
        """Affiche les graphiques d'entraînement"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.scores)
        plt.title('Score par épisode')
        plt.xlabel('Épisode')
        plt.ylabel('Score')
        
        plt.subplot(1, 3, 2)
        plt.plot(self.epsilon_history)
        plt.title('Epsilon par épisode')
        plt.xlabel('Épisode')
        plt.ylabel('Epsilon')
        
        plt.subplot(1, 3, 3)
        plt.plot(self.avg_scores)
        plt.title('Score moyen sur 100 épisodes')
        plt.xlabel('Épisode')
        plt.ylabel('Score moyen')
        
        plt.tight_layout()
        plt.savefig('training_stats.png')
        plt.close()

# Fonction pour prétraiter l'état du jeu
def preprocess_state(player, game_objects):
    """
    Extrait les caractéristiques pertinentes du joueur et des obstacles proches
    
    Args:
        player: Objet joueur
        game_objects: Liste des objets de jeu
    
    Returns:
        Un vecteur d'état normalisé pour l'apprentissage
    """
    # Initialiser l'état avec des valeurs par défaut
    state = np.zeros(STATE_SIZE)
    
    # Position et vitesse du joueur (normalisées)
    state[0] = player.rect.x / WIDTH  # Position horizontale
    state[1] = player.rect.y / HEIGHT  # Position verticale
    state[2] = player.velocity / 20.0  # Vitesse verticale normalisée
    
    # Recherche des obstacles les plus proches devant le joueur
    obstacles_ahead = []
    for obj in game_objects:
        # Obtenir la position de l'obstacle
        obj_x = 0
        obj_y = 0
        obj_width = 0
        obj_height = 0
        
        # Déterminer le type d'obstacle et obtenir ses dimensions
        if hasattr(obj, 'x') and hasattr(obj, 'width'):
            obj_x = obj.x
            obj_width = obj.width
            # Pour la hauteur, différentes approches selon le type
            if hasattr(obj, 'height'):
                obj_y = GROUND_HEIGHT - obj.height
                obj_height = obj.height
            else:
                # Pour les objets sans attribut height explicite
                obj_y = GROUND_HEIGHT - CUBE_SIZE
                obj_height = CUBE_SIZE
        elif hasattr(obj, 'rect'):
            obj_x = obj.rect.x
            obj_y = obj.rect.y
            obj_width = obj.rect.width
            obj_height = obj.rect.height
        
        # Ne considérer que les obstacles devant le joueur
        if obj_x > player.rect.right:
            obstacles_ahead.append((obj_x, obj_y, obj_width, obj_height))
    
    # Trier les obstacles par distance
    obstacles_ahead.sort(key=lambda x: x[0])
    
    # Ajouter les informations des 3 obstacles les plus proches à l'état
    for i in range(min(3, len(obstacles_ahead))):
        obj_x, obj_y, obj_width, obj_height = obstacles_ahead[i]
        
        # Pour chaque obstacle, ajouter:
        state_index = 3 + i * 2
        
        # Distance horizontale normalisée
        state[state_index] = (obj_x - player.rect.right) / WIDTH
        
        # Hauteur de l'obstacle normalisée
        state[state_index + 1] = obj_height / HEIGHT
    
    # Retourner l'état
    return state

# Fonction pour récompenser l'agent
def calculate_reward(player, game_objects, score, prev_score, done):
    """
    Calcule la récompense basée sur l'état actuel du jeu
    
    Args:
        player: Objet joueur
        game_objects: Liste des objets du jeu
        score: Score actuel
        prev_score: Score précédent
        done: Indicateur si le jeu est terminé
    
    Returns:
        La récompense calculée
    """
    reward = 0
    
    # Pénalité pour la mort
    if done:
        reward = -100
    else:
        # Récompenser la progression
        if score > prev_score:
            reward += 10 * (score - prev_score)  # Forte récompense pour franchir un obstacle
        
        # Petit bonus pour survivre
        reward += 0.1
        
        # Bonus pour franchir des obstacles proches
        for obj in game_objects:
            obj_x = 0
            obj_width = 0
            
            if hasattr(obj, 'x') and hasattr(obj, 'width'):
                obj_x = obj.x
                obj_width = obj.width
            elif hasattr(obj, 'rect'):
                obj_x = obj.rect.x
                obj_width = obj.rect.width
            
            # Si le joueur est très proche d'un obstacle mais survit
            if obj_x < player.rect.right + 50 and obj_x + obj_width > player.rect.left - 50:
                reward += 0.5  # Petit bonus pour la proximité
    
    return reward

# Fonction d'entraînement
def train_ai():
    from klass import Player
    import pygame
    import sys
    from main import screen, clock, FPS, WIDTH, HEIGHT, GROUND_HEIGHT, BLACK, WHITE
    
    # Initialiser l'agent
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    model_path = "dqn_model.pth"
    
    # Vérifier si un modèle existe déjà et le charger
    if os.path.exists(model_path):
        agent.load_model(model_path)
    
    # Statistiques d'entraînement
    episode_scores = []
    episode_count = 0
    
    print("Démarrage de l'entraînement...")
    
    # Boucle d'entraînement
    for episode in range(EPISODES):
        player = Player()
        game_objects = []
        score = 0
        prev_score = 0
        last_object = pygame.time.get_ticks()
        current_speed = 6  # Vitesse initiale
        
        # Distances minimales entre obstacles
        min_obstacle_distances = {
            6: 150, 7: 175, 8: 500, 9: 250, 10: 275, 11: 300
        }
        
        obstacle_intervals = {
            6: [800, 1400], 7: [900, 1600], 8: [1200, 1800],
            9: [1300, 2000], 10: [1400, 2100], 11: [1500, 2200]
        }
        
        object_interval = random.randint(*obstacle_intervals[current_speed])
        
        # Configurer les seuils de vitesse
        speed_threshold_7 = random.randint(10, 20)
        min_threshold_8 = max(25, 2 * speed_threshold_7 - 5)
        max_threshold_8 = 2 * speed_threshold_7 + 10
        speed_threshold_8 = random.randint(min_threshold_8, max_threshold_8)
        min_threshold_9 = max(40, 2 * speed_threshold_8 - 15)
        max_threshold_9 = 2 * speed_threshold_8 + 5
        speed_threshold_9 = random.randint(min_threshold_9, max_threshold_9)
        speed_threshold_random = 100
        next_random_change = speed_threshold_random + random.randint(25, 50)
        
        # Obtenir l'état initial
        state = preprocess_state(player, game_objects)
        
        done = False
        
        # Accumulateur pour les actions sautées (optimisation de vitesse)
        skip_action_counter = 0
        
        # Boucle principale du jeu
        while not done:
            current_time = pygame.time.get_ticks()
            
            # Gestion des événements pour permettre de quitter le jeu
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # Créer un nouvel obstacle si nécessaire
            can_spawn_obstacle = True
            min_distance = min_obstacle_distances[current_speed]
            
            if game_objects:
                last_obstacle = game_objects[-1]
                last_obstacle_right = 0
                
                # Déterminer la position de l'obstacle
                if hasattr(last_obstacle, 'x') and hasattr(last_obstacle, 'width'):
                    last_obstacle_right = last_obstacle.x + last_obstacle.width
                elif hasattr(last_obstacle, 'rect'):
                    last_obstacle_right = last_obstacle.rect.right
                
                if WIDTH - last_obstacle_right < min_distance:
                    can_spawn_obstacle = False
            
            # Ajouter un obstacle si nécessaire
            if can_spawn_obstacle and current_time - last_object > object_interval:
                # Ici, vous devez importer et créer les obstacles
                # Nous allons simplifier pour cet exemple
                from klass import Obstacle
                obj = Obstacle(WIDTH)
                obj.set_speed(current_speed)
                game_objects.append(obj)
                
                last_object = current_time
                object_interval = random.randint(*obstacle_intervals[current_speed])
            
            # N'agir que tous les 4 frames pour accélérer l'entraînement
            if skip_action_counter == 0:
                # Choisir une action (0: ne rien faire, 1: sauter)
                action = agent.act(state)
                
                # Exécuter l'action
                if action == 1 and not player.is_jumping:
                    player.jump()
            
            skip_action_counter = (skip_action_counter + 1) % 4
            
            # Mettre à jour le joueur et les objets
            player.update(game_objects)
            
            objects_to_remove = []
            
            # Mettre à jour les objets du jeu
            for obj in game_objects[:]:
                obj.update()
                
                # Vérifier les collisions et marquer les objets à supprimer
                if hasattr(obj, 'get_rect') and player.rect.colliderect(obj.get_rect()):
                    player.is_alive = False
                    done = True
                    break
                
                # Vérifier si l'objet est hors de l'écran
                obj_right = 0
                if hasattr(obj, 'x') and hasattr(obj, 'width'):
                    obj_right = obj.x + obj.width
                elif hasattr(obj, 'rect'):
                    obj_right = obj.rect.right
                
                if obj_right < 0:
                    objects_to_remove.append(obj)
            
            # Supprimer les objets et mettre à jour le score
            for obj in objects_to_remove:
                if obj in game_objects:
                    game_objects.remove(obj)
                    score += 1
                    
                    # Mettre à jour la vitesse en fonction du score
                    if score == speed_threshold_7 and current_speed < 7:
                        current_speed = 7
                        for game_obj in game_objects:
                            if hasattr(game_obj, 'set_speed'):
                                game_obj.set_speed(current_speed)
                    elif score == speed_threshold_8 and current_speed < 8:
                        current_speed = 8
                        for game_obj in game_objects:
                            if hasattr(game_obj, 'set_speed'):
                                game_obj.set_speed(current_speed)
                    elif score == speed_threshold_9 and current_speed < 9:
                        current_speed = 9
                        for game_obj in game_objects:
                            if hasattr(game_obj, 'set_speed'):
                                game_obj.set_speed(current_speed)
                    elif score == speed_threshold_random:
                        new_speed = random.randint(9, 11)
                        current_speed = new_speed
                        for game_obj in game_objects:
                            if hasattr(game_obj, 'set_speed'):
                                game_obj.set_speed(current_speed)
                        next_random_change = score + random.randint(25, 50)
            
            # Vérifier si le joueur est mort
            if not player.is_alive:
                done = True
            
            # Obtenir le nouvel état et calculer la récompense
            next_state = preprocess_state(player, game_objects)
            reward = calculate_reward(player, game_objects, score, prev_score, done)
            prev_score = score
            
            # Stocker l'expérience
            agent.remember(state, action, reward, next_state, done)
            
            # Passer à l'état suivant
            state = next_state
            
            # Apprendre à partir des expériences passées
            agent.replay()
            
            # Afficher l'état du jeu (optionnel pendant l'entraînement)
            if episode % 10 == 0:  # N'afficher que tous les 10 épisodes pour accélérer
                screen.fill(WHITE)
                pygame.draw.rect(screen, BLACK, (0, GROUND_HEIGHT, WIDTH, HEIGHT - GROUND_HEIGHT))
                player.draw(screen)
                
                for obj in game_objects:
                    obj.draw(screen)
                
                font = pygame.font.SysFont(None, 36)
                score_text = font.render(f"Score: {score}", True, BLACK)
                screen.blit(score_text, (20, 20))
                
                episode_text = font.render(f"Episode: {episode + 1}/{EPISODES}", True, BLACK)
                screen.blit(episode_text, (20, 60))
                
                epsilon_text = font.render(f"Epsilon: {agent.epsilon:.4f}", True, BLACK)
                screen.blit(epsilon_text, (20, 100))
                
                pygame.display.flip()
                
                # Contrôler la vitesse du jeu pendant l'entraînement
                clock.tick(FPS * 2)  # Double vitesse pour entraînement
        
        # Fin de l'épisode
        episode_count += 1
        agent.scores.append(score)
        agent.epsilon_history.append(agent.epsilon)
        
        # Calculer le score moyen sur les 100 derniers épisodes
        avg_score = np.mean(agent.scores[-100:])
        agent.avg_scores.append(avg_score)
        
        # Mise à jour régulière du modèle cible (toutes les 10 épisodes)
        if episode % 10 == 0:
            agent.update_target_model()
        
        # Afficher les progrès
        if episode % PRINT_EVERY == 0:
            print(f"Episode: {episode}/{EPISODES}, Score: {score}, Epsilon: {agent.epsilon:.4f}, Avg Score: {avg_score:.2f}")
        
        # Sauvegarder le modèle régulièrement
        if episode % SAVE_MODEL_EVERY == 0:
            agent.save_model(model_path)
            # Créer un graphique de l'entraînement
            agent.plot_training()
    
    # Sauvegarder le modèle final
    agent.save_model(model_path)
    agent.plot_training()
    print("Entraînement terminé!")
    return agent

# Fonction pour jouer avec l'IA entraînée
def ai_reinforcement_play():
    from klass import Player
    import pygame
    import sys
    from main import screen, clock, FPS, WIDTH, HEIGHT, GROUND_HEIGHT, BLACK, WHITE
    
    # Créer un agent IA et charger le modèle entraîné
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    model_path = "dqn_model.pth"
    
    if os.path.exists(model_path):
        agent.load_model(model_path)
    else:
        print("Aucun modèle trouvé! Veuillez d'abord entraîner l'IA.")
        return
    
    # Désactiver l'exploration pour le mode démonstration
    agent.epsilon = 0.0
    
    # Initialiser le jeu
    player = Player()
    game_objects = []
    score = 0
    last_object = pygame.time.get_ticks()
    current_speed = 6
    
    # Mêmes configurations que dans l'entraînement
    min_obstacle_distances = {
        6: 150, 7: 175, 8: 500, 9: 250, 10: 275, 11: 300
    }
    
    obstacle_intervals = {
        6: [800, 1400], 7: [900, 1600], 8: [1200, 1800],
        9: [1300, 2000], 10: [1400, 2100], 11: [1500, 2200]
    }
    
    object_interval = random.randint(*obstacle_intervals[current_speed])
    
    # Configurer les seuils de vitesse
    speed_threshold_7 = random.randint(10, 20)
    min_threshold_8 = max(25, 2 * speed_threshold_7 - 5)
    max_threshold_8 = 2 * speed_threshold_7 + 10
    speed_threshold_8 = random.randint(min_threshold_8, max_threshold_8)
    min_threshold_9 = max(40, 2 * speed_threshold_8 - 15)
    max_threshold_9 = 2 * speed_threshold_8 + 5
    speed_threshold_9 = random.randint(min_threshold_9, max_threshold_9)
    speed_threshold_random = 100
    next_random_change = speed_threshold_random + random.randint(25, 50)
    
    print("L'IA commence à jouer...")
    
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Créer un nouvel obstacle si nécessaire
        can_spawn_obstacle = True
        min_distance = min_obstacle_distances[current_speed]
        
        if game_objects:
            last_obstacle = game_objects[-1]
            last_obstacle_right = 0
            
            # Déterminer la position de l'obstacle
            if hasattr(last_obstacle, 'x') and hasattr(last_obstacle, 'width'):
                last_obstacle_right = last_obstacle.x + last_obstacle.width
            elif hasattr(last_obstacle, 'rect'):
                last_obstacle_right = last_obstacle.rect.right
            
            if WIDTH - last_obstacle_right < min_distance:
                can_spawn_obstacle = False
        
        # Ajouter un obstacle si nécessaire
        if can_spawn_obstacle and current_time - last_object > object_interval:
            # Importer et créer tous les types d'obstacles
            from klass import (Obstacle, DoublePikes, TriplePikes, QuadruplePikes, Block,
                              BlockGapBlockWithSpike, BouncingObstacle, DoubleBlockPillar,
                              FivePikesWithOrb, JumpPad, QuintuplePikesWithJumpPad, PurpleOrb,
                              JumppadOrbsObstacle)
            
            # Choisir un obstacle en fonction de la vitesse
            obj = None
            
            if current_speed == 6:
                choice = random.random()
                if choice < 0.35:
                    obj = Obstacle(WIDTH)
                elif choice < 0.5:
                    obj = JumpPad(WIDTH)
                elif choice < 0.65:
                    obj = Block(WIDTH)
                else:
                    obj = PurpleOrb(WIDTH)
            elif current_speed == 7:
                choice = random.random()
                if choice < 0.15:
                    obj = Obstacle(WIDTH)
                elif choice < 0.3:
                    obj = Block(WIDTH)
                elif choice < 0.5:
                    obj = DoublePikes(WIDTH)
                elif choice < 0.7:
                    obj = QuintuplePikesWithJumpPad(WIDTH)
                elif choice < 0.85:
                    obj = FivePikesWithOrb(WIDTH)
                else:
                    obj = PurpleOrb(WIDTH)
            elif current_speed == 8:
                choice = random.random()
                if choice < 0.1:
                    obj = Obstacle(WIDTH)
                elif choice < 0.2:
                    obj = DoublePikes(WIDTH)
                elif choice < 0.35:
                    obj = DoubleBlockPillar(WIDTH)
                elif choice < 0.5:
                    obj = BlockGapBlockWithSpike(WIDTH)
                elif choice < 0.6:
                    obj = BouncingObstacle(WIDTH)
                elif choice < 0.7:
                    obj = FivePikesWithOrb(WIDTH)
                elif choice < 0.85:
                    obj = PurpleOrb(WIDTH)
                else:
                    obj = JumppadOrbsObstacle(WIDTH)
            elif current_speed >= 9:
                choice = random.random()
                if choice < 0.3:
                    obj = DoublePikes(WIDTH)
                elif choice < 0.45:
                    obj = BlockGapBlockWithSpike(WIDTH)
                elif choice < 0.55:
                    obj = TriplePikes(WIDTH)
                elif choice < 0.65:
                    obj = QuadruplePikes(WIDTH)
                elif choice < 0.75:
                    obj = FivePikesWithOrb(WIDTH)
                elif choice < 0.85:
                    obj = PurpleOrb(WIDTH)
                else:
                    obj = JumppadOrbsObstacle(WIDTH)
            
            # Configurer et ajouter l'obstacle
            if obj:
                obj.set_speed(current_speed)
                game_objects.append(obj)
                
                last_object = current_time
                object_interval = random.randint(*obstacle_intervals[current_speed])
        
        # Obtenir l'état actuel
        state = preprocess_state(player, game_objects)
        
        # L'IA prend une décision
        action = agent.act(state)
        
        # Exécuter l'action
        if action == 1 and not player.is_jumping:
            player.jump()
        
        # Mettre à jour le joueur et les objets
        player.update(game_objects)
        
        objects_to_remove = []
        
        # Mettre à jour et vérifier les collisions
        for obj in game_objects[:]:
            obj.update()