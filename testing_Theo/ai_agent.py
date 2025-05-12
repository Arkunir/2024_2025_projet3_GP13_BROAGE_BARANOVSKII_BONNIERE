import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

# Définition du réseau de neurones pour le DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Agent d'apprentissage par renforcement
class RLAgent:
    def __init__(self, state_size, action_size, device="cpu"):
        self.state_size = state_size  # Taille de l'état d'entrée
        self.action_size = action_size  # Nombre d'actions possibles (sauter ou ne pas sauter)
        self.memory = deque(maxlen=10000)  # Mémoire pour stocker les expériences
        self.gamma = 0.95  # Facteur d'actualisation
        self.epsilon = 1.0  # Taux d'exploration initial
        self.epsilon_min = 0.01  # Taux d'exploration minimal
        self.epsilon_decay = 0.995  # Décroissance du taux d'exploration
        self.learning_rate = 0.001  # Taux d'apprentissage
        self.device = device
        
        # Initialisation des réseaux
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Compteur d'étapes pour la mise à jour du réseau cible
        self.update_target_counter = 0
        
        # Charger un modèle si disponible
        self.load_model()
    
    def load_model(self, path="model/rl_model.pth"):
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                print(f"Modèle chargé avec succès depuis {path}")
                # Synchroniser le réseau cible
                self.target_model.load_state_dict(self.model.state_dict())
            except Exception as e:
                print(f"Erreur lors du chargement du modèle: {e}")
    
    def save_model(self, path="model/rl_model.pth"):
        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Modèle sauvegardé à {path}")
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # Exploration aléatoire avec probabilité epsilon
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation: choisir la meilleure action selon le modèle
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        # Échantillonner un mini-batch de la mémoire
        minibatch = random.sample(self.memory, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Calcul des valeurs Q actuelles et des cibles
        current_q_values = self.model(states).gather(1, actions)
        
        with torch.no_grad():
            # Double DQN: sélectionner actions avec le modèle principal
            argmax_actions = self.model(next_states).detach().max(1)[1].unsqueeze(1)
            # Et évaluer avec le modèle cible
            next_q_values = self.target_model(next_states).gather(1, argmax_actions)
            
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Calcul de la perte et mise à jour des poids
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping pour stabiliser l'apprentissage
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Mettre à jour le taux d'exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Mettre à jour le réseau cible périodiquement
        self.update_target_counter += 1
        if self.update_target_counter % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def get_state(self, player, game_objects, screen_width):
        """
        Extraire l'état du jeu pour l'agent RL
        """
        # Si aucun obstacle, retourner un état par défaut
        if not game_objects:
            return np.zeros(self.state_size)
        
        # Trouver les obstacles les plus proches devant le joueur
        obstacles_ahead = [obj for obj in game_objects if obj.x > player.rect.right]
        if not obstacles_ahead:
            # Si aucun obstacle devant, utiliser les données du dernier obstacle
            return np.zeros(self.state_size)
        
        # Prendre les 3 obstacles les plus proches
        sorted_obstacles = sorted(obstacles_ahead, key=lambda obj: obj.x)
        closest_obstacles = sorted_obstacles[:3]
        
        # Remplir avec des obstacles "vides" si nécessaire
        while len(closest_obstacles) < 3:
            # Obstacle "vide" à la distance maximale
            class EmptyObstacle:
                def __init__(self):
                    self.x = screen_width * 2
                    self.width = 0
                    self.height = 0
            closest_obstacles.append(EmptyObstacle())
        
        # Construire l'état
        state = []
        for i, obstacle in enumerate(closest_obstacles):
            # Distance normalisée entre le joueur et l'obstacle
            dist_x = (obstacle.x - player.rect.right) / screen_width
            
            # Type d'obstacle (encodé selon la classe)
            obstacle_type = self.encode_obstacle_type(obstacle)
            
            # Position verticale du joueur normalisée
            player_y = player.rect.y / 500  # GROUND_HEIGHT
            
            # Vitesse verticale du joueur normalisée
            player_vel_y = player.vel_y / 20  # Normaliser par une valeur max approximative
            
            # Ajouter ces caractéristiques à l'état
            state.extend([dist_x, obstacle_type, player_y, player_vel_y])
        
        return np.array(state, dtype=np.float32)
    
    def encode_obstacle_type(self, obstacle):
        """
        Encoder le type d'obstacle en valeur numérique
        """
        obstacle_class = obstacle.__class__.__name__
        
        # Dictionnaire d'encodage des types d'obstacles
        obstacle_types = {
            "Obstacle": 0.1,
            "Block": 0.2,
            "DoublePikes": 0.3,
            "TriplePikes": 0.4,
            "QuadruplePikes": 0.5,
            "BlockGapBlockWithSpike": 0.6,
            "BouncingObstacle": 0.7,
            "DoubleBlockPillar": 0.8,
            "FivePikesWithOrb": 0.9,
            "JumpPad": 0.95,
            "QuintuplePikesWithJumpPad": 0.98,
            "PurpleOrb": 0.99,
            "JumppadOrbsObstacle": 1.0,
            "EmptyObstacle": 0.0
        }
        
        return obstacle_types.get(obstacle_class, 0.0)
