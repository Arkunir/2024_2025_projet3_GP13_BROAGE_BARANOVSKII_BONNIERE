# # reinforcement_agent.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np
# import random
# from collections import deque
# import os

# class DQNModel(nn.Module):
#     """Réseau de neurones pour le Deep Q-Learning"""
    
#     def __init__(self, input_size, output_size):
#         super(DQNModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, output_size)
    
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

# class ReplayBuffer:
#     """Mémoire d'expériences pour le replay"""
    
#     def __init__(self, capacity):
#         self.memory = deque(maxlen=capacity)
    
#     def push(self, state, action, reward, next_state, done):
#         """Ajoute une transition à la mémoire"""
#         self.memory.append((state, action, reward, next_state, done))
    
#     def sample(self, batch_size):
#         """Échantillonne un batch de transitions depuis la mémoire"""
#         return random.sample(self.memory, batch_size)
    
#     def __len__(self):
#         return len(self.memory)

# class DQNAgent:
#     """Agent utilisant l'algorithme Deep Q-Network"""
    
#     def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99,
#                  epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
#                  buffer_size=10000, batch_size=64, update_target_every=100):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.gamma = gamma  # Facteur de remise
#         self.epsilon = epsilon_start  # Taux d'exploration
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.batch_size = batch_size
#         self.update_target_every = update_target_every
#         self.update_counter = 0
        
#         # Utiliser CUDA si disponible
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using device: {self.device}")
        
#         # Créer les réseaux de neurones (principal et cible)
#         self.model = DQNModel(state_size, action_size).to(self.device)
#         self.target_model = DQNModel(state_size, action_size).to(self.device)
#         self.target_model.load_state_dict(self.model.state_dict())
#         self.target_model.eval()  # Mode évaluation pour le réseau cible
        
#         # Optimizer et critère de perte
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#         self.criterion = nn.MSELoss()
        
#         # Mémoire de replay
#         self.memory = ReplayBuffer(buffer_size)
    
#     def act(self, state, evaluate=False):
#         """Choisit une action selon la politique epsilon-greedy"""
#         if not evaluate and random.random() <= self.epsilon:
#             return random.randrange(self.action_size)  # Action aléatoire (exploration)
        
#         # Convertir l'état en tensor
#         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
#         # Mode évaluation pour éviter le calcul de gradients inutile
#         with torch.no_grad():
#             q_values = self.model(state)
        
#         # Choisir l'action avec la plus grande valeur Q
#         return q_values.argmax().item()  # Action optimale (exploitation)
    
#     def remember(self, state, action, reward, next_state, done):
#         """Stocke une transition dans la mémoire de replay"""
#         self.memory.push(state, action, reward, next_state, done)
    
#     def replay(self):
#         """Effectue une étape d'apprentissage par expérience replay"""
#         if len(self.memory) < self.batch_size:
#             return
        
#         # Échantillonner un batch de transitions depuis la mémoire
#         batch = self.memory.sample(self.batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
        
#         # Convertir en tensors
#         states = torch.FloatTensor(np.array(states)).to(self.device)
#         actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
#         rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
#         next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
#         dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
#         # Calcul des Q-values actuels pour les actions prises
#         current_q = self.model(states).gather(1, actions)
        
#         # Calcul des Q-values cibles pour les actions optimales
#         with torch.no_grad():
#             next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
        
#         # Calcul des Q-values cibles selon l'équation de Bellman
#         target_q = rewards + self.gamma * next_q * (1 - dones)
        
#         # Calcul de la perte et mise à jour des poids
#         loss = self.criterion(current_q, target_q)
        
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
        
#         # Mise à jour du réseau cible si nécessaire
#         self.update_counter += 1
#         if self.update_counter % self.update_target_every == 0:
#             self.target_model.load_state_dict(self.model.state_dict())
        
#         # Réduire epsilon (moins d'exploration au fil du temps)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
    
#     def save(self, path):
#         """Sauvegarde le modèle"""
#         torch.save({
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'epsilon': self.epsilon
#         }, path)
#         print(f"Agent sauvegardé dans {path}")
    
#     def load(self, path):
#         """Charge le modèle"""
#         if os.path.isfile(path):
#             checkpoint = torch.load(path, map_location=self.device)
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             self.target_model.load_state_dict(checkpoint['model_state_dict'])
#             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             self.epsilon = checkpoint['epsilon']
#             print(f"Agent chargé depuis {path}")
#         else:
#             print(f"Fichier {path} non trouvé, aucun modèle chargé.")