import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
from collections import deque

# Définition du réseau neuronal pour l'apprentissage par renforcement
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Mémoire replay pour stocker les expériences
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Agent qui utilise le DQN pour choisir les actions
class DQNAgent:
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparamètres
        self.gamma = 0.99  # facteur d'actualisation
        self.epsilon = 1.0  # exploration vs exploitation
        self.epsilon_min = 0.01  # epsilon minimal
        self.epsilon_decay = 0.995  # taux de décroissance d'epsilon
        self.learning_rate = 0.001  # taux d'apprentissage
        self.batch_size = 64  # taille du batch pour l'apprentissage
        self.update_target_frequency = 10  # fréquence de mise à jour du réseau cible
        
        # Réseaux
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimiseur
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Mémoire
        self.memory = ReplayMemory(10000)
        
        # Compteurs
        self.training_step = 0
    
    def get_state(self, player, game_objects, ground_height, screen_width, screen_height):
        """
        Extraire les caractéristiques pertinentes du jeu pour former l'état
        """
        state = []
        
        # Caractéristiques du joueur (position relative, vitesse)
        state.append(player.rect.x / screen_width)
        state.append(player.rect.y / screen_height)
        state.append(player.vel_y / 20)  # Normaliser la vitesse
        state.append(1 if player.is_jumping else 0)
        
        # Caractéristiques des 3 obstacles les plus proches
        obstacles_data = []
        for obj in game_objects:
            # Position de l'obstacle
            if hasattr(obj, 'rect'):
                obstacle_x = obj.rect.x
                obstacle_y = obj.rect.y
                obstacle_width = obj.rect.width
                obstacle_height = obj.rect.height
            elif hasattr(obj, 'x'):
                obstacle_x = obj.x
                if hasattr(obj, 'width'):
                    obstacle_width = obj.width
                else:
                    obstacle_width = 50  # Valeur par défaut si width n'est pas disponible
                
                if hasattr(obj, 'get_rect'):
                    rect = obj.get_rect()
                    obstacle_y = rect.y
                    obstacle_height = rect.height
                else:
                    obstacle_y = ground_height - 50  # Valeur par défaut
                    obstacle_height = 50  # Valeur par défaut
            
            # Ne considérer que les obstacles devant le joueur
            if obstacle_x > player.rect.x:
                obstacles_data.append((obstacle_x, obstacle_y, obstacle_width, obstacle_height))
        
        # Trier les obstacles par distance au joueur
        obstacles_data.sort(key=lambda x: x[0])
        
        # Ajouter les données des 3 obstacles les plus proches
        for i in range(min(3, len(obstacles_data))):
            obstacle_x, obstacle_y, obstacle_width, obstacle_height = obstacles_data[i]
            state.append(obstacle_x / screen_width)  # Position X normalisée
            state.append(obstacle_y / screen_height)  # Position Y normalisée
            state.append(obstacle_width / screen_width)  # Largeur normalisée
            state.append(obstacle_height / screen_height)  # Hauteur normalisée
            state.append((obstacle_x - player.rect.right) / screen_width)  # Distance au joueur normalisée
        
        # Si moins de 3 obstacles, compléter avec des zéros
        remaining_obstacles = 3 - len(obstacles_data)
        for _ in range(remaining_obstacles):
            state.extend([0, 0, 0, 0, 0])  # 5 valeurs par obstacle
        
        return np.array(state, dtype=np.float32)
    
    def act(self, state, training=True):
        """
        Choisir une action selon la politique epsilon-greedy
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, next_state, reward, done):
        """
        Stocker l'expérience dans la mémoire
        """
        self.memory.push(state, action, next_state, reward, done)
    
    def update_epsilon(self):
        """
        Diminuer epsilon au fil du temps pour réduire l'exploration
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self):
        """
        Entraîner le réseau à partir d'un batch d'expériences
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Échantillonner un batch de transitions
        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*transitions)
        
        # Convertir en tenseurs PyTorch
        state_batch = torch.FloatTensor(batch_state).to(self.device)
        action_batch = torch.LongTensor(batch_action).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch_next_state).to(self.device)
        reward_batch = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(batch_done).unsqueeze(1).to(self.device)
        
        # Calculer les valeurs Q pour les actions prises
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Calculer les valeurs Q cibles
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1, keepdim=True)[0]
        
        # Calculer les valeurs Q cibles attendues
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        # Calculer la perte Huber (moins sensible aux valeurs aberrantes)
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        # Optimiser le modèle
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradient pour éviter explosion du gradient
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Mettre à jour le réseau cible
        self.training_step += 1
        if self.training_step % self.update_target_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save_model(self, path="models"):
        """
        Sauvegarder le modèle
        """
        if not os.path.exists(path):
            os.makedirs(path)
            
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, os.path.join(path, 'dqn_agent.pth'))
    
    def load_model(self, path="models"):
        """
        Charger un modèle sauvegardé
        """
        if os.path.exists(os.path.join(path, 'dqn_agent.pth')):
            checkpoint = torch.load(os.path.join(path, 'dqn_agent.pth'))
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']
            self.target_net.eval()
            return True
        return False

# Fonction principale pour l'entraînement ou la démonstration de l'IA
def ai_reinforcement_play(training_mode=True, episodes=1000):
    # Importations nécessaires pour le jeu
    from klass import Player
    import pygame
    
    # Définir les valeurs du jeu
    WIDTH, HEIGHT = 800, 600
    GROUND_HEIGHT = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du périphérique: {device}")
    
    # Initialiser l'agent
    state_size = 19  # 4 pour le joueur + 15 pour les obstacles (3 obstacles avec 5 caractéristiques chacun)
    action_size = 2   # 0: ne rien faire, 1: sauter
    agent = DQNAgent(state_size, action_size, device)

    # Créer le répertoire 'models' si nécessaire
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Charger le modèle existant si disponible et demandé
    model_loaded = False
    if not training_mode:
        model_loaded = agent.load_model()
        if model_loaded:
            print("Modèle chargé avec succès!")
            agent.epsilon = 0.01  # Mode démonstration, très peu d'exploration
        else:
            print("Aucun modèle trouvé. Utilisation d'un agent non entraîné.")
    
    # Statistiques d'entraînement
    scores = []
    epsilon_history = []
    
    # Boucle principale d'entraînement
    for episode in range(episodes):
        # Initialiser l'environnement du jeu
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Geometry Dash Clone - IA Reinforcement Learning")
        clock = pygame.time.Clock()
        
        # Créer le joueur et les objets du jeu
        player = Player()
        game_objects = []
        score = 0
        last_object = pygame.time.get_ticks()
        current_speed = 6  # Vitesse initiale
        
        # Paramètres du jeu (récupérés du fichier principal)
        object_interval = random.randint(800, 1400)
        
        # Dictionnaire pour les distances minimales entre obstacles
        min_obstacle_distances = {
            6: 150, 7: 175, 8: 500, 9: 250, 10: 275, 11: 300
        }
        
        # Dictionnaire pour les intervalles d'obstacles
        obstacle_intervals = {
            6: [800, 1400], 7: [900, 1600], 8: [1200, 1800],
            9: [1300, 2000], 10: [1400, 2100], 11: [1500, 2200]
        }
        
        # Obtenir l'état initial
        state = agent.get_state(player, game_objects, GROUND_HEIGHT, WIDTH, HEIGHT)
        
        # Variables pour le calcul de la récompense
        frames_alive = 0
        total_reward = 0
        
        running = True
        
        while running:
            current_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return
            
            # L'agent choisit une action
            action = agent.act(state, training_mode)
            
            # Appliquer l'action
            if action == 1 and not player.is_jumping:  # Sauter
                player.jump()
            
            # Logique de génération d'obstacles (similaire au jeu principal)
            can_spawn_obstacle = True
            min_distance = min_obstacle_distances[current_speed]
            
            if game_objects:
                last_obstacle = game_objects[-1]
                
                # Déterminer la position du dernier obstacle
                if hasattr(last_obstacle, 'rect'):
                    last_obstacle_right = last_obstacle.rect.right
                elif hasattr(last_obstacle, 'x') and hasattr(last_obstacle, 'width'):
                    last_obstacle_right = last_obstacle.x + last_obstacle.width
                else:
                    last_obstacle_right = getattr(last_obstacle, 'x', 0) + 50  # Valeur par défaut
                
                if WIDTH - last_obstacle_right < min_distance:
                    can_spawn_obstacle = False
            
            # Créer un nouvel obstacle si nécessaire
            if can_spawn_obstacle and current_time - last_object > object_interval:
                # Simuler la création d'obstacles comme dans le fichier principal
                # Pour simplifier, on n'utilise que quelques types d'obstacles ici
                from klass import Obstacle, Block, DoublePikes
                
                choice = random.random()
                if choice < 0.4:
                    obj = Obstacle(WIDTH)
                elif choice < 0.7:
                    obj = Block(WIDTH)
                else:
                    obj = DoublePikes(WIDTH)
                
                obj.set_speed(current_speed)
                game_objects.append(obj)
                
                last_object = current_time
                object_interval = random.randint(*obstacle_intervals[current_speed])
            
            # Mettre à jour l'état du jeu
            player.update(game_objects)
            
            # Vérifier si le jeu est terminé
            done = not player.is_alive
            
            # Mettre à jour les objets du jeu
            objects_to_remove = []
            for obj in game_objects:
                obj.update()
                
                # Vérifier si l'objet est hors de l'écran
                if ((hasattr(obj, 'x') and hasattr(obj, 'width') and obj.x + obj.width < 0) or
                    (hasattr(obj, 'rect') and obj.rect.right < 0)):
                    objects_to_remove.append(obj)
            
            # Supprimer les objets et augmenter le score
            for obj in objects_to_remove:
                if obj in game_objects:
                    game_objects.remove(obj)
                    score += 1
                    
                    # Ajuster la vitesse si nécessaire (comme dans le jeu original)
                    if score > 0 and score % 10 == 0 and current_speed < 11:
                        current_speed += 1
                        for game_obj in game_objects:
                            if hasattr(game_obj, 'set_speed'):
                                game_obj.set_speed(current_speed)
            
            # Calculer la récompense
            reward = 0.1  # Petite récompense pour rester en vie
            if done:
                reward = -10.0  # Pénalité pour perdre
            elif objects_to_remove:
                reward += 1.0  # Récompense pour avoir évité un obstacle

            frames_alive += 1
            total_reward += reward
            
            # Obtenir le nouvel état
            next_state = agent.get_state(player, game_objects, GROUND_HEIGHT, WIDTH, HEIGHT)
            
            # Stocker l'expérience dans la mémoire de l'agent
            if training_mode:
                agent.remember(state, action, next_state, reward, done)
                
                # Entraîner l'agent
                loss = agent.train()
            
            # Mettre à jour l'état
            state = next_state
            
            # Dessiner l'état du jeu
            screen.fill((255, 255, 255))
            pygame.draw.rect(screen, (0, 0, 0), (0, GROUND_HEIGHT, WIDTH, HEIGHT - GROUND_HEIGHT))
            
            player.draw(screen)
            
            for obj in game_objects:
                obj.draw(screen)
            
            # Afficher les informations
            font = pygame.font.SysFont(None, 36)
            score_text = font.render(f"Score: {score}", True, (0, 0, 0))
            screen.blit(score_text, (20, 20))
            
            mode_text = font.render(f"Mode: {'Entraînement' if training_mode else 'Démonstration'}", True, (0, 0, 0))
            screen.blit(mode_text, (20, 60))
            
            if training_mode:
                epsilon_text = font.render(f"Epsilon: {agent.epsilon:.3f}", True, (0, 0, 0))
                screen.blit(epsilon_text, (20, 100))
                
                episode_text = font.render(f"Épisode: {episode+1}/{episodes}", True, (0, 0, 0))
                screen.blit(episode_text, (20, 140))
            
            pygame.display.flip()
            clock.tick(60)
            
            if done:
                break
        
        # Fin de l'épisode
        if training_mode:
            agent.update_epsilon()
        
        scores.append(score)
        epsilon_history.append(agent.epsilon)
        
        # Afficher les statistiques
        avg_score = np.mean(scores[-100:]) if len(scores) > 0 else 0
        print(f"Épisode: {episode+1}/{episodes}, Score: {score}, Moyenne (100 derniers): {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        # Sauvegarder le modèle tous les 100 épisodes
        if training_mode and (episode + 1) % 100 == 0:
            agent.save_model()
            print(f"Modèle sauvegardé à l'épisode {episode+1}")
        
        pygame.quit()
    
    # Sauvegarder le modèle final
    if training_mode:
        agent.save_model()
        print("Entraînement terminé. Modèle final sauvegardé!")
    
    return scores, epsilon_history

# Fonction pour visualiser les performances lors de l'entraînement
def plot_training_results(scores, epsilons):
    import matplotlib.pyplot as plt
    
    # Créer une nouvelle figure
    plt.figure(figsize=(12, 5))
    
    # Graphique des scores
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Score par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Score')
    
    # Moyenne mobile pour voir la tendance
    window_size = min(100, len(scores))
    if window_size > 0:
        moving_avg = [np.mean(scores[max(0, i-window_size):i+1]) for i in range(len(scores))]
        plt.plot(moving_avg, color='red')
        plt.legend(['Score', 'Moyenne mobile'])
    
    # Graphique d'epsilon
    plt.subplot(1, 2, 2)
    plt.plot(epsilons)
    plt.title('Epsilon par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

# Fonction pour tester le meilleur modèle
def best_ai_play():
    # Charger et exécuter le meilleur modèle en mode démonstration
    scores, _ = ai_reinforcement_play(training_mode=False, episodes=5)
    print(f"Score moyen sur 5 parties: {np.mean(scores):.2f}")
    return scores

# Si ce fichier est exécuté directement
if __name__ == "__main__":
    import sys
    
    # Vérifier les arguments de ligne de commande
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "train":
            print("Mode d'entraînement activé")
            episodes = 1000
            if len(sys.argv) > 2:
                try:
                    episodes = int(sys.argv[2])
                except ValueError:
                    pass
            scores, epsilons = ai_reinforcement_play(training_mode=True, episodes=episodes)
            plot_training_results(scores, epsilons)
        elif mode == "test":
            print("Mode de test activé")
            best_ai_play()
    else:
        # Mode par défaut
        print("Mode d'entraînement par défaut")
        print("Pour spécifier un mode: python reinforcement_ai.py [train|test] [nb_episodes]")
        scores, epsilons = ai_reinforcement_play(training_mode=True, episodes=500)
        plot_training_results(scores, epsilons)
        
class QLearningAgent:
    """Agent d'apprentissage par renforcement utilisant Q-learning"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995, state_size=21):
        """Initialiser l'agent
        
        Args:
            learning_rate (float): Taux d'apprentissage
            discount_factor (float): Facteur de réduction pour les récompenses futures
            exploration_rate (float): Taux d'exploration initial
            exploration_decay (float): Taux de décroissance de l'exploration
            state_size (int): Taille de l'état (nombre de caractéristiques)
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay
        self.actions = [0, 1]  # 0: ne pas sauter, 1: sauter
        
        # Pour un problème complexe comme celui-ci, nous utilisons un dictionnaire pour stocker les valeurs Q
        self.q_table = {}
        
        # Discrétiser l'espace d'états continu
        self.state_size = state_size
        self.bins = {}
        self.setup_bins()
    
    def setup_bins(self):
        """Configurer les bins pour discrétiser l'état continu"""
        # Player y position (0 à 1)
        self.bins['player_y'] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Player velocity (-1 à 1 normalisé)
        self.bins['player_velocity'] = [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]
        
        # Player jumping state (booléen)
        self.bins['is_jumping'] = [0, 1]
        
        # Game speed (0 à 1 normalisé)
        self.bins['game_speed'] = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Pour les obstacles (4 obstacles avec 4 caractéristiques chacun)
        for i in range(4):
            # Distance (0 à 1 normalisé)
            self.bins[f'distance_{i}'] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
            
            # Hauteur (0 à 1)
            self.bins[f'height_{i}'] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            
            # Largeur (0 à 1)
            self.bins[f'width_{i}'] = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
            
            # Type (0, 1, 2, 3, 4)
            self.bins[f'type_{i}'] = [0, 1, 2, 3, 4]
    
    def discretize_state(self, state):
        """Convertir un état continu en un état discret
        
        Args:
            state: Un tuple avec les caractéristiques d'état
            
        Returns:
            tuple: Une version discrétisée de l'état
        """
        player_y, player_velocity, is_jumping, game_speed, *obstacles_info = state
        
        # Discrétiser les caractéristiques du joueur
        discrete_player_y = self._get_bin_index(player_y, self.bins['player_y'])
        discrete_player_velocity = self._get_bin_index(player_velocity, self.bins['player_velocity'])
        discrete_is_jumping = int(is_jumping)  # Déjà binaire
        discrete_game_speed = self._get_bin_index(game_speed, self.bins['game_speed'])
        
        # Discrétiser les caractéristiques des obstacles
        discrete_obstacles = []
        for i in range(4):  # 4 obstacles
            if i*4 + 3 < len(obstacles_info):
                distance = obstacles_info[i*4]
                height = obstacles_info[i*4 + 1]
                width = obstacles_info[i*4 + 2]
                obj_type = obstacles_info[i*4 + 3]
                
                # Simplifier l'état en ne conservant que les caractéristiques des obstacles proches
                if distance < 0.5:  # Ne considérer que les obstacles relativement proches
                    discrete_distance = self._get_bin_index(distance, self.bins[f'distance_{i}'])
                    discrete_height = self._get_bin_index(height, self.bins[f'height_{i}'])
                    discrete_width = self._get_bin_index(width, self.bins[f'width_{i}'])
                    discrete_type = int(obj_type)  # Déjà discrétisé
                    
                    discrete_obstacles.extend([discrete_distance, discrete_height, discrete_width, discrete_type])
                else:
                    # Si l'obstacle est loin, simplifier sa représentation
                    discrete_obstacles.extend([len(self.bins[f'distance_{i}']) - 1, 0, 0, 0])
        
        # Créer un tuple de l'état discrétisé en ne gardant que les caractéristiques importantes
        # Pour simplifier, ne considérons que le joueur et le premier obstacle
        discrete_state = (
            discrete_player_y,
            discrete_player_velocity,
            discrete_is_jumping,
            discrete_game_speed,
        )
        
        # Ajouter seulement le premier obstacle pour réduire la dimension de l'espace d'états
        if len(discrete_obstacles) >= 4:
            discrete_state += tuple(discrete_obstacles[:4])
        
        return discrete_state
    
    def _get_bin_index(self, value, bins):
        """Trouver l'indice du bin pour une valeur donnée
        
        Args:
            value: La valeur à discrétiser
            bins: Liste des limites des bins
            
        Returns:
            int: L'indice du bin correspondant
        """
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                return i
        return len(bins) - 2  # Retourner l'avant-dernier indice si valeur >= dernier bin
    
    def get_action(self, state):
        """Choisir une action en fonction de l'état actuel
        
        Args:
            state: L'état actuel du jeu
            
        Returns:
            int: L'action choisie (0 ou 1)
        """
        # Discrétiser l'état
        discrete_state = self.discretize_state(state)
        
        # Exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            return np.random.choice(self.actions)  # Explorer: action aléatoire
        else:
            # Exploiter: choisir la meilleure action connue
            return self.get_best_action(discrete_state)
    
    def get_best_action(self, state):
        """Obtenir la meilleure action pour un état donné
        
        Args:
            state: L'état discrétisé
            
        Returns:
            int: La meilleure action (0 ou 1)
        """
        # Si l'état n'a jamais été vu, initialiser les valeurs Q à 0
        if state not in self.q_table:
            self.q_table[state] = {0: 0, 1: 0}
        
        # Choisir l'action avec la valeur Q la plus élevée
        # En cas d'égalité, choisir aléatoirement
        q_values = self.q_table[state]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        
        return np.random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state, done):
        """Mettre à jour la table Q en fonction de l'expérience
        
        Args:
            state: L'état actuel
            action: L'action effectuée
            reward: La récompense obtenue
            next_state: L'état suivant
            done: Booléen indiquant si l'épisode est terminé
        """
        # Discrétiser les états
        current_state = self.discretize_state(state)
        next_state_disc = self.discretize_state(next_state)
        
        # S'assurer que l'état actuel est dans la table Q
        if current_state not in self.q_table:
            self.q_table[current_state] = {0: 0, 1: 0}
        
        # Obtenir la valeur Q actuelle
        current_q = self.q_table[current_state].get(action, 0)
        
        # Calculer la cible Q
        if done:
            # Si l'épisode est terminé, il n'y a pas d'état futur
            target_q = reward
        else:
            # S'assurer que l'état suivant est dans la table Q
            if next_state_disc not in self.q_table:
                self.q_table[next_state_disc] = {0: 0, 1: 0}
            
            # Obtenir la meilleure valeur Q pour l'état suivant
            next_max_q = max(self.q_table[next_state_disc].values())
            
            # Calculer la cible Q en utilisant l'équation de Bellman
            target_q = reward + self.discount_factor * next_max_q
        
        # Mettre à jour la valeur Q
        self.q_table[current_state][action] = current_q + self.learning_rate * (target_q - current_q)
        
        # Réduire le taux d'exploration
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
    
    def save_model(self, filepath):
        """Sauvegarder le modèle dans un fichier
        
        Args:
            filepath: Chemin du fichier pour la sauvegarde
        """
        model_data = {
            'q_table': self.q_table,
            'exploration_rate': self.exploration_rate
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Charger le modèle depuis un fichier
        
        Args:
            filepath: Chemin du fichier pour le chargement
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
                self.q_table = model_data['q_table']
                self.exploration_rate = model_data['exploration_rate']
            
            return True
        else:
            return False

def ai_reinforcement_play(training_mode=True, episodes=100, headless=True):
    """Mode de jeu utilisant l'IA par renforcement
    
    Args:
        training_mode (bool): Si True, l'agent sera entraîné, sinon il jouera avec le modèle existant
        episodes (int): Nombre d'épisodes d'entraînement
        headless (bool): Si True, le jeu sera exécuté sans interface graphique
    """
    env = GameEnvironment(headless=headless)
    agent = QLearningAgent()
    
    # Si nous ne sommes pas en mode entraînement, essayer de charger un modèle existant
    if not training_mode:
        if agent.load_model(MODEL_PATH):
            print("Modèle chargé avec succès!")
            # En mode démo, nous voulons explorer moins
            agent.exploration_rate = 0.01
        else:
            print("Aucun modèle trouvé. Utilisation d'un modèle non entraîné.")
    
    # Statistiques
    all_rewards = []
    all_scores = []
    best_score = 0
    best_episode = 0
    model_saved = False
    
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 10000:  # Limiter à 10000 étapes par épisode
            # Choisir une action
            action = agent.get_action(state)
            
            # Effectuer l'action
            next_state, reward, done, info = env.step(action)
            
            # Apprendre de l'expérience (si en mode entraînement)
            if training_mode:
                agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Enregistrer les statistiques
        all_rewards.append(total_reward)
        all_scores.append(env.score)
        
        # Afficher les progrès
        avg_reward = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
        avg_score = np.mean(all_scores[-100:]) if len(all_scores) >= 100 else np.mean(all_scores)
        
        print(f"Episode {episode+1}/{episodes}, Score: {env.score}, Avg Score: {avg_score:.2f}, Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Exploration: {agent.exploration_rate:.4f}")
        
        # Sauvegarder le meilleur modèle
        if env.score > best_score:
            best_score = env.score
            best_episode = episode + 1
            if training_mode:
                agent.save_model(MODEL_PATH)
                model_saved = True
                print(f"Nouveau meilleur score! Modèle sauvegardé avec un score de {best_score} à l'épisode {best_episode}")
    
    # Sauvegarder le dernier modèle si aucun meilleur n'a été trouvé
    if training_mode and not model_saved:
        agent.save_model(MODEL_PATH)
        print("Modèle final sauvegardé.")
    
    duration = time.time() - start_time
    print(f"Entraînement/démonstration terminé en {duration:.2f} secondes.")
    print(f"Meilleur score: {best_score} à l'épisode {best_episode}")
    
    # Si nous ne sommes pas en mode headless, attendre avant de quitter
    if not headless:
        input("Appuyez sur Entrée pour continuer...")

def best_ai_play():
    """Démonstration du meilleur agent IA"""
    env = GameEnvironment(headless=False)  # Mode visuel pour la démonstration
    agent = QLearningAgent()
    
    # Essayer de charger le modèle
    if not agent.load_model(MODEL_PATH):
        print("Aucun modèle trouvé. Veuillez d'abord entraîner l'IA.")
        input("Appuyez sur Entrée pour continuer...")
        return
    
    # Désactiver l'exploration pour la démonstration
    agent.exploration_rate = 0.0
    
    # Statistiques
    episodes = 5  # Nombre d'épisodes de démonstration
    all_scores = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Choisir l'action optimale
            action = agent.get_best_action(agent.discretize_state(state))
            
            # Effectuer l'action
            next_state, _, done, _ = env.step(action)
            
            state = next_state
            
            # Permettre à l'utilisateur de quitter
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
                # Permettre à l'utilisateur de passer à l'épisode suivant avec la touche 'n'
                if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                    done = True
        
        all_scores.append(env.score)
        print(f"Episode {episode+1}, Score: {env.score}")
    
    avg_score = sum(all_scores) / len(all_scores)
    print(f"Score moyen sur {episodes} épisodes: {avg_score:.2f}")
    
    input("Démonstration terminée. Appuyez sur Entrée pour continuer...")

# Ajout d'une classe Button pour le menu
class Button:
    """Classe pour créer des boutons cliquables"""
    
    def __init__(self, text, x, y, width, height, color, hover_color):
        """Initialiser un bouton
        
        Args:
            text (str): Texte à afficher sur le bouton
            x, y (int): Position du bouton
            width, height (int): Dimensions du bouton
            color (tuple): Couleur normale du bouton (R, G, B)
            hover_color (tuple): Couleur du bouton au survol (R, G, B)
        """
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.current_color = color
        self.font = pygame.font.SysFont(None, 32)
        
    def draw(self, surface):
        """Dessiner le bouton
        
        Args:
            surface: Surface pygame sur laquelle dessiner
        """
        pygame.draw.rect(surface, self.current_color, self.rect, border_radius=10)
        pygame.draw.rect(surface, BLACK, self.rect, 2, border_radius=10)  # Bordure
        
        text_surface = self.font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
        
    def update(self, mouse_pos):
        """Mettre à jour la couleur du bouton en fonction de la position de la souris
        
        Args:
            mouse_pos (tuple): Position de la souris (x, y)
        """
        if self.rect.collidepoint(mouse_pos):
            self.current_color = self.hover_color
        else:
            self.current_color = self.color
            
    def check_clicked(self, mouse_pos, mouse_clicked):
        """Vérifier si le bouton a été cliqué
        
        Args:
            mouse_pos (tuple): Position de la souris (x, y)
            mouse_clicked (bool): Si la souris a été cliquée
            
        Returns:
            bool: True si le bouton a été cliqué
        """
        return self.rect.collidepoint(mouse_pos) and mouse_clicked

# Si ce fichier est exécuté directement, lancez le menu principal
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Geometry Dash Clone")
    clock = pygame.time.Clock()
    
    show_menu()