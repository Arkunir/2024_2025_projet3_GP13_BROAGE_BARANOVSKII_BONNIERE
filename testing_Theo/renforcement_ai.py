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