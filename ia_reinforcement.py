import numpy as np
import pygame
import random
import pickle
import os
from collections import deque
from klass import Player
from klass import MovingObject
from klass import Obstacle
from klass import DoublePikes
from klass import TriplePikes   
from klass import QuadruplePikes
from klass import Block
from klass import BlockGapBlockWithSpike
from klass import BouncingObstacle
from klass import DoubleBlockPillar
from klass import FivePikesWithOrb
from klass import JumpPad
from klass import QuintuplePikesWithJumpPad
from klass import PurpleOrb
from klass import JumppadOrbsObstacle
    

class GeometryDashAI:
    """
    Agent d'apprentissage par renforcement pour Geometry Dash Clone utilisant Q-learning
    """
    def __init__(self, state_size=4, action_size=2, load_model=True):
        self.state_size = state_size      # Taille du vecteur d'état 
        self.action_size = action_size    # Nombre d'actions possibles (sauter ou ne pas sauter)
        
        # Paramètres d'apprentissage
        self.gamma = 0.95                 # Facteur d'actualisation
        self.epsilon = 1.0                # Exploration vs exploitation
        self.epsilon_min = 0.01           # Valeur minimale d'epsilon
        self.epsilon_decay = 0.995        # Taux de décroissance d'epsilon
        self.learning_rate = 0.001        # Taux d'apprentissage
        
        # Mémoire pour stocker les expériences
        self.memory = deque(maxlen=2000)
        
        # Table Q - initialisation à zéro
        # Les états sont discrétisés (distance à l'obstacle, type d'obstacle, vitesse de jeu, position du joueur)
        self.q_table = {}
        
        # Charger le modèle sauvegardé s'il existe
        self.model_path = 'geometry_dash_ai_model.pkl'
        if load_model and os.path.exists(self.model_path):
            self.load_model()
    
    def preprocess_state(self, raw_state):
        """
        Discrétise l'état brut pour le rendre utilisable par notre table Q
        """
        distance_to_obstacle, obstacle_type, game_speed, player_y = raw_state
        
        # Discrétisation des valeurs continues
        distance_bins = [0, 100, 200, 300, 400, 500, 600, float('inf')]
        distance_idx = next(i for i, val in enumerate(distance_bins) if distance_to_obstacle < val)
        
        player_y_bins = [0, 300, 350, 400, 450, 500, float('inf')]
        player_y_idx = next(i for i, val in enumerate(player_y_bins) if player_y < val)
        
        # Convertir en tuple pour pouvoir l'utiliser comme clé de dictionnaire
        return (distance_idx, obstacle_type, game_speed, player_y_idx)
    
    def get_action(self, state):
        """
        Sélectionne une action selon la politique epsilon-greedy
        """
        processed_state = self.preprocess_state(state)
        
        # Exploration: choix aléatoire
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation: choisir la meilleure action selon Q-table
        if processed_state not in self.q_table:
            self.q_table[processed_state] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[processed_state])
    
    def memorize(self, state, action, reward, next_state, done):
        """
        Stocke l'expérience dans la mémoire
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """
        Rejoue les expériences pour l'apprentissage
        """
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            processed_state = self.preprocess_state(state)
            processed_next_state = self.preprocess_state(next_state)
            
            # S'assurer que les états existent dans la Q-table
            if processed_state not in self.q_table:
                self.q_table[processed_state] = np.zeros(self.action_size)
            
            if processed_next_state not in self.q_table:
                self.q_table[processed_next_state] = np.zeros(self.action_size)
            
            # Mise à jour Q-value
            target = reward
            if not done:
                target += self.gamma * np.max(self.q_table[processed_next_state])
            
            current_q = self.q_table[processed_state][action]
            self.q_table[processed_state][action] += self.learning_rate * (target - current_q)
        
        # Décrémenter epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self):
        """
        Sauvegarde la table Q
        """
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Modèle sauvegardé dans {self.model_path}")
    
    def load_model(self):
        """
        Charge la table Q
        """
        try:
            with open(self.model_path, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Modèle chargé depuis {self.model_path}")
            # Réduire l'exploration si on charge un modèle
            self.epsilon = 0.1
        except:
            print("Aucun modèle existant n'a été trouvé. Création d'un nouveau modèle.")


def ai_reinforcement_play():
    """
    Mode de jeu avec IA par renforcement
    """
    print("Mode IA par renforcement activé")
    
    # Initialisation du jeu

    WIDTH, HEIGHT = 800, 600
    FPS = 60
    GRAVITY = 1
    JUMP_STRENGTH = -18
    GROUND_HEIGHT = 500
    CUBE_SIZE = 50
    INITIAL_SCROLL_SPEED = 6
    
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Geometry Dash Clone - IA Renforcement")
    clock = pygame.time.Clock()
    
    # Initialisation de l'IA
    agent = GeometryDashAI(state_size=4, action_size=2, load_model=True)
    
    # États pour l'IA
    batch_size = 32
    scores = []
    episodes = 100  # Nombre maximum d'épisodes d'entraînement
    
    # Configurations du jeu (repris de main())
    min_obstacle_distances = {
        6: 150, 7: 175, 8: 500, 9: 250, 10: 275, 11: 300
    }
    
    obstacle_intervals = {
        6: [800, 1400], 7: [900, 1600], 8: [1200, 1800],
        9: [1300, 2000], 10: [1400, 2100], 11: [1500, 2200]
    }
    
    # Statistiques d'entraînement
    max_score = 0
    training_info_font = pygame.font.SysFont(None, 24)
    
    for e in range(episodes):
        # Initialisation pour un épisode
        player = Player()
        game_objects = []
        score = 0
        last_object = pygame.time.get_ticks()
        
        current_speed = INITIAL_SCROLL_SPEED
        previous_speed = current_speed
        object_interval = random.randint(*obstacle_intervals[INITIAL_SCROLL_SPEED])
        
        speed_threshold_7 = random.randint(10, 20)
        min_threshold_8 = max(25, 2 * speed_threshold_7 - 5)
        max_threshold_8 = 2 * speed_threshold_7 + 10
        speed_threshold_8 = random.randint(min_threshold_8, max_threshold_8)
        
        min_threshold_9 = max(40, 2 * speed_threshold_8 - 15)
        max_threshold_9 = 2 * speed_threshold_8 + 5             
        speed_threshold_9 = random.randint(min_threshold_9, max_threshold_9)
        
        speed_threshold_random = 100
        next_random_change = speed_threshold_random + random.randint(25, 50)
        
        # État initial 
        state = [WIDTH, 0, current_speed, player.y]  # [distance_to_obstacle, obstacle_type, game_speed, player_y]
        total_reward = 0
        space_pressed = False
        ai_action = 0  # 0: ne pas sauter, 1: sauter
        
        running = True
        episode_time_start = pygame.time.get_ticks()
        
        # Boucle de jeu
        while running:
            current_time = pygame.time.get_ticks()
            
            # Gestion des événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Sauvegarder le modèle avant de quitter
                    agent.save_model()
                    pygame.quit()
                    import sys
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # Sauvegarder et retourner au menu
                        agent.save_model()
                        running = False
            
            # Partie IA : obtenir l'observation de l'environnement
            # Trouver l'obstacle le plus proche devant le joueur
            closest_obstacle = None
            closest_distance = float('inf')
            obstacle_type = 0  # Type par défaut
            
            for obj in game_objects:
                # Calculer la position de l'obstacle
                obj_x = 0
                obj_width = 0
                
                if isinstance(obj, Obstacle):
                    obj_x = obj.x
                    obj_width = obj.width
                    obstacle_type = 1
                elif isinstance(obj, Block):
                    obj_x = obj.rect.left
                    obj_width = obj.rect.width
                    obstacle_type = 2
                elif isinstance(obj, DoublePikes):
                    obj_x = obj.x
                    obj_width = obj.width
                    obstacle_type = 3
                elif isinstance(obj, TriplePikes):
                    obj_x = obj.x
                    obj_width = obj.width
                    obstacle_type = 4
                elif isinstance(obj, QuadruplePikes):
                    obj_x = obj.x
                    obj_width = obj.width
                    obstacle_type = 5
                elif isinstance(obj, BouncingObstacle):
                    obj_x = obj.x
                    obj_width = obj.width
                    obstacle_type = 6
                elif isinstance(obj, DoubleBlockPillar):
                    obj_x = obj.x
                    obj_width = obj.width
                    obstacle_type = 7
                elif isinstance(obj, BlockGapBlockWithSpike):
                    obj_x = obj.x
                    obj_width = obj.width
                    obstacle_type = 8
                elif isinstance(obj, JumpPad):
                    obj_x = obj.x
                    obj_width = obj.width
                    obstacle_type = 9
                elif isinstance(obj, QuintuplePikesWithJumpPad):
                    obj_x = obj.x
                    obj_width = obj.width
                    obstacle_type = 10
                elif isinstance(obj, FivePikesWithOrb):
                    obj_x = obj.x
                    obj_width = obj.width
                    obstacle_type = 11
                elif isinstance(obj, PurpleOrb):
                    obj_x = obj.x
                    obj_width = obj.width
                    obstacle_type = 12
                elif isinstance(obj, JumppadOrbsObstacle):
                    obj_x = obj.x
                    obj_width = obj.width
                    obstacle_type = 13
                
                # Vérifier si c'est l'obstacle le plus proche devant le joueur
                if obj_x > player.x and obj_x - player.x < closest_distance:
                    closest_distance = obj_x - player.x
                    closest_obstacle = obj
            
            # Si aucun obstacle n'est devant, on met une grande valeur
            if closest_obstacle is None:
                closest_distance = WIDTH
            
            # Construire l'état actuel pour l'agent
            next_state = [closest_distance, obstacle_type, current_speed, player.y]
            
            # Obtenir l'action de l'IA
            ai_action = agent.get_action(next_state)
            
            # Appliquer l'action
            if ai_action == 1:  # Sauter
                if not space_pressed or player.just_landed:
                    if not player.is_jumping:
                        player.jump()
                space_pressed = True
            else:
                space_pressed = False
            
            # Calculer la récompense
            reward = 0.1  # Petite récompense pour survivre
            if not player.is_alive:
                reward = -10  # Pénalité pour mourir
            elif closest_distance < 100 and player.is_alive:
                reward = 1  # Bonne récompense pour avoir évité un obstacle proche
            
            # Gérer la création d'objets (comme dans main())
            can_spawn_obstacle = True
            min_distance = min_obstacle_distances[current_speed]
            
            if game_objects:
                last_obstacle = game_objects[-1]
                
                # Calculer la fin du dernier obstacle (comme dans main())
                # [Code condensé pour la lisibilité]
                if isinstance(last_obstacle, Obstacle):
                    last_obstacle_right = last_obstacle.x + last_obstacle.width
                elif isinstance(last_obstacle, Block):
                    last_obstacle_right = last_obstacle.rect.right
                else:  # Pour tous les autres types
                    last_obstacle_right = last_obstacle.x + last_obstacle.width
                
                if WIDTH - last_obstacle_right < min_distance:
                    can_spawn_obstacle = False
            
            # Créer un nouvel obstacle si les conditions sont remplies
            if can_spawn_obstacle and current_time - last_object > object_interval:
                obj = None
                
                # Sélection d'obstacle basée sur la vitesse (comme dans main())
                # [Logique condensée pour la lisibilité]
                if current_speed <= 6:
                    choice = random.random()
                    if choice < 0.35:
                        obj = Obstacle(WIDTH)
                    elif choice < 0.5:
                        obj = JumpPad(WIDTH)
                    elif choice < 0.65:
                        obj = Block(WIDTH)
                    else:
                        obj = PurpleOrb(WIDTH)
                else:
                    choice = random.random()
                    if choice < 0.3:
                        obj = DoublePikes(WIDTH)
                    elif choice < 0.5:
                        obj = BlockGapBlockWithSpike(WIDTH)
                    elif choice < 0.7:
                        obj = PurpleOrb(WIDTH)
                    else:
                        obj = JumppadOrbsObstacle(WIDTH)
                
                # Vérifier que obj a bien été défini avant de l'utiliser
                if obj:
                    obj.set_speed(current_speed)
                    game_objects.append(obj)
                    
                    last_object = current_time
                    object_interval = random.randint(*obstacle_intervals[current_speed])
            
            # Mettre à jour le joueur
            player.update(game_objects)
            
            # Vérifier si le joueur est mort
            if not player.is_alive:
                print(f"Episode {e+1}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.4f}")
                running = False
            
            # Gestion des objets et collisions (comme dans main())
            objects_to_remove = []
            
            # [Code de collision condensé pour la lisibilité - similaire au main()]
            for obj in game_objects[:]:
                obj.update()
                
                # Vérifier les collisions en fonction du type d'objet
                if check_collision(obj, player):
                    player.is_alive = False
                    running = False
                    break
                
                # Vérifier si l'objet est hors de l'écran
                if is_offscreen(obj):
                    objects_to_remove.append(obj)
            
            # Supprimer les objets hors écran et incrémenter le score
            for obj in objects_to_remove:
                if obj in game_objects:
                    game_objects.remove(obj)
                    score += 1
                    reward += 1  # Récompense pour chaque obstacle passé
                    
                    # Mise à jour de la vitesse en fonction du score (comme dans main())
                    # [Logique de changement de vitesse]
                    if score < speed_threshold_random:
                        if score == speed_threshold_7 and current_speed < 7:
                            current_speed = 7
                        elif score == speed_threshold_8 and current_speed < 8:
                            current_speed = 8
                        elif score == speed_threshold_9 and current_speed < 9:
                            current_speed = 9
                    elif score == speed_threshold_random or (score >= speed_threshold_random and score == next_random_change):
                        new_speed = random.randint(9, 11)
                        current_speed = new_speed
                        next_random_change = score + random.randint(25, 50)
                    
                    # Vérifier si la vitesse a changé
                    if current_speed != previous_speed:
                        if hasattr(player, 'change_skin_randomly'):
                            player.change_skin_randomly()
                        previous_speed = current_speed
            
            # Mémoriser l'expérience
            is_done = not player.is_alive or not running
            agent.memorize(state, ai_action, reward, next_state, is_done)
            state = next_state
            total_reward += reward
            
            # Apprentissage par lots
            agent.replay(batch_size)
            
            # Affichage
            screen.fill(WHITE)
            pygame.draw.rect(screen, BLACK, (0, GROUND_HEIGHT, WIDTH, HEIGHT - GROUND_HEIGHT))
            
            player.draw(screen)
            
            for obj in game_objects:
                obj.draw(screen)
            
            # Afficher les informations
            font = pygame.font.SysFont(None, 36)
            score_text = font.render(f"Score: {score}", True, BLACK)
            screen.blit(score_text, (20, 20))
            
            speed_text = font.render(f"Vitesse: {current_speed}", True, BLACK)
            screen.blit(speed_text, (20, 60))
            
            # Informations d'entraînement
            info_y = 100
            episode_text = training_info_font.render(f"Episode: {e+1}/{episodes}", True, BLACK)
            screen.blit(episode_text, (20, info_y))
            
            epsilon_text = training_info_font.render(f"Epsilon: {agent.epsilon:.4f}", True, BLACK)
            screen.blit(epsilon_text, (20, info_y + 25))
            
            action_text = training_info_font.render(f"Action: {'JUMP' if ai_action == 1 else 'WAIT'}", True, RED if ai_action == 1 else BLACK)
            screen.blit(action_text, (20, info_y + 50))
            
            reward_text = training_info_font.render(f"Récompense: {total_reward:.1f}", True, BLACK)
            screen.blit(reward_text, (20, info_y + 75))
            
            max_score_text = training_info_font.render(f"Meilleur score: {max_score}", True, BLACK)
            screen.blit(max_score_text, (20, info_y + 100))
            
            dist_text = training_info_font.render(f"Distance: {int(closest_distance)}", True, BLACK)
            screen.blit(dist_text, (20, info_y + 125))
            
            if score >= speed_threshold_random:
                next_change_text = font.render(f"Prochain changement à: {next_random_change}", True, BLACK)
                screen.blit(next_change_text, (20, info_y + 150))
            
            pygame.display.flip()
            clock.tick(FPS)
        
        # À la fin de l'épisode
        scores.append(score)
        if score > max_score:
            max_score = score
            agent.save_model()  # Sauvegarder le modèle si nouveau meilleur score
        
        # Pause entre les épisodes
        screen.fill(WHITE)
        episode_text = font.render(f"Fin de l'épisode {e+1}/{episodes}", True, BLACK)
        score_text = font.render(f"Score: {score}", True, BLACK)
        best_score_text = font.render(f"Meilleur score: {max_score}", True, BLACK)
        wait_text = font.render("Attente de 2 secondes avant le prochain épisode...", True, BLACK)
        
        screen.blit(episode_text, (WIDTH // 2 - 150, HEIGHT // 2 - 60))
        screen.blit(score_text, (WIDTH // 2 - 70, HEIGHT // 2 - 20))
        screen.blit(best_score_text, (WIDTH // 2 - 100, HEIGHT // 2 + 20))
        screen.blit(wait_text, (WIDTH // 2 - 230, HEIGHT // 2 + 60))
        
        pygame.display.flip()
        pygame.time.wait(2000)
    
    # Fin de l'entraînement
    agent.save_model()
    print(f"Entraînement terminé. Meilleur score: {max_score}")
    return


def check_collision(obj, player):
    """
    Vérifier les collisions entre un joueur et un objet
    """
    if isinstance(obj, Obstacle) and player.rect.colliderect(obj.get_rect()):
        return True
    elif isinstance(obj, (DoublePikes, TriplePikes, QuadruplePikes)):
        for rect in obj.get_rects():
            if player.rect.colliderect(rect):
                return True
    elif isinstance(obj, DoubleBlockPillar):
        for rect in obj.get_rects():
            if player.rect.colliderect(rect):
                return True
    elif isinstance(obj, FivePikesWithOrb):
        for i, rect in enumerate(obj.get_rects()):
            if i < 5 and player.rect.colliderect(rect):  # Les 5 premiers rectangles sont les pics
                return True
    elif isinstance(obj, QuintuplePikesWithJumpPad):
        if hasattr(obj, 'get_rects'):
            rects = obj.get_rects()
            for i in range(5, min(10, len(rects))):  # Indices 5 à 9 sont les pics
                if player.rect.colliderect(rects[i]):
                    return True
    elif isinstance(obj, JumppadOrbsObstacle):
        if obj.check_collision(player, [False] * 323):  # Simulation sans touches pressées
            return True
    
    return False


def is_offscreen(obj):
    """
    Vérifier si un objet est sorti de l'écran
    """
    if isinstance(obj, Block):
        return obj.rect.right < 0
    elif hasattr(obj, 'x') and hasattr(obj, 'width'):
        return obj.x + obj.width < 0
    
    return False


def best_ai_play():
    """
    Mode de jeu avec la meilleure IA entraînée
    """
    print("Mode meilleure IA activé")
    
    # Charger le meilleur modèle et jouer avec epsilon = 0 (pas d'exploration)
    agent = GeometryDashAI(state_size=4, action_size=2, load_model=True)
    agent.epsilon = 0  # Mode exploitation seulement
    
    # Lancer le jeu avec l'agent
    # Le code est presque identique à ai_reinforcement_play mais sans l'apprentissage
    
    # Initialisation du jeu (même code que pour ai_reinforcement_play)
    # ... [code initial]
    
    # Différence principale : pas d'apprentissage (pas d'appel à replay ou memorize)
    print("Charger et utiliser le modèle entraîné précédemment sans explorer davantage")
    pygame.time.wait(2000)  # Attente pour montrer le message
    
    # Appeler la fonction d'IA avec apprentissage désactivé
    ai_reinforcement_play()  # Cette fonction a déjà le code pour jouer, nous réutilisons donc celle-ci