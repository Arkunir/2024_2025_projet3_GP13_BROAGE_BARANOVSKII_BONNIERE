import pygame
import random
import numpy as np
import pickle
import os
from copy import deepcopy
from klass import Player, Obstacle, Block, DoublePikes, TriplePikes, QuadruplePikes
from klass import BouncingObstacle, DoubleBlockPillar, BlockGapBlockWithSpike
from klass import FivePikesWithOrb, JumpPad, QuintuplePikesWithJumpPad, PurpleOrb, JumppadOrbsObstacle

# Paramètres constants
WIDTH, HEIGHT = 800, 600
FPS = 60
GROUND_HEIGHT = 500
INITIAL_SCROLL_SPEED = 6
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Paramètres pour le Q-learning
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EXPLORATION_RATE = 0.3
EXPLORATION_DECAY = 0.9999
MIN_EXPLORATION_RATE = 0.01
TRAINING_EPISODES = 1000

# Chemin pour sauvegarder/charger le modèle Q-learning
MODEL_PATH = "q_learning_model.pkl"

class GameState:
    """Représentation discrétisée de l'état du jeu pour Q-learning"""
    def __init__(self):
        self.distance_to_next_obstacle = 0  # Distance au prochain obstacle
        self.obstacle_type = 0              # Type d'obstacle (codé)
        self.obstacle_height = 0            # Hauteur de l'obstacle
        self.obstacle_width = 0             # Largeur de l'obstacle
        self.player_height = 0              # Hauteur du joueur
        self.player_y_velocity = 0          # Vitesse verticale du joueur
        self.player_is_jumping = False      # Si le joueur est en train de sauter
        self.game_speed = 0                 # Vitesse du jeu
    
    def to_tuple(self):
        """Convertit l'état en tuple pour indexer la table Q"""
        # Discrétisation des valeurs continues
        distance_bin = self._discretize_distance(self.distance_to_next_obstacle)
        height_bin = int(self.player_height / 50)  # Hauteur normalisée
        velocity_bin = int((self.player_y_velocity + 20) / 5)  # Vitesse normalisée
        
        # Limiter les valeurs pour éviter les dépassements
        velocity_bin = max(0, min(velocity_bin, 7))
        height_bin = max(0, min(height_bin, 9))
        
        return (
            distance_bin,
            self.obstacle_type,
            int(self.obstacle_height / 50),
            int(self.obstacle_width / 50),
            height_bin,
            velocity_bin,
            int(self.player_is_jumping),
            int(self.game_speed)
        )
    
    def _discretize_distance(self, distance):
        """Discrétise la distance en intervalles"""
        if distance <= 0:
            return 0
        elif distance < 50:
            return 1
        elif distance < 100:
            return 2
        elif distance < 200:
            return 3
        elif distance < 300:
            return 4
        elif distance < 400:
            return 5
        else:
            return 6


class QAgent:
    """Agent utilisant Q-learning pour apprendre à jouer au jeu"""
    def __init__(self):
        self.q_table = {}  # Table Q: état -> action -> valeur
        self.exploration_rate = EXPLORATION_RATE
        
    def get_action(self, state, training=True):
        """Détermine l'action à effectuer (sauter ou ne pas sauter)"""
        state_tuple = state.to_tuple()
        
        # Si l'état n'est pas dans la table Q, l'initialiser
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {0: 0.0, 1: 0.0}  # 0: ne pas sauter, 1: sauter
        
        # Exploration vs exploitation
        if training and random.random() < self.exploration_rate:
            # Exploration: action aléatoire
            return random.choice([0, 1])
        else:
            # Exploitation: meilleure action connue
            values = self.q_table[state_tuple]
            return max(values, key=values.get)
    
    def update_q_value(self, state, action, reward, next_state):
        """Met à jour la valeur Q pour un état-action"""
        state_tuple = state.to_tuple()
        next_state_tuple = next_state.to_tuple()
        
        # Initialiser si l'état n'est pas dans la table Q
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {0: 0.0, 1: 0.0}
        if next_state_tuple not in self.q_table:
            self.q_table[next_state_tuple] = {0: 0.0, 1: 0.0}
        
        # Formule de mise à jour Q
        current_q = self.q_table[state_tuple][action]
        max_next_q = max(self.q_table[next_state_tuple].values())
        
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
        self.q_table[state_tuple][action] = new_q
    
    def decay_exploration(self):
        """Diminue le taux d'exploration"""
        self.exploration_rate = max(MIN_EXPLORATION_RATE, 
                                    self.exploration_rate * EXPLORATION_DECAY)
    
    def save_model(self, path=MODEL_PATH):
        """Sauvegarde le modèle (table Q)"""
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Modèle sauvegardé avec {len(self.q_table)} états")
    
    def load_model(self, path=MODEL_PATH):
        """Charge un modèle (table Q) existant"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Modèle chargé avec {len(self.q_table)} états")
            return True
        return False


def get_game_state(player, game_objects, current_speed):
    """Extrait l'état du jeu pour l'agent RL"""
    state = GameState()
    
    # Trouver le prochain obstacle
    next_obstacle = None
    min_distance = float('inf')
    
    for obj in game_objects:
        # Déterminer la position x de l'objet
        obj_x = 0
        if hasattr(obj, 'x'):
            obj_x = obj.x
        elif hasattr(obj, 'rect'):
            obj_x = obj.rect.x
        
        # Vérifier si l'objet est devant le joueur
        if obj_x > player.rect.right:
            distance = obj_x - player.rect.right
            if distance < min_distance:
                min_distance = distance
                next_obstacle = obj
    
    # Si aucun obstacle n'est trouvé, retourner un état par défaut
    if next_obstacle is None:
        state.distance_to_next_obstacle = WIDTH
        state.obstacle_type = 0
        state.obstacle_height = 0
        state.obstacle_width = 0
    else:
        # Déterminer le type d'obstacle
        if isinstance(next_obstacle, Block):
            state.obstacle_type = 1
            state.obstacle_height = next_obstacle.rect.height
            state.obstacle_width = next_obstacle.rect.width
        elif isinstance(next_obstacle, Obstacle):
            state.obstacle_type = 2
            state.obstacle_height = next_obstacle.height
            state.obstacle_width = next_obstacle.width
        elif isinstance(next_obstacle, DoublePikes):
            state.obstacle_type = 3
            state.obstacle_height = 50  # Hauteur approximative
            state.obstacle_width = next_obstacle.width
        elif isinstance(next_obstacle, TriplePikes):
            state.obstacle_type = 4
            state.obstacle_height = 50
            state.obstacle_width = next_obstacle.width
        elif isinstance(next_obstacle, JumpPad):
            state.obstacle_type = 5
            state.obstacle_height = 20  # Hauteur approximative
            state.obstacle_width = next_obstacle.width
        elif isinstance(next_obstacle, QuintuplePikesWithJumpPad):
            state.obstacle_type = 6
            state.obstacle_height = 50
            state.obstacle_width = next_obstacle.width
        elif isinstance(next_obstacle, PurpleOrb):
            state.obstacle_type = 7
            state.obstacle_height = 30  # Hauteur approximative
            state.obstacle_width = next_obstacle.width
        elif isinstance(next_obstacle, JumppadOrbsObstacle):
            state.obstacle_type = 8
            state.obstacle_height = 50
            state.obstacle_width = next_obstacle.width
        elif isinstance(next_obstacle, BlockGapBlockWithSpike):
            state.obstacle_type = 9
            state.obstacle_height = 50
            state.obstacle_width = next_obstacle.width
        elif isinstance(next_obstacle, DoubleBlockPillar):
            state.obstacle_type = 10
            state.obstacle_height = 100  # Hauteur approximative
            state.obstacle_width = next_obstacle.width
        elif isinstance(next_obstacle, FivePikesWithOrb):
            state.obstacle_type = 11
            state.obstacle_height = 50
            state.obstacle_width = next_obstacle.width
        else:
            state.obstacle_type = 12  # Type inconnu
            state.obstacle_height = 50
            state.obstacle_width = 50
        
        # Déterminer la distance au prochain obstacle
        if hasattr(next_obstacle, 'x'):
            state.distance_to_next_obstacle = next_obstacle.x - player.rect.right
        elif hasattr(next_obstacle, 'rect'):
            state.distance_to_next_obstacle = next_obstacle.rect.left - player.rect.right
    
    # Informations sur le joueur
    state.player_height = GROUND_HEIGHT - player.rect.bottom
    state.player_y_velocity = player.y_velocity
    state.player_is_jumping = player.is_jumping
    state.game_speed = current_speed
    
    return state


def train_agent(agent, episodes=TRAINING_EPISODES):
    """Entraîne l'agent sur plusieurs épisodes"""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Formation IA Q-learning")
    clock = pygame.time.Clock()
    
    best_score = 0
    
    for episode in range(episodes):
        # Initialisation de l'épisode
        player = Player()
        game_objects = []
        score = 0
        last_object = pygame.time.get_ticks()
        current_speed = INITIAL_SCROLL_SPEED
        
        # Paramètres pour la génération des obstacles
        min_obstacle_distances = {
            6: 150, 7: 175, 8: 200, 9: 250, 10: 275, 11: 300
        }
        
        obstacle_intervals = {
            6: [800, 1400], 7: [900, 1600], 8: [1200, 1800], 
            9: [1300, 2000], 10: [1400, 2100], 11: [1500, 2200]
        }
        
        object_interval = random.randint(*obstacle_intervals[INITIAL_SCROLL_SPEED])
        
        # Seuils de changement de vitesse
        speed_threshold_7 = random.randint(10, 20)
        speed_threshold_8 = random.randint(max(25, 2 * speed_threshold_7 - 5), 
                                          2 * speed_threshold_7 + 10)
        speed_threshold_9 = random.randint(max(40, 2 * speed_threshold_8 - 15), 
                                          2 * speed_threshold_8 + 5)
        speed_threshold_random = 100
        next_random_change = speed_threshold_random + random.randint(25, 50)
        
        # Variables pour le Q-learning
        current_state = GameState()
        running = True
        
        # Pour afficher la progression
        if episode % 10 == 0:
            print(f"Épisode {episode}/{episodes}, Meilleur score: {best_score}, "
                  f"Exploration: {agent.exploration_rate:.4f}")
        
        while running:
            current_time = pygame.time.get_ticks()
            
            # Gestion des événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Obtenir l'état actuel du jeu
            current_state = get_game_state(player, game_objects, current_speed)
            
            # Décider de l'action à effectuer
            action = agent.get_action(current_state, training=True)
            
            # Appliquer l'action
            if action == 1 and not player.is_jumping:  # Sauter
                player.jump()
            
            # Générer un nouvel obstacle si nécessaire
            can_spawn_obstacle = True
            min_distance = min_obstacle_distances[current_speed]
            
            if game_objects:
                last_obstacle = game_objects[-1]
                
                # Calculer la position de fin du dernier obstacle
                if hasattr(last_obstacle, 'x') and hasattr(last_obstacle, 'width'):
                    last_obstacle_right = last_obstacle.x + last_obstacle.width
                elif hasattr(last_obstacle, 'rect'):
                    last_obstacle_right = last_obstacle.rect.right
                else:
                    last_obstacle_right = 0
                
                # Vérifier si l'espace est suffisant
                if WIDTH - last_obstacle_right < min_distance:
                    can_spawn_obstacle = False
            
            # Créer un nouvel obstacle si possible
            if can_spawn_obstacle and current_time - last_object > object_interval:
                obj = None
                
                # Logique de sélection d'obstacle en fonction de la vitesse (copié du jeu original)
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
                else:  # Vitesse 9+
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
                
                if obj:
                    obj.set_speed(current_speed)
                    game_objects.append(obj)
                    last_object = current_time
                    object_interval = random.randint(*obstacle_intervals[current_speed])
            
            # Mettre à jour le joueur
            player.update(game_objects)
            
            # Vérifier si le jeu est terminé
            if not player.is_alive:
                running = False
            
            # Mettre à jour les objets
            objects_to_remove = []
            
            for obj in game_objects[:]:
                obj.update()
                
                # Vérifier les collisions spécifiques
                if isinstance(obj, JumppadOrbsObstacle):
                    if obj.check_collision(player, {}):  # Pas de touches pressées
                        player.is_alive = False
                        running = False
                        break
                
                # Vérifier si l'objet est hors de l'écran
                if ((hasattr(obj, 'x') and hasattr(obj, 'width') and obj.x + obj.width < 0) or
                    (hasattr(obj, 'rect') and obj.rect.right < 0)):
                    objects_to_remove.append(obj)
            
            # Supprimer les objets hors écran
            for obj in objects_to_remove:
                if obj in game_objects:
                    game_objects.remove(obj)
                    score += 1
                    
                    # Gestion des changements de vitesse
                    if score == speed_threshold_7 and current_speed < 7:
                        current_speed = 7
                        for game_obj in game_objects:
                            game_obj.set_speed(current_speed)
                    elif score == speed_threshold_8 and current_speed < 8:
                        current_speed = 8
                        for game_obj in game_objects:
                            game_obj.set_speed(current_speed)
                    elif score == speed_threshold_9 and current_speed < 9:
                        current_speed = 9
                        for game_obj in game_objects:
                            game_obj.set_speed(current_speed)
                    elif score == speed_threshold_random:
                        current_speed = random.randint(9, 11)
                        for game_obj in game_objects:
                            game_obj.set_speed(current_speed)
                        next_random_change = score + random.randint(25, 50)
                    elif score >= speed_threshold_random and score == next_random_change:
                        current_speed = random.randint(9, 11)
                        for game_obj in game_objects:
                            game_obj.set_speed(current_speed)
                        next_random_change = score + random.randint(25, 50)
            
            # Obtenir le nouvel état
            next_state = get_game_state(player, game_objects, current_speed)
            
            # Calculer la récompense
            reward = 0.1  # Petite récompense pour survivre
            
            # Récompense supplémentaire pour les obstacles évités
            for obj in objects_to_remove:
                reward += 1.0
            
            # Pénalité pour la collision/mort
            if not player.is_alive:
                reward = -10.0
            
            # Mettre à jour la table Q
            agent.update_q_value(current_state, action, reward, next_state)
            
            if episode % 10 == 0:  # Afficher seulement certains épisodes pour accélérer l'entraînement
                # Afficher le jeu
                screen.fill(WHITE)
                pygame.draw.rect(screen, BLACK, (0, GROUND_HEIGHT, WIDTH, HEIGHT - GROUND_HEIGHT))
                player.draw(screen)
                for obj in game_objects:
                    obj.draw(screen)
                
                # Afficher les informations
                font = pygame.font.SysFont(None, 36)
                score_text = font.render(f"Score: {score}", True, BLACK)
                screen.blit(score_text, (20, 20))
                
                episode_text = font.render(f"Épisode: {episode}/{episodes}", True, BLACK)
                screen.blit(episode_text, (20, 60))
                
                exploration_text = font.render(f"Exploration: {agent.exploration_rate:.4f}", True, BLACK)
                screen.blit(exploration_text, (20, 100))
                
                pygame.display.flip()
            
            clock.tick(FPS)
        
        # Fin de l'épisode
        agent.decay_exploration()
        if score > best_score:
            best_score = score
            # Sauvegarder le modèle à chaque amélioration significative
            if best_score > 20:
                agent.save_model(f"q_learning_model_score_{best_score}.pkl")
        
        # Sauvegarder régulièrement
        if episode % 100 == 0:
            agent.save_model()
    
    # Sauvegarder le modèle final
    agent.save_model()
    print(f"Entraînement terminé. Meilleur score: {best_score}")
    pygame.quit()


def ai_reinforcement_play():
    """Fonction principale pour jouer avec l'IA par renforcement"""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Geometry Dash - IA par Renforcement")
    clock = pygame.time.Clock()
    
    # Créer un agent et charger un modèle existant
    agent = QAgent()
    model_loaded = agent.load_model()
    
    if not model_loaded:
        print("Aucun modèle trouvé. Lancement de l'entraînement...")
        train_agent(agent)
    
    # Démarrer une partie
    player = Player()
    game_objects = []
    score = 0
    last_object = pygame.time.get_ticks()
    current_speed = INITIAL_SCROLL_SPEED
    
    # Paramètres pour la génération des obstacles
    min_obstacle_distances = {
        6: 150, 7: 175, 8: 200, 9: 250, 10: 275, 11: 300
    }
    
    obstacle_intervals = {
        6: [800, 1400], 7: [900, 1600], 8: [1200, 1800], 
        9: [1300, 2000], 10: [1400, 2100], 11: [1500, 2200]
    }
    
    object_interval = random.randint(*obstacle_intervals[INITIAL_SCROLL_SPEED])
    
    # Seuils de changement de vitesse
    speed_threshold_7 = random.randint(10, 20)
    speed_threshold_8 = random.randint(max(25, 2 * speed_threshold_7 - 5), 
                                      2 * speed_threshold_7 + 10)
    speed_threshold_9 = random.randint(max(40, 2 * speed_threshold_8 - 15), 
                                      2 * speed_threshold_8 + 5)
    speed_threshold_random = 100
    next_random_change = speed_threshold_random + random.randint(25, 50)
    
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Obtenir l'état actuel du jeu
        current_state = get_game_state(player, game_objects, current_speed)
        
        # Décider de l'action à effectuer (pas d'exploration en mode jeu)
        action = agent.get_action(current_state, training=False)
        
        # Appliquer l'action
        if action == 1 and not player.is_jumping:  # Sauter
            player.jump()
        
        # Générer un nouvel obstacle si nécessaire
        can_spawn_obstacle = True
        min_distance = min_obstacle_distances[current_speed]
        
        if game_objects:
            last_obstacle = game_objects[-1]
            
            # Calculer la position de fin du dernier obstacle
            if hasattr(last_obstacle, 'x') and hasattr(last_obstacle, 'width'):
                last_obstacle_right = last_obstacle.x + last_obstacle.width
            elif hasattr(last_obstacle, 'rect'):
                last_obstacle_right = last_obstacle.rect.right
            else:
                last_obstacle_right = 0
            
            # Vérifier si l'espace est suffisant
            if WIDTH - last_obstacle_right < min_distance:
                can_spawn_obstacle = False
        
        # Créer un nouvel obstacle si possible
        if can_spawn_obstacle and current_time - last_object > object_interval:
            obj = None
            
            # Logique de sélection d'obstacle identique à celle du jeu original
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
            else:  # Vitesse 9+
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
            
            if obj:
                obj.set_speed(current_speed)
                game_objects.append(obj)
                last_object = current_time
                object_interval = random.randint(*obstacle_intervals[current_speed])
        
        # Mettre à jour le joueur
        player.update(game_objects)
        
        # Vérifier si le jeu est terminé
        if not player.is_alive:
            print(f"Partie terminée! Score: {score}")
            running = False
        
        # Mettre à jour les objets
        objects_to_remove = []
        
        for obj in game_objects[:]:
            obj.update()
            
            # Vérifier les collisions spécifiques
            if isinstance(obj, JumppadOrbsObstacle):
                if obj.check_collision(player, {}):  # Pas de touches pressées
                    player.is_alive = False
                    running = False
                    break
            
            # Vérifier si l'objet est hors de l'écran
            if ((hasattr(obj, 'x') and hasattr(obj, 'width') and obj.x + obj.width < 0) or
                (hasattr(obj, 'rect') and obj.rect.right < 0)):
                objects_to_remove.append(obj)
        
        # Supprimer les objets hors écran
        for obj in objects_to_remove:
            if obj in game_objects:
                game_objects.remove(obj)
                score += 1
                
                # Gestion des changements de vitesse
                if score == speed_threshold_7