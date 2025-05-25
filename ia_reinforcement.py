import numpy as np
import pygame
import random
import pickle
import os
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import threading
import time
from klass import Player
from klass import MovingObject
from klass import Obstacle
from klass import DoublePikes
from klass import TriplePikes   
from klass import QuadruplePikes
from klass import Block
from klass import BouncingObstacle
from klass import FivePikesWithOrb
from klass import JumpPad
from klass import QuintuplePikesWithJumpPad

class GeometryDashAI:
    def __init__(self, state_size=8, action_size=2, load_model=True):
        self.state_size = state_size
        self.action_size = action_size
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.002
        self.epsilon_decay = 0.9997
        self.learning_rate = 0.01
        
        self.memory = deque(maxlen=10000)
        
        self.q_table = {}
        
        self.training_episodes = 0
        self.high_score = 0
        self.last_scores = deque(maxlen=100)
        
        self.model_path = 'geometry_dash_ai_modelv5.pkl'
        if load_model and os.path.exists(self.model_path):
            self.load_model()
    
    def preprocess_state(self, raw_state):
        (distance_to_obstacle, obstacle_type, game_speed, player_y, 
         player_velocity, next_obstacle_distance, next_obstacle_type, obstacle_height) = raw_state
        
        distance_bins = [0, 50, 100, 150, 200, 300, 400, 600, 800, float('inf')]
        distance_idx = next(i for i, val in enumerate(distance_bins) if distance_to_obstacle < val)
        
        player_y_bins = [0, 300, 350, 380, 400, 420, 450, 480, 500, 550, float('inf')]
        player_y_idx = next(i for i, val in enumerate(player_y_bins) if player_y < val)
        
        velocity_bins = [-20, -15, -10, -5, 0, 5, 10, 15, 20, float('inf')]
        velocity_idx = next(i for i, val in enumerate(velocity_bins) if player_velocity < val)
        
        next_distance_bins = [0, 150, 300, 500, 800, 1200, float('inf')]
        next_distance_idx = next(i for i, val in enumerate(next_distance_bins) if next_obstacle_distance < val)
        
        height_bins = [0, 50, 100, 150, 200, float('inf')]
        height_idx = next(i for i, val in enumerate(height_bins) if obstacle_height < val)
        
        return (distance_idx, obstacle_type, game_speed, player_y_idx, velocity_idx, next_distance_idx, next_obstacle_type, height_idx)
    
    def get_action(self, state):
        processed_state = self.preprocess_state(state)
        
        if np.random.rand() <= self.epsilon:
            if processed_state[0] <= 2 and processed_state[1] != 0:
                return 1 if np.random.rand() < 0.7 else 0
            else:
                return random.randrange(self.action_size)
        
        if processed_state not in self.q_table:
            self.q_table[processed_state] = np.zeros(self.action_size)
            if processed_state[0] <= 2 and processed_state[1] != 0:
                self.q_table[processed_state][1] = 0.1
        
        return np.argmax(self.q_table[processed_state])
    
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            processed_state = self.preprocess_state(state)
            processed_next_state = self.preprocess_state(next_state)
            
            if processed_state not in self.q_table:
                self.q_table[processed_state] = np.zeros(self.action_size)
            
            if processed_next_state not in self.q_table:
                self.q_table[processed_next_state] = np.zeros(self.action_size)
            
            target = reward
            if not done:
                target += self.gamma * np.max(self.q_table[processed_next_state])
            
            current_q = self.q_table[processed_state][action]
            self.q_table[processed_state][action] += self.learning_rate * (target - current_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def calculate_reward(self, player_alive, distance_to_obstacle, obstacle_passed, player_y, ground_height, obstacle_height=0, is_jumping=False, nearest_obstacle_distance=float('inf'), jumppad_touched=False, obstacle_type=0, next_obstacle_type=0, contact_with_block=False):
        if not player_alive:
            return -10000  # Grosse récompense négative pour la mort

        reward = 0

        # Petite récompense pour le défilement automatique (survie)
        reward += 0.0005

        # Grosse récompense pour les obstacles franchis (sauf JumpPad)
        if obstacle_passed and obstacle_type != 9:  # 9 est le type pour JumpPad
            reward += 0.15

        # Récompense importante pour avoir touché un jumppad
        if jumppad_touched:
            reward += 0.4  # Récompense significative pour encourager l'utilisation des JumpPad

        # Pénalité/récompense pour les sauts inutiles/appropriés
        if is_jumping and distance_to_obstacle > 200:
            reward -= 0.1
        else:
            reward += 0.1  # Récompense pour ne pas sauter inutilement

        # Récompense pour avoir la bonne hauteur face à un obstacle élevé
        if obstacle_height > 100 and player_y < ground_height - 100:
            reward += 0.01

        return reward
    
    def save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'high_score': self.high_score,
                'training_episodes': self.training_episodes
            }, f)
        print(f"Modèle sauvegardé dans {self.model_path}")
    
    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    self.q_table = data['q_table']
                    self.epsilon = data.get('epsilon', 0.1)
                    self.high_score = data.get('high_score', 0)
                    self.training_episodes = data.get('training_episodes', 0)
                else:
                    self.q_table = data
                    self.epsilon = 0.1
            print(f"Modèle chargé depuis {self.model_path} avec {len(self.q_table)} états et epsilon={self.epsilon}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            print("Création d'un nouveau modèle.")

def get_obstacle_data(obj):
    obj_x = 0
    obj_width = 0
    obj_height = 0
    obstacle_type = 0
    
    if isinstance(obj, Obstacle):
        obj_x = obj.x
        obj_width = obj.width
        obj_height = obj.height if hasattr(obj, 'height') else 50
        obstacle_type = 1
    elif isinstance(obj, Block):
        obj_x = obj.rect.left
        obj_width = obj.rect.width
        obj_height = obj.rect.height
        obstacle_type = 2
    elif isinstance(obj, DoublePikes):
        obj_x = obj.x
        obj_width = obj.width
        obj_height = 100
        obstacle_type = 3
    elif isinstance(obj, TriplePikes):
        obj_x = obj.x
        obj_width = obj.width
        obj_height = 150
        obstacle_type = 4
    elif isinstance(obj, QuadruplePikes):
        obj_x = obj.x
        obj_width = obj.width
        obj_height = 150
        obstacle_type = 5
    elif isinstance(obj, BouncingObstacle):
        obj_x = obj.x
        obj_width = obj.width
        obj_height = 100
        obstacle_type = 6
    elif isinstance(obj, JumpPad):
        obj_x = obj.x
        obj_width = obj.width
        obj_height = 50
        obstacle_type = 9
    elif isinstance(obj, QuintuplePikesWithJumpPad):
        obj_x = obj.x
        obj_width = obj.width
        obj_height = 150
        obstacle_type = 10
    elif isinstance(obj, FivePikesWithOrb):
        obj_x = obj.x
        obj_width = obj.width
        obj_height = 150
        obstacle_type = 11
    
    return obj_x, obj_width, obj_height, obstacle_type


def find_obstacles(game_objects, player_x):
    obstacles_data = []
    
    for obj in game_objects:
        obj_x, obj_width, obj_height, obj_type = get_obstacle_data(obj)
        
        if obj_x > player_x:
            distance = obj_x - player_x
            obstacles_data.append((distance, obj_x, obj_width, obj_height, obj_type))
    
    obstacles_data.sort(key=lambda x: x[0])
    
    if not obstacles_data:
        return 800, 0, 0, 0, 1600, 0, 0
    
    current_distance, current_x, current_width, current_height, current_type = obstacles_data[0]
    
    if len(obstacles_data) > 1:
        next_distance, next_x, next_width, next_height, next_type = obstacles_data[1]
    else:
        next_distance, next_x, next_width, next_height, next_type = 1600, 0, 0, 0, 0
    
    return current_distance, current_type, current_height, current_width, next_distance, next_type, next_height


def check_collision(obj, player):
    if isinstance(obj, Obstacle) and player.rect.colliderect(obj.get_rect()):
        return True
    elif isinstance(obj, Block) and player.rect.colliderect(obj.rect):
        return True
    elif isinstance(obj, (DoublePikes, TriplePikes, QuadruplePikes)):
        for rect in obj.get_rects():
            if player.rect.colliderect(rect):
                return True
    elif isinstance(obj, FivePikesWithOrb):
        for i, rect in enumerate(obj.get_rects()):
            if i < 5 and player.rect.colliderect(rect):
                return True
    elif isinstance(obj, QuintuplePikesWithJumpPad):
        if hasattr(obj, 'get_rects'):
            rects = obj.get_rects()
            for i in range(5, min(10, len(rects))):
                if player.rect.colliderect(rects[i]):
                    return True
    
    return False


def is_offscreen(obj):
    if isinstance(obj, Block):
        return obj.rect.right < 0
    elif hasattr(obj, 'x') and hasattr(obj, 'width'):
        return obj.x + obj.width < 0
    
    return False


def ai_reinforcement_play():
    print("Mode IA par renforcement activé")

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
    GREEN = (0, 255, 0)
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Geometry Dash Clone - IA Renforcement")
    clock = pygame.time.Clock()
    
    agent = GeometryDashAI(state_size=8, action_size=2, load_model=True)
    agent.training_episodes += 1
    
    batch_size = 64
    scores = []
    episodes = 1500
    
    min_obstacle_distances = {
        6: 150, 7: 175, 8: 500, 9: 250, 10: 275, 11: 300
    }
    
    obstacle_intervals = {
        6: [800, 1400], 7: [900, 1600], 8: [1200, 1800],
        9: [1300, 2000], 10: [1400, 2100], 11: [1500, 2200]
    }
    
    max_score = agent.high_score
    current_episode = agent.training_episodes
    avg_score = 0
    
    training_info_font = pygame.font.SysFont(None, 24)
    
    for e in range(episodes):
        print(f"Début de l'épisode {current_episode}/{agent.training_episodes + episodes}")
        
        player = Player()
        game_objects = []
        score = 0
        last_object = pygame.time.get_ticks()
        obstacles_passed = 0
        
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
        
        state = [WIDTH, 0, current_speed, player.y, 0, WIDTH*2, 0, 0]
        
        total_reward = 0
        space_pressed = False
        ai_action = 0
        
        running = True
        episode_time_start = pygame.time.get_ticks()
        frames_survived = 0
        
        while running:
            current_time = pygame.time.get_ticks()
            frames_survived += 1
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    agent.save_model()
                    pygame.quit()
                    import sys
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        agent.save_model()
                        running = False
            
            current_distance, obstacle_type, obstacle_height, obstacle_width, next_distance, next_obstacle_type, next_height = find_obstacles(game_objects, player.x)
            
            next_state = [
                current_distance,
                obstacle_type,
                current_speed,
                player.y,
                player.velocity_y,
                next_distance,
                next_obstacle_type,
                obstacle_height
            ]
            
            ai_action = agent.get_action(next_state)
            
            if ai_action == 1:
                if not space_pressed or player.just_landed:
                    if not player.is_jumping:
                        player.jump()
                space_pressed = True
            else:
                space_pressed = False
            
            can_spawn_obstacle = True
            min_distance = min_obstacle_distances[current_speed]
            
            if game_objects:
                last_obstacle = game_objects[-1]
                
                if isinstance(last_obstacle, Obstacle):
                    last_obstacle_right = last_obstacle.x + last_obstacle.width
                elif isinstance(last_obstacle, Block):
                    last_obstacle_right = last_obstacle.rect.right
                else:
                    last_obstacle_right = last_obstacle.x + last_obstacle.width
                
                if WIDTH - last_obstacle_right < min_distance:
                    can_spawn_obstacle = False
            
            if can_spawn_obstacle and current_time - last_object > object_interval:
                obj = None
                
                if current_speed == 6:
                    choice = random.random()
                    if choice < 0.35:
                        obj = Obstacle(WIDTH)
                    elif choice < 0.6:
                        obj = JumpPad(WIDTH)
                    else:
                        obj = Block(WIDTH)
                elif current_speed == 7:
                    choice = random.random()
                    if choice < 0.2:
                        obj = Obstacle(WIDTH)
                    elif choice < 0.4:
                        obj = JumpPad(WIDTH)
                    elif choice < 0.6:
                        obj = QuintuplePikesWithJumpPad(WIDTH)
                    elif choice < 0.8:
                        obj = DoublePikes(WIDTH)
                    else:
                        obj = Block(WIDTH)
                elif current_speed == 8:
                    choice = random.random()
                    if choice < 0.2:
                        obj = Obstacle(WIDTH)
                    elif choice < 0.5:
                        obj = DoublePikes(WIDTH)
                    elif choice < 0.75:
                        obj = Block(WIDTH)
                    else:
                        obj = BouncingObstacle(WIDTH)
                elif current_speed >= 9:
                    choice = random.random()
                    if choice < 0.15:
                        obj = Obstacle(WIDTH)
                    elif choice < 0.3:
                        obj = Block(WIDTH)
                    elif choice < 0.6:
                        obj = DoublePikes(WIDTH)
                    elif choice < 0.85:
                        obj = TriplePikes(WIDTH)
                    else:
                        obj = QuadruplePikes(WIDTH)
                
                if obj:
                    obj.set_speed(current_speed)
                    game_objects.append(obj)
                    
                    last_object = current_time
                    object_interval = random.randint(*obstacle_intervals[current_speed])
            
            player.update(game_objects)
            
            if not player.is_alive:
                print(f"Épisode {current_episode}, Score: {score}, Epsilon: {agent.epsilon:.4f}")
                running = False
            
            objects_to_remove = []
            obstacle_passed = False
            
            for obj in game_objects[:]:
                obj.update()
                
                # Gestion spécifique pour le JumpPad
                if isinstance(obj, JumpPad):
                    if hasattr(obj, 'get_rect') and callable(getattr(obj, 'get_rect')):
                        pad_rect = obj.get_rect()
                        if (player.rect.bottom >= pad_rect.top and 
                            player.rect.right > pad_rect.left and 
                            player.rect.left < pad_rect.right):
                            if hasattr(obj, 'activate') and callable(getattr(obj, 'activate')):
                                obj.activate(player)
                
                # Gestion spécifique pour QuintuplePikesWithJumpPad
                elif isinstance(obj, QuintuplePikesWithJumpPad):
                    if hasattr(obj, 'get_rects') and callable(getattr(obj, 'get_rects')):
                        rects = obj.get_rects()
                        if len(rects) > 0:
                            jumppad_rect = rects[-1]
                            if player.rect.colliderect(jumppad_rect):
                                if hasattr(obj, 'activate_jump_pad') and callable(getattr(obj, 'activate_jump_pad')):
                                    obj.activate_jump_pad(player)
                        
                        for i in range(5, min(10, len(rects))):
                            if player.rect.colliderect(rects[i]):
                                player.is_alive = False
                                print("Game Over! Collision avec un pic quintuple")
                                running = False
                                break
                
                else:
                    if check_collision(obj, player):
                        player.is_alive = False
                        running = False
                        break
                
                if is_offscreen(obj):
                    objects_to_remove.append(obj)
                    obstacle_passed = True
            
            for obj in objects_to_remove:
                if obj in game_objects:
                    game_objects.remove(obj)
                    score += 1
                    obstacles_passed += 1
                    
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
                    
                    if current_speed != previous_speed:
                        if hasattr(player, 'change_skin_randomly'):
                            player.change_skin_randomly()
                        previous_speed = current_speed
            
            reward = agent.calculate_reward(
                player.is_alive,
                current_distance,
                obstacle_passed,
                player.y,
                GROUND_HEIGHT,
                obstacle_height,
                player.is_jumping,  # Ajout du paramètre indiquant si le joueur est en train de sauter
                current_distance    # Utiliser current_distance comme nearest_obstacle_distance
            )
            
            is_done = not player.is_alive or not running
            agent.memorize(state, ai_action, reward, next_state, is_done)
            state = next_state
            total_reward += reward
            
            if frames_survived % 5 == 0:
                agent.replay(batch_size)
            
            screen.fill(WHITE)
            pygame.draw.rect(screen, BLACK, (0, GROUND_HEIGHT, WIDTH, HEIGHT - GROUND_HEIGHT))
            
            player.draw(screen)
            
            for obj in game_objects:
                obj.draw(screen)
            
            font = pygame.font.SysFont(None, 36)
            score_text = font.render(f"Score: {score}", True, BLACK)
            screen.blit(score_text, (20, 20))
            
            speed_text = font.render(f"Vitesse: {current_speed}", True, BLACK)
            screen.blit(speed_text, (20, 60))
            
            info_y = 100
            episode_text = training_info_font.render(f"Episode: {current_episode}/{episodes}", True, BLACK)
            screen.blit(episode_text, (20, info_y))
            
            epsilon_text = training_info_font.render(f"Epsilon: {agent.epsilon:.4f}", True, BLACK)
            screen.blit(epsilon_text, (20, info_y + 25))
            
            action_text = training_info_font.render(f"Action: {'JUMP' if ai_action == 1 else 'WAIT'}", True, RED if ai_action == 1 else BLACK)
            screen.blit(action_text, (20, info_y + 50))
            
            reward_text = training_info_font.render(f"Récompense: {total_reward:.1f}", True, BLACK)
            screen.blit(reward_text, (20, info_y + 75))
            
            max_score_text = training_info_font.render(f"Meilleur score: {max_score}", True, BLACK)
            screen.blit(max_score_text, (20, info_y + 100))
            
            avg_score_text = training_info_font.render(f"Score moyen: {avg_score:.1f}", True, BLACK)
            screen.blit(avg_score_text, (20, info_y + 125))
            
            dist_text = training_info_font.render(f"Distance: {int(current_distance)}", True, BLACK)
            screen.blit(dist_text, (20, info_y + 150))
            
            states_text = training_info_font.render(f"États connus: {len(agent.q_table)}", True, BLACK)
            screen.blit(states_text, (20, info_y + 175))
            
            pygame.display.flip()
            clock.tick(FPS)
        
        episode_duration = (pygame.time.get_ticks() - episode_time_start) / 1000
        
        print(f"Épisode {current_episode} terminé en {episode_duration:.2f}s. Score: {score}, Récompense: {total_reward:.1f}")
        
        agent.last_scores.append(score)
        if score > max_score:
            max_score = score
            agent.high_score = max_score
            print(f"Nouveau meilleur score! {max_score}")
        
        avg_score = sum(agent.last_scores) / len(agent.last_scores)
        
        for _ in range(5):
            agent.replay(batch_size * 2)
        
        if current_episode % 10 == 0:
            agent.save_model()
            print(f"Modèle sauvegardé à l'épisode {current_episode}")
            
        current_episode += 1
        agent.training_episodes += 1
    
    agent.save_model()
    print(f"Entraînement terminé. Meilleur score: {max_score}")


def best_ai_play():
    print("Mode Meilleure IA activé")
    
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
    GREEN = (0, 255, 0)
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Geometry Dash Clone - Meilleure IA")
    clock = pygame.time.Clock()
    
    agent = GeometryDashAI(state_size=8, action_size=2, load_model=True)
    agent.epsilon = 0
    
    player = Player()
    game_objects = []
    score = 0
    last_object = pygame.time.get_ticks()
    
    min_obstacle_distances = {
        6: 150, 7: 175, 8: 500, 9: 250, 10: 275, 11: 300
    }
    
    obstacle_intervals = {
        6: [800, 1400], 7: [900, 1600], 8: [1200, 1800],
        9: [1300, 2000], 10: [1400, 2100], 11: [1500, 2200]
    }
    
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
    
    state = [WIDTH, 0, current_speed, player.y, 0, WIDTH*2, 0, 0]
    
    total_reward = 0
    space_pressed = False
    ai_action = 0
    
    running = True
    start_time = pygame.time.get_ticks()
    
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        current_distance, obstacle_type, obstacle_height, obstacle_width, next_distance, next_obstacle_type, next_height = find_obstacles(game_objects, player.x)
        
        state = [
            current_distance,
            obstacle_type,
            current_speed,
            player.y,
            player.velocity_y,
            next_distance,
            next_obstacle_type,
            obstacle_height
        ]
        
        ai_action = agent.get_action(state)
        
        if ai_action == 1:
            if not space_pressed or player.just_landed:
                if not player.is_jumping:
                    player.jump()
            space_pressed = True
        else:
            space_pressed = False
        
        can_spawn_obstacle = True
        min_distance = min_obstacle_distances[current_speed]
        
        if game_objects:
            last_obstacle = game_objects[-1]
            
            if isinstance(last_obstacle, Block):
                last_obstacle_right = last_obstacle.rect.right
            elif hasattr(last_obstacle, 'x') and hasattr(last_obstacle, 'width'):
                last_obstacle_right = last_obstacle.x + last_obstacle.width
            else:
                last_obstacle_right = 0
            
            if WIDTH - last_obstacle_right < min_distance:
                can_spawn_obstacle = False
        
        if can_spawn_obstacle and current_time - last_object > object_interval:
            obj = None
            
        if current_speed == 6:
            choice = random.random()
            if choice < 0.35:
                    obj = Obstacle(WIDTH)
            elif choice < 0.6:
                    obj = JumpPad(WIDTH)
            else:
                    obj = Block(WIDTH)
        elif current_speed == 7:
                choice = random.random()
                if choice < 0.2:
                    obj = Obstacle(WIDTH)
                elif choice < 0.4:
                    obj = JumpPad(WIDTH)
                elif choice < 0.6:
                    obj = QuintuplePikesWithJumpPad(WIDTH)
                elif choice < 0.8:
                    obj = DoublePikes(WIDTH)
                else:
                    obj = Block(WIDTH)
        elif current_speed == 8:
                choice = random.random()
                if choice < 0.2:
                    obj = Obstacle(WIDTH)
                elif choice < 0.5:
                    obj = DoublePikes(WIDTH)
                elif choice < 0.75:
                    obj = Block(WIDTH)
                else:
                    obj = BouncingObstacle(WIDTH)
        elif current_speed >= 9:
                choice = random.random()
                if choice < 0.15:
                    obj = Obstacle(WIDTH)
                elif choice < 0.3:
                    obj = Block(WIDTH)
                elif choice < 0.6:
                    obj = DoublePikes(WIDTH)
                elif choice < 0.85:
                    obj = TriplePikes(WIDTH)
                else:
                    obj = QuadruplePikes(WIDTH)
            
        if obj:
                obj.set_speed(current_speed)
                game_objects.append(obj)
                
                last_object = current_time
                object_interval = random.randint(*obstacle_intervals[current_speed])
        
        player.update(game_objects)
        
        if not player.is_alive:
            print(f"Game Over! Score final: {score}")
            running = False
        
        objects_to_remove = []
        
        for obj in game_objects[:]:
            obj.update()
            
            if check_collision(obj, player):
                player.is_alive = False
                running = False
                break
            
            if is_offscreen(obj):
                objects_to_remove.append(obj)
        
        for obj in objects_to_remove:
            if obj in game_objects:
                game_objects.remove(obj)
                score += 1
                
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
                
                if current_speed != previous_speed:
                    if hasattr(player, 'change_skin_randomly'):
                        player.change_skin_randomly()
                    previous_speed = current_speed
        
        screen.fill(WHITE)
        pygame.draw.rect(screen, BLACK, (0, GROUND_HEIGHT, WIDTH, HEIGHT - GROUND_HEIGHT))
        
        player.draw(screen)
        
        for obj in game_objects:
            obj.draw(screen)
        
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (20, 20))
        
        speed_text = font.render(f"Vitesse: {current_speed}", True, BLACK)
        screen.blit(speed_text, (20, 60))
        
        elapsed_time = (pygame.time.get_ticks() - start_time) / 1000  # en secondes
        time_text = font.render(f"Temps: {elapsed_time:.1f}s", True, BLACK)
        screen.blit(time_text, (20, 100))
        
        ai_info_font = pygame.font.SysFont(None, 24)
        action_text = ai_info_font.render(f"Action IA: {'SAUTER' if ai_action == 1 else 'ATTENDRE'}", True, RED if ai_action == 1 else BLACK)
        screen.blit(action_text, (20, 140))
        
        distance_text = ai_info_font.render(f"Distance obstacle: {int(current_distance)}", True, BLACK)
        screen.blit(distance_text, (20, 165))
        
        obstacle_text = ai_info_font.render(f"Type obstacle: {obstacle_type}", True, BLACK)
        screen.blit(obstacle_text, (20, 190))
        
        states_known_text = ai_info_font.render(f"États connus: {len(agent.q_table)}", True, BLACK)
        screen.blit(states_known_text, (20, 215))
        
        if score >= agent.high_score:
            high_score_text = ai_info_font.render(f"NOUVEAU RECORD! +{score - agent.high_score}", True, GREEN)
            screen.blit(high_score_text, (WIDTH - 250, 20))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    if score > agent.high_score:
        agent.high_score = score
        agent.save_model()
        print(f"Nouveau meilleur score sauvegardé: {score}")


if __name__ == "__main__":
    ai_reinforcement_play()
