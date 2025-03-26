import pygame
import sys
import random

pygame.init()

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
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Geometry Dash Clone")
clock = pygame.time.Clock()

class Player:
    def __init__(self):
        self.rect = pygame.Rect(100, GROUND_HEIGHT - CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        self.velocity_y = 0
        self.is_jumping = False
        self.is_alive = True
        self.standing_on = None
        self.just_landed = False

    def update(self, game_objects):
        was_jumping = self.is_jumping
        
        self.velocity_y += GRAVITY
        
        old_y = self.rect.y
        
        self.rect.y += self.velocity_y
        
        self.standing_on = None
        
        self.just_landed = False
        
        for obj in game_objects:
            if isinstance(obj, Block):
                if (old_y + CUBE_SIZE <= obj.rect.top and 
                    self.rect.y + CUBE_SIZE >= obj.rect.top and
                    self.rect.right > obj.rect.left and
                    self.rect.left < obj.rect.right):
                    
                    self.rect.bottom = obj.rect.top
                    self.velocity_y = 0
                    
                    if self.is_jumping:
                        self.just_landed = True
                    
                    self.is_jumping = False
                    self.standing_on = obj
                    
                elif (self.rect.colliderect(obj.rect) and 
                      not (old_y + CUBE_SIZE <= obj.rect.top)):
                    self.is_alive = False
                    print("Game Over! Collision latérale avec un bloc")
                    
            elif isinstance(obj, BlockGapBlockWithSpike):
                bloc_rects = [obj.get_rects()[0], obj.get_rects()[1]]
                pic_rect = obj.get_rects()[2]
                
                if self.rect.colliderect(pic_rect):
                    self.is_alive = False
                    print("Game Over! Collision avec un pic")
                
                for bloc_rect in bloc_rects:
                    if (old_y + CUBE_SIZE <= bloc_rect.top and 
                        self.rect.y + CUBE_SIZE >= bloc_rect.top and
                        self.rect.right > bloc_rect.left and
                        self.rect.left < bloc_rect.right):
                        
                        self.rect.bottom = bloc_rect.top
                        self.velocity_y = 0
                        
                        if self.is_jumping:
                            self.just_landed = True
                        
                        self.is_jumping = False
                        self.standing_on = obj
                        
                    elif (self.rect.colliderect(bloc_rect) and 
                          not (old_y + CUBE_SIZE <= bloc_rect.top)):
                        self.is_alive = False
                        print("Game Over! Collision latérale avec un bloc de la structure")
        
        if self.rect.y >= GROUND_HEIGHT - CUBE_SIZE and not self.standing_on:
            self.rect.y = GROUND_HEIGHT - CUBE_SIZE
            self.velocity_y = 0
            
            if self.is_jumping:
                self.just_landed = True
            
            self.is_jumping = False
            
    def jump(self):
        if not self.is_jumping:
            self.velocity_y = JUMP_STRENGTH
            self.is_jumping = True
            
    def draw(self):
        pygame.draw.rect(screen, BLUE, self.rect)

class MovingObject:
    def __init__(self, x):
        self.scroll_speed = INITIAL_SCROLL_SPEED
        
    def set_speed(self, speed):
        self.scroll_speed = speed

class Obstacle(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.y = GROUND_HEIGHT - CUBE_SIZE
        self.width = CUBE_SIZE
        self.height = CUBE_SIZE
        
    def update(self):
        self.x -= self.scroll_speed
        
    def draw(self):
        pygame.draw.polygon(screen, RED, [
            (self.x, self.y + self.height),
            (self.x + self.width, self.y + self.height),
            (self.x + self.width/2, self.y)
        ])
        
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

class DoublePikes(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.y = GROUND_HEIGHT - CUBE_SIZE
        self.width = CUBE_SIZE * 2
        self.height = CUBE_SIZE
        
    def update(self):
        self.x -= self.scroll_speed
        
    def draw(self):
        pygame.draw.polygon(screen, RED, [
            (self.x, self.y + self.height),
            (self.x + CUBE_SIZE, self.y + self.height),
            (self.x + CUBE_SIZE/2, self.y)
        ])
        
        pygame.draw.polygon(screen, RED, [
            (self.x + CUBE_SIZE, self.y + self.height),
            (self.x + CUBE_SIZE*2, self.y + self.height),
            (self.x + CUBE_SIZE*1.5, self.y)
        ])
        
    def get_rects(self):
        return [
            pygame.Rect(self.x, self.y, CUBE_SIZE, self.height),
            pygame.Rect(self.x + CUBE_SIZE, self.y, CUBE_SIZE, self.height)
        ]

class TriplePikes(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.y = GROUND_HEIGHT - CUBE_SIZE
        self.width = CUBE_SIZE * 3
        self.height = CUBE_SIZE
        
    def update(self):
        self.x -= self.scroll_speed
        
    def draw(self):
        pygame.draw.polygon(screen, RED, [
            (self.x, self.y + self.height),
            (self.x + CUBE_SIZE, self.y + self.height),
            (self.x + CUBE_SIZE/2, self.y)
        ])
        
        pygame.draw.polygon(screen, RED, [
            (self.x + CUBE_SIZE, self.y + self.height),
            (self.x + CUBE_SIZE*2, self.y + self.height),
            (self.x + CUBE_SIZE*1.5, self.y)
        ])
        
        pygame.draw.polygon(screen, RED, [
            (self.x + CUBE_SIZE*2, self.y + self.height),
            (self.x + CUBE_SIZE*3, self.y + self.height),
            (self.x + CUBE_SIZE*2.5, self.y)
        ])
        
    def get_rects(self):
        return [
            pygame.Rect(self.x, self.y, CUBE_SIZE, self.height),
            pygame.Rect(self.x + CUBE_SIZE, self.y, CUBE_SIZE, self.height),
            pygame.Rect(self.x + CUBE_SIZE*2, self.y, CUBE_SIZE, self.height)
        ]

class QuadruplePikes(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.y = GROUND_HEIGHT - CUBE_SIZE
        self.width = CUBE_SIZE * 4
        self.height = CUBE_SIZE
        
    def update(self):
        self.x -= self.scroll_speed
        
    def draw(self):
        pygame.draw.polygon(screen, RED, [
            (self.x, self.y + self.height),
            (self.x + CUBE_SIZE, self.y + self.height),
            (self.x + CUBE_SIZE/2, self.y)
        ])
        
        pygame.draw.polygon(screen, RED, [
            (self.x + CUBE_SIZE, self.y + self.height),
            (self.x + CUBE_SIZE*2, self.y + self.height),
            (self.x + CUBE_SIZE*1.5, self.y)
        ])
        
        pygame.draw.polygon(screen, RED, [
            (self.x + CUBE_SIZE*2, self.y + self.height),
            (self.x + CUBE_SIZE*3, self.y + self.height),
            (self.x + CUBE_SIZE*2.5, self.y)
        ])
        
        pygame.draw.polygon(screen, RED, [
            (self.x + CUBE_SIZE*3, self.y + self.height),
            (self.x + CUBE_SIZE*4, self.y + self.height),
            (self.x + CUBE_SIZE*3.5, self.y)
        ])
        
    def get_rects(self):
        return [
            pygame.Rect(self.x, self.y, CUBE_SIZE, self.height),
            pygame.Rect(self.x + CUBE_SIZE, self.y, CUBE_SIZE, self.height),
            pygame.Rect(self.x + CUBE_SIZE*2, self.y, CUBE_SIZE, self.height),
            pygame.Rect(self.x + CUBE_SIZE*3, self.y, CUBE_SIZE, self.height)
        ]

class Block(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.rect = pygame.Rect(x, GROUND_HEIGHT - CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        
    def update(self):
        self.rect.x -= self.scroll_speed
        
    def draw(self):
        pygame.draw.rect(screen, BLACK, self.rect)

class BlockGapBlockWithSpike(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.y = GROUND_HEIGHT - CUBE_SIZE
        self.width = CUBE_SIZE * 4
        self.height = CUBE_SIZE
        
    def update(self):
        self.x -= self.scroll_speed
        
    def draw(self):
        pygame.draw.rect(screen, BLACK, pygame.Rect(
            self.x, self.y, CUBE_SIZE, CUBE_SIZE))
        
        pygame.draw.rect(screen, BLACK, pygame.Rect(
            self.x + CUBE_SIZE * 3, self.y, CUBE_SIZE, CUBE_SIZE))
        
        pygame.draw.polygon(screen, RED, [
            (self.x + CUBE_SIZE * 3, self.y),
            (self.x + CUBE_SIZE * 4, self.y),
            (self.x + CUBE_SIZE * 3.5, self.y - CUBE_SIZE)
        ])
        
    def get_rects(self):
        return [
            pygame.Rect(self.x, self.y, CUBE_SIZE, self.height),
            pygame.Rect(self.x + CUBE_SIZE * 3, self.y, CUBE_SIZE, self.height),
            pygame.Rect(self.x + CUBE_SIZE * 3, self.y - CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        ]

class Button:
    def __init__(self, text, x, y, width, height, color, hover_color):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False
        
    def draw(self, surface):
        current_color = self.hover_color if self.is_hovered else self.color
        
        pygame.draw.rect(surface, current_color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        
        font = pygame.font.SysFont(None, 30)
        text_surface = font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
        
    def update(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        
    def check_clicked(self, mouse_pos, mouse_clicked):
        return self.rect.collidepoint(mouse_pos) and mouse_clicked
    
def ai_test_play():
    player = Player()
    game_objects = []
    score = 0
    last_object = pygame.time.get_ticks()
    
    # Utiliser les mêmes paramètres que dans la fonction main()
    obstacle_intervals = {
        6: [800, 1400],
        7: [900, 1600],
        8: [1100, 1800],
        9: [1300, 2000],
        10: [1400, 2100],
        11: [1500, 2200]
    }
    
    object_interval = random.randint(*obstacle_intervals[INITIAL_SCROLL_SPEED])
    current_speed = INITIAL_SCROLL_SPEED
    
    # Seuils pour les changements de vitesse
    speed_threshold_7 = random.randint(10, 20)
    min_threshold_8 = max(25, 2 * speed_threshold_7 - 5)
    max_threshold_8 = 2 * speed_threshold_7 + 10
    speed_threshold_8 = random.randint(min_threshold_8, max_threshold_8)
    min_threshold_9 = max(40, 2 * speed_threshold_8 - 15)
    max_threshold_9 = 2 * speed_threshold_8 + 5
    speed_threshold_9 = random.randint(min_threshold_9, max_threshold_9)
    speed_threshold_random = 100
    next_random_change = speed_threshold_random + random.randint(25, 50)
    
    # Nouvelle variable pour suivre le second saut
    second_jump_required = False
    
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
                
        # Logique de l'IA de test améliorée avec adaptation précise
        needs_to_jump = False
        
        # Paramètres de saut spécifiques à chaque vitesse et type d'obstacle
        jump_params = {
            6: {
                "Obstacle": {"detection": 20, "jump_timing": 12},
                "Block": {"detection": 22, "jump_timing": 22}
            },
            7: {
                "Obstacle": {"detection": 22, "jump_timing": 13},
                "Block": {"detection": 24, "jump_timing": 26},
                "DoublePikes": {"detection": 25, "jump_timing": 10}
            },
            8: {
                "Obstacle": {"detection": 24, "jump_timing": 14},
                "Block": {"detection": 26, "jump_timing": 29},
                "DoublePikes": {"detection": 27, "jump_timing": 10},
                "BlockGapBlockWithSpike": {"detection": 28, "jump_timing": 29}
            },
            9: {
                "Obstacle": {"detection": 26, "jump_timing": 15},
                "Block": {"detection": 28, "jump_timing": 16},
                "DoublePikes": {"detection": 29, "jump_timing": 18},
                "BlockGapBlockWithSpike": {"detection": 30, "jump_timing": 19},
                "TriplePikes": {"detection": 32, "jump_timing": 21}
            },
            10: {
                "DoublePikes": {"detection": 31, "jump_timing": 19},
                "BlockGapBlockWithSpike": {"detection": 32, "jump_timing": 20},
                "TriplePikes": {"detection": 34, "jump_timing": 22}
            },
            11: {
                "DoublePikes": {"detection": 33, "jump_timing": 20},
                "BlockGapBlockWithSpike": {"detection": 34, "jump_timing": 21},
                "TriplePikes": {"detection": 36, "jump_timing": 23},
                "QuadruplePikes": {"detection": 38, "jump_timing": 25}
            }
        }
        
        base_detection_distance = 200
        next_obstacle_x = float('inf')
        next_obstacle_type = None
        
        for obj in game_objects:
            obstacle_info = []
            
            if isinstance(obj, Obstacle):
                obstacle_info.append((obj.x, "Obstacle"))
            elif isinstance(obj, Block):
                obstacle_info.append((obj.rect.x, "Block"))
            elif isinstance(obj, DoublePikes):
                obstacle_info.append((obj.x, "DoublePikes"))
            elif isinstance(obj, TriplePikes):
                obstacle_info.append((obj.x, "TriplePikes"))
            elif isinstance(obj, QuadruplePikes):
                obstacle_info.append((obj.x, "QuadruplePikes"))
            elif isinstance(obj, BlockGapBlockWithSpike):
                obstacle_info.append((obj.x, "BlockGapBlockWithSpike"))
            
            for pos, obj_type in obstacle_info:
                # Récupérer les paramètres de saut spécifiques
                speed_params = jump_params.get(current_speed, {})
                obstacle_params = speed_params.get(obj_type, {})
                
                if obstacle_params:
                    detection_multiplier = obstacle_params.get("detection", 20)
                    jump_timing_multiplier = obstacle_params.get("jump_timing", 12)
                else:
                    # Valeurs par défaut si pas de paramètres spécifiques
                    detection_multiplier = 20
                    jump_timing_multiplier = 12
                
                detection_distance = base_detection_distance + (current_speed * detection_multiplier)
                
                # Calculer la distance entre le joueur et l'obstacle
                distance = pos - player.rect.right
                
                # Si l'obstacle est dans la zone de détection et devant le joueur
                if 0 < distance < detection_distance:
                    # Mettre à jour la position du prochain obstacle si c'est le plus proche
                    if pos < next_obstacle_x:
                        next_obstacle_x = pos
                        next_obstacle_type = obj_type
                        needs_to_jump = True
        
            # Faire sauter l'IA au bon moment si nécessaire
            if needs_to_jump and not player.is_jumping:
                # Récupérer les paramètres de saut spécifiques
                speed_params = jump_params.get(current_speed, {})
    
                obstacle_params = speed_params.get(next_obstacle_type, {})
    
                if obstacle_params:
                    jump_timing_multiplier = obstacle_params.get("jump_timing", 15)
                else:
                    jump_timing_multiplier = 15
    
                optimal_jump_distance = current_speed * jump_timing_multiplier
    
                # Décision de saut
                if next_obstacle_x - player.rect.right <= optimal_jump_distance:
                    player.jump()
        
                    # Gestion du second saut pour BlockGapBlockWithSpike
                    if next_obstacle_type == "BlockGapBlockWithSpike":
                        # Utiliser le flag just_landed pour le second saut
                        def check_second_jump():
                            if player.just_landed:
                                player.jump()
                                return True
                            return False
            
                        # Programmer un rappel pour le second saut
                        pygame.time.set_timer(pygame.USEREVENT, 100)  # Événement toutes les 100 ms
                        for event in pygame.event.get():
                            if event.type == pygame.USEREVENT:
                                if check_second_jump():
                                    pygame.time.set_timer(pygame.USEREVENT, 0)  # Arrêter le timer

        # Même logique que le jeu principal pour l'ajout d'obstacles
        if current_time - last_object > object_interval:
            # Choisir le type d'obstacle selon la vitesse
            if current_speed == 6:
                choice = random.random()
                if choice < 0.5:
                    obj = Obstacle(WIDTH)
                else:
                    obj = Block(WIDTH)
            elif current_speed == 7:
                choice = random.random()
                if choice < 0.5:
                    obj = Obstacle(WIDTH)
                elif choice < 0.8:
                    obj = Block(WIDTH)
                else:
                    obj = DoublePikes(WIDTH)
            elif current_speed == 8:
                choice = random.random()
                if choice < 0.2:
                    obj = Obstacle(WIDTH)
                elif choice < 0.4:
                    obj = Block(WIDTH)
                elif choice < 0.7:
                    obj = DoublePikes(WIDTH)
                else:
                    obj = BlockGapBlockWithSpike(WIDTH)
            elif current_speed == 9:
                choice = random.random()
                if choice < 0.08:
                    obj = Obstacle(WIDTH)
                elif choice < 0.16:
                    obj = Block(WIDTH)
                elif choice < 0.36:
                    obj = DoublePikes(WIDTH)
                elif choice < 0.66:
                    obj = BlockGapBlockWithSpike(WIDTH)
                elif choice < 0.96:
                    obj = TriplePikes(WIDTH)
                else:
                    obj = QuadruplePikes(WIDTH)
            elif current_speed >= 10:
                choice = random.random()
                if choice < 0.25:
                    obj = DoublePikes(WIDTH)
                elif choice < 0.60:
                    obj = BlockGapBlockWithSpike(WIDTH)
                elif choice < 0.95:
                    obj = TriplePikes(WIDTH)
                else:
                    obj = QuadruplePikes(WIDTH)
                
            obj.set_speed(current_speed)
            game_objects.append(obj)
            last_object = current_time
            object_interval = random.randint(*obstacle_intervals[current_speed])
        
        # Mettre à jour le joueur et vérifier les collisions
        player.update(game_objects)
        
        if not player.is_alive:
            print("Game Over! Score:", score)
            running = False
            return
        
        # Mettre à jour les objets du jeu et vérifier les collisions
        for obj in game_objects[:]:
            obj.update()
            
            # Vérifier les différentes collisions avec les obstacles
            if isinstance(obj, Obstacle) and player.rect.colliderect(obj.get_rect()):
                player.is_alive = False
                print("Game Over! Collision avec un obstacle")
                running = False
                return
            
            elif isinstance(obj, DoublePikes):
                for rect in obj.get_rects():
                    if player.rect.colliderect(rect):
                        player.is_alive = False
                        print("Game Over! Collision avec un double pic")
                        running = False
                        return
            
            elif isinstance(obj, TriplePikes):
                for rect in obj.get_rects():
                    if player.rect.colliderect(rect):
                        player.is_alive = False
                        print("Game Over! Collision avec un triple pic")
                        running = False
                        return
                        
            elif isinstance(obj, QuadruplePikes):
                for rect in obj.get_rects():
                    if player.rect.colliderect(rect):
                        player.is_alive = False
                        print("Game Over! Collision avec un quadruple pic")
                        running = False
                        return
            
            # Supprimer les objets sortis de l'écran et augmenter le score
            if ((isinstance(obj, Obstacle) and obj.x + obj.width < 0) or
                (isinstance(obj, Block) and obj.rect.right < 0) or
                (isinstance(obj, DoublePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, TriplePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, QuadruplePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, BlockGapBlockWithSpike) and obj.x + obj.width < 0)):
                game_objects.remove(obj)
                score += 1
                
                # Vérifier les paliers de vitesse
                if score < speed_threshold_random:
                    if score == speed_threshold_7 and current_speed < 7:
                        current_speed = 7
                        print(f"Passage à la vitesse 7 à {score} points!")
                        for game_obj in game_objects:
                            game_obj.set_speed(current_speed)
                    elif score == speed_threshold_8 and current_speed < 8:
                        current_speed = 8
                        print(f"Passage à la vitesse 8 à {score} points!")
                        for game_obj in game_objects:
                            game_obj.set_speed(current_speed)
                    elif score == speed_threshold_9 and current_speed < 9:
                        current_speed = 9
                        print(f"Passage à la vitesse 9 à {score} points!")
                        for game_obj in game_objects:
                            game_obj.set_speed(current_speed)
                elif score == speed_threshold_random:
                    new_speed = random.randint(9, 11)
                    current_speed = new_speed
                    print(f"Passage à la vitesse aléatoire {new_speed} à {score} points!")
                    for game_obj in game_objects:
                        game_obj.set_speed(current_speed)
                    next_random_change = score + random.randint(25, 50)
                    print(f"Prochain changement à {next_random_change} points")
                elif score >= speed_threshold_random and score == next_random_change:
                    new_speed = random.randint(9, 11)
                    current_speed = new_speed
                    print(f"Nouveau changement à la vitesse aléatoire {new_speed} à {score} points!")
                    for game_obj in game_objects:
                        game_obj.set_speed(current_speed)
                    next_random_change = score + random.randint(25, 50)
                    print(f"Prochain changement à {next_random_change} points")
        
        # Dessiner l'écran
        screen.fill(WHITE)
        pygame.draw.rect(screen, BLACK, (0, GROUND_HEIGHT, WIDTH, HEIGHT - GROUND_HEIGHT))
        player.draw()
        
        for obj in game_objects:
            obj.draw()
            
        # Afficher le score et la vitesse
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (20, 20))
        
        speed_text = font.render(f"Vitesse: {current_speed}", True, BLACK)
        screen.blit(speed_text, (20, 60))

        if score >= speed_threshold_random:
            next_change_text = font.render(f"Prochain changement à: {next_random_change}", True, BLACK)
            screen.blit(next_change_text, (20, 100))
        
        # Indiquer que c'est l'IA qui joue
        ai_text = font.render("Mode IA Test", True, (255, 0, 0))
        screen.blit(ai_text, (WIDTH - 150, 20))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    return

# Déplacer la fonction main() avant show_menu()
def main():
    player = Player()
    game_objects = []  # Liste combinée pour les obstacles et les blocs
    score = 0
    last_object = pygame.time.get_ticks()
    
    # Intervalles de spawn des obstacles selon la vitesse (modifiés pour être plus rapprochés)
    # Format: [min_interval, max_interval] en millisecondes
    obstacle_intervals = {
        6: [800, 1400],  # Vitesse 6
        7: [900, 1600],  # Vitesse 7
        8: [1100, 1800],  # Vitesse 8
        9: [1300, 2000],  # Vitesse 9
        10: [1400, 2100], # Vitesse 10 - légèrement plus long que la vitesse 9
        11: [1500, 2200]  # Vitesse 11 - légèrement plus long que la vitesse 10
    }
    
    # Intervalle initial
    object_interval = random.randint(*obstacle_intervals[INITIAL_SCROLL_SPEED])
    
    # Vitesse de défilement actuelle
    current_speed = INITIAL_SCROLL_SPEED
    
    # Pour détecter quand la touche espace est d'abord pressée
    space_pressed = False
    
    # Seuils de score aléatoires pour les changements de vitesse
    speed_threshold_7 = random.randint(10, 20)
    
    # Pour le seuil de vitesse 8, prendre le max entre 25 et (2*seuil_7 - 5)
    min_threshold_8 = max(25, 2 * speed_threshold_7 - 5)
    max_threshold_8 = 2 * speed_threshold_7 + 10
    speed_threshold_8 = random.randint(min_threshold_8, max_threshold_8)
    
    # Pour le seuil de vitesse 9, prendre le max entre 40 et (2*seuil_8 - 15)
    min_threshold_9 = max(40, 2 * speed_threshold_8 - 15)
    max_threshold_9 = 2 * speed_threshold_8 + 5
    speed_threshold_9 = random.randint(min_threshold_9, max_threshold_9)
    
    # Nouveau seuil pour vitesse aléatoire (100)
    speed_threshold_random = 100
    
    # Prochain seuil pour le changement de vitesse aléatoire
    next_random_change = speed_threshold_random + random.randint(25, 50)
    
    # Boucle de jeu
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            # Détecter quand la touche espace est relâchée
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    space_pressed = False

        # Vérifier si la touche espace est maintenue enfoncée
        keys = pygame.key.get_pressed()
        
        # Si la touche espace vient d'être enfoncée ou si on vient d'atterrir
        if keys[pygame.K_SPACE]:
            # Si c'est une nouvelle pression ou si on vient juste d'atterrir
            if not space_pressed or player.just_landed:
                if not player.is_jumping:  # Ne sauter que si on n'est pas déjà en saut
                    player.jump()
                space_pressed = True
        
        # Ajouter de nouveaux obstacles ou blocs
        if current_time - last_object > object_interval:
            # Choisir le type d'obstacle en fonction de la vitesse actuelle
            if current_speed == 6:
                # Vitesse 6: 50% pic, 50% bloc
                choice = random.random()
                if choice < 0.5:  # 50 % pour un obstacle simple
                    obj = Obstacle(WIDTH)
                else:  # 50% pour un bloc
                    obj = Block(WIDTH)
            elif current_speed == 7:
                # Vitesse 7: distribution originale
                choice = random.random()
                if choice < 0.5:  # 50% pour un obstacle simple
                    obj = Obstacle(WIDTH)
                elif choice < 0.8:  # 30% pour un bloc
                    obj = Block(WIDTH)
                else:  # 20% pour une structure de double pics
                    obj = DoublePikes(WIDTH)
            elif current_speed == 8:
                # Vitesse 8: 20% pic, 20% bloc, 30% doublepic, 30% nouvelle structure
                choice = random.random()
                if choice < 0.2:  # 20% pour un obstacle simple 
                    obj = Obstacle(WIDTH)
                elif choice < 0.4:  # 20% pour un bloc
                    obj = Block(WIDTH)
                elif choice < 0.7:  # 30% pour un double pic
                    obj = DoublePikes(WIDTH)
                else:  # 30% pour la nouvelle structure
                    obj = BlockGapBlockWithSpike(WIDTH)
            elif current_speed == 9:
                # Vitesse 9: 8% pic, 8% bloc, 20% doublepic, 30% nouvelle structure, 30% triple pics, 4% quadruple pics
                choice = random.random()
                if choice < 0.08:  # 8% pour un obstacle simple
                    obj = Obstacle(WIDTH)
                elif choice < 0.16:  # 8% pour un bloc
                    obj = Block(WIDTH)
                elif choice < 0.36:  # 20% pour un double pic
                    obj = DoublePikes(WIDTH)
                elif choice < 0.66:  # 30% pour la nouvelle structure
                    obj = BlockGapBlockWithSpike(WIDTH)
                elif choice < 0.96:  # 30% pour les triples pics
                    obj = TriplePikes(WIDTH)
                else:  # 4% pour les quadruples pics
                    obj = QuadruplePikes(WIDTH)
            elif current_speed >= 10:
                # Vitesses 10 et 11: pas de pic seul ni de bloc seul, distribution similaire à 9
                choice = random.random()
                if choice < 0.25:  # 25% pour un double pic (augmenté)
                    obj = DoublePikes(WIDTH)
                elif choice < 0.60:  # 35% pour la nouvelle structure (augmenté)
                    obj = BlockGapBlockWithSpike(WIDTH)
                elif choice < 0.95:  # 35% pour les triples pics (augmenté)
                    obj = TriplePikes(WIDTH)
                else:  # 5% pour les quadruples pics (légèrement augmenté)
                    obj = QuadruplePikes(WIDTH)
                
            # Définir la vitesse actuelle pour le nouvel objet
            obj.set_speed(current_speed)
            game_objects.append(obj)
                
            last_object = current_time
            # Utiliser l'intervalle correspondant à la vitesse actuelle
            object_interval = random.randint(*obstacle_intervals[current_speed])
        
        # Mettre à jour le joueur
        player.update(game_objects)
        
        if not player.is_alive:
            print("Game Over! Score:", score)
            running = False
        
        # Mettre à jour les objets du jeu
        for obj in game_objects[:]:
            obj.update()
            
            # Vérifier les collisions avec les obstacles rouges
            if isinstance(obj, Obstacle) and player.rect.colliderect(obj.get_rect()):
                player.is_alive = False
                print("Game Over! Collision avec un obstacle")
                running = False
            
            # Vérifier les collisions avec les doubles pics
            elif isinstance(obj, DoublePikes):
                for rect in obj.get_rects():
                    if player.rect.colliderect(rect):
                        player.is_alive = False
                        print("Game Over! Collision avec un double pic")
                        running = False
                        break
            
            # Vérifier les collisions avec les triples pics
            elif isinstance(obj, TriplePikes):
                for rect in obj.get_rects():
                    if player.rect.colliderect(rect):
                        player.is_alive = False
                        print("Game Over! Collision avec un triple pic")
                        running = False
                        break
                        
            # Vérifier les collisions avec les quadruples pics
            elif isinstance(obj, QuadruplePikes):
                for rect in obj.get_rects():
                    if player.rect.colliderect(rect):
                        player.is_alive = False
                        print("Game Over! Collision avec un quadruple pic")
                        running = False
                        break
            
            # Remarque: Les collisions avec la structure spéciale sont gérées dans la méthode update du joueur
                
            # Supprimer les objets hors écran
            if ((isinstance(obj, Obstacle) and obj.x + obj.width < 0) or
                (isinstance(obj, Block) and obj.rect.right < 0) or
                (isinstance(obj, DoublePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, TriplePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, QuadruplePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, BlockGapBlockWithSpike) and obj.x + obj.width < 0)):
                game_objects.remove(obj)
                score += 1
                
                # Vérifier si le score a atteint les paliers pour la progression standard des vitesses
                if score < speed_threshold_random:  # Uniquement avant d'atteindre le seuil pour vitesse aléatoire
                    if score == speed_threshold_7 and current_speed < 7:
                        current_speed = 7
                        print(f"Passage à la vitesse 7 à {score} points!")
                        # Mettre à jour la vitesse de tous les objets existants
                        for game_obj in game_objects:
                            game_obj.set_speed(current_speed)
                    elif score == speed_threshold_8 and current_speed < 8:
                        current_speed = 8
                        print(f"Passage à la vitesse 8 à {score} points!")
                        # Mettre à jour la vitesse de tous les objets existants
                        for game_obj in game_objects:
                            game_obj.set_speed(current_speed)
                    elif score == speed_threshold_9 and current_speed < 9:
                        current_speed = 9
                        print(f"Passage à la vitesse 9 à {score} points!")
                        # Mettre à jour la vitesse de tous les objets existants
                        for game_obj in game_objects:
                            game_obj.set_speed(current_speed)
                # Vérifier les seuils pour la vitesse aléatoire
                elif score == speed_threshold_random:
                    # Premier changement aléatoire à 100 points
                    new_speed = random.randint(9, 11)
                    current_speed = new_speed
                    print(f"Passage à la vitesse aléatoire {new_speed} à {score} points!")
                    # Mettre à jour la vitesse de tous les objets existants
                    for game_obj in game_objects:
                        game_obj.set_speed(current_speed)
                    # Définir le prochain seuil
                    next_random_change = score + random.randint(25, 50)
                    print(f"Prochain changement à {next_random_change} points")
                elif score >= speed_threshold_random and score == next_random_change:
                    # Changements aléatoires subséquents
                    new_speed = random.randint(9, 11)
                    current_speed = new_speed
                    print(f"Nouveau changement à la vitesse aléatoire {new_speed} à {score} points!")
                    # Mettre à jour la vitesse de tous les objets existants
                    for game_obj in game_objects:
                        game_obj.set_speed(current_speed)
                    # Définir le prochain seuil
                    next_random_change = score + random.randint(25, 50)
                    print(f"Prochain changement à {next_random_change} points")
        
        # Dessiner l'arrière-plan
        screen.fill(WHITE)
        
        # Dessiner le sol
        pygame.draw.rect(screen, BLACK, (0, GROUND_HEIGHT, WIDTH, HEIGHT - GROUND_HEIGHT))
        
        # Dessiner le joueur
        player.draw()
        
        # Dessiner les objets
        for obj in game_objects:
            obj.draw()
            
        # Afficher le score
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (20, 20))
        
        # Afficher la vitesse actuelle
        speed_text = font.render(f"Vitesse: {current_speed}", True, BLACK)
        screen.blit(speed_text, (20, 60))

        # Afficher le prochain changement de vitesse si la partie est avancée
        if score >= speed_threshold_random:
            next_change_text = font.render(f"Prochain changement à: {next_random_change}", True, BLACK)
            screen.blit(next_change_text, (20, 100))
        
        # Mettre à jour l'écran
        pygame.display.flip()
         
        # Contrôler la vitesse du jeu
        clock.tick(FPS)
    
    # Revenir au menu après la fin de la partie
    show_menu()

# Fonction pour afficher le menu
def show_menu():
    # Couleurs pour les boutons
    button_color = (200, 200, 200)  # Gris clair
    hover_color = (150, 150, 150)   # Gris foncé
    
    # Créer les boutons
    button_width, button_height = 200, 50
    start_x = WIDTH // 2 - button_width // 2
    
    player_button = Button("Joueur", start_x, 200, button_width, button_height, button_color, hover_color)
    ai_reinforcement_button = Button("IA par Renforcement", start_x, 280, button_width, button_height, button_color, hover_color)
    ai_test_button = Button("IA Test", start_x, 360, button_width, button_height, button_color, hover_color)
    
    # Boucle du menu
    menu_running = True
    while menu_running:
        screen.fill(WHITE)
        
        # Afficher le titre
        font_title = pygame.font.SysFont(None, 60)
        title_text = font_title.render("Geometry Dash Clone", True, BLACK)
        title_rect = title_text.get_rect(center=(WIDTH // 2, 100))
        screen.blit(title_text, title_rect)
        
        # Obtenir la position de la souris
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Bouton gauche de la souris
                    mouse_clicked = True
        
        # Mettre à jour et dessiner les boutons
        player_button.update(mouse_pos)
        ai_reinforcement_button.update(mouse_pos)
        ai_test_button.update(mouse_pos)
        
        player_button.draw(screen)
        ai_reinforcement_button.draw(screen)
        ai_test_button.draw(screen)
        
        # Vérifier si un bouton a été cliqué
        if player_button.check_clicked(mouse_pos, mouse_clicked):
            menu_running = False
            main()  # Lancer le jeu normal
        elif ai_reinforcement_button.check_clicked(mouse_pos, mouse_clicked):
            # Ne fait rien pour l'instant, comme demandé
            font = pygame.font.SysFont(None, 24)
            info_text = font.render("Fonctionnalité non implémentée", True, RED)
            screen.blit(info_text, (WIDTH // 2 - 120, 450))
            pygame.display.flip()
            pygame.time.wait(1000)  # Afficher le message pendant 1 seconde
        elif ai_test_button.check_clicked(mouse_pos, mouse_clicked):
            menu_running = False
            ai_test_play()  # Lancer le mode IA test
            show_menu()  # Revenir au menu après la partie
        
        pygame.display.flip()
        clock.tick(30)

# Modifier la structure principale pour lancer le menu au démarrage
if __name__ == "__main__":
    show_menu()    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
