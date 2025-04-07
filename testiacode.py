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
BLUE = (0, 0, 255) # cardinal de pauvrelieu <-- IMPORTANT
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
        square_size = self.height  # Size of the inscribed square
        return pygame.Rect(self.x + (self.width / 2 - square_size / 2), self.y, square_size, square_size)


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
        square_size = self.height  # Size of the inscribed square
        return [
            pygame.Rect(self.x, self.y, square_size, square_size),
            pygame.Rect(self.x + CUBE_SIZE, self.y, square_size, square_size)
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
    
    obstacle_intervals = {
        6: [800, 1400],
        7: [900, 1600],
        8: [1200, 1800],
        9: [1300, 2000],
        10: [1400, 2100],
        11: [1500, 2200]
    }
    
    object_interval = random.randint(*obstacle_intervals[INITIAL_SCROLL_SPEED])
    current_speed = INITIAL_SCROLL_SPEED
    
    speed_threshold_7 = random.randint(10, 20)
    min_threshold_8 = max(25, 2 * speed_threshold_7 - 5)
    max_threshold_8 = 2 * speed_threshold_7 + 10
    speed_threshold_8 = random.randint(min_threshold_8, max_threshold_8)
    min_threshold_9 = max(40, 2 * speed_threshold_8 - 15)
    max_threshold_9 = 2 * speed_threshold_8 + 5
    speed_threshold_9 = random.randint(min_threshold_9, max_threshold_9)
    speed_threshold_random = 100
    next_random_change = speed_threshold_random + random.randint(25, 50)
    
    # Dictionnaire pour le spam de sauts
    jump_spam_timers = {}
    
    # Paramètres complets pour chaque obstacle et chaque vitesse
    jump_params = {
        6: {
            "Obstacle": {"detection": 20, "jump_timing": 12, "jump_count": 1},
            "Block": {"detection": 22, "jump_timing": 22, "jump_count": 1},
            "DoublePikes": {"detection": 24, "jump_timing": 14, "jump_count": 1},
            "TriplePikes": {"detection": 26, "jump_timing": 16, "jump_count": 1},
            "QuadruplePikes": {"detection": 28, "jump_timing": 18, "jump_count": 2},
            "BlockGapBlockWithSpike": {"detection": 26, "jump_timing": 26, "jump_count": 2}
        },
        7: {
            "Obstacle": {"detection": 22, "jump_timing": 13, "jump_count": 1},
            "Block": {"detection": 24, "jump_timing": 26, "jump_count": 1},
            "DoublePikes": {"detection": 25, "jump_timing": 10, "jump_count": 1},
            "TriplePikes": {"detection": 27, "jump_timing": 17, "jump_count": 1},
            "QuadruplePikes": {"detection": 29, "jump_timing": 19, "jump_count": 2},
            "BlockGapBlockWithSpike": {"detection": 28, "jump_timing": 28, "jump_count": 2}
        },
        8: {
            "Obstacle": {"detection": 24, "jump_timing": 14, "jump_count": 1},
            "Block": {"detection": 26, "jump_timing": 29, "jump_count": 1},
            "DoublePikes": {"detection": 27, "jump_timing": 10, "jump_count": 1},
            "TriplePikes": {"detection": 29, "jump_timing": 8, "jump_count": 1},
            "QuadruplePikes": {"detection": 31, "jump_timing": 7, "jump_count": 2},
            "BlockGapBlockWithSpike": {"detection": 10, "jump_timing": 31, "jump_count": 2}
        },
        9: {
            "Obstacle": {"detection": 26, "jump_timing": 8, "jump_count": 1},
            "Block": {"detection": 28, "jump_timing": 8, "jump_count": 1},
            "DoublePikes": {"detection": 29, "jump_timing": 10, "jump_count": 1},
            "TriplePikes": {"detection": 32, "jump_timing": 8, "jump_count": 1},
            "QuadruplePikes": {"detection": 34, "jump_timing": 3, "jump_count": 2},
            "BlockGapBlockWithSpike": {"detection": 40, "jump_timing": 30, "jump_count": 2}
        },
        10: {
            "Obstacle": {"detection": 28, "jump_timing": 16, "jump_count": 1},
            "Block": {"detection": 30, "jump_timing": 34, "jump_count": 1},
            "DoublePikes": {"detection": 31, "jump_timing": 10, "jump_count": 1},
            "TriplePikes": {"detection": 34, "jump_timing": 22, "jump_count": 1},
            "QuadruplePikes": {"detection": 36, "jump_timing": 24, "jump_count": 2},
            "BlockGapBlockWithSpike": {"detection": 34, "jump_timing": 30, "jump_count": 2}
        },
        11: {
            "Obstacle": {"detection": 30, "jump_timing": 17, "jump_count": 1},
            "Block": {"detection": 32, "jump_timing": 36, "jump_count": 1},
            "DoublePikes": {"detection": 33, "jump_timing": 10, "jump_count": 1},
            "TriplePikes": {"detection": 36, "jump_timing": 23, "jump_count": 1},
            "QuadruplePikes": {"detection": 38, "jump_timing": 25, "jump_count": 2},
            "BlockGapBlockWithSpike": {"detection": 36, "jump_timing": 37, "jump_count": 2}
        }
    }
    
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
                
        needs_to_jump = False
        
        # Gestion du spam de sauts
        for obj_id in list(jump_spam_timers.keys()):
            timer_info = jump_spam_timers[obj_id]
            if current_time >= timer_info["next_jump_time"]:
                if timer_info["jumps_remaining"] > 0 and not player.is_jumping:
                    player.jump()
                    timer_info["jumps_remaining"] -= 1
                    timer_info["next_jump_time"] = current_time + 120  # 100ms entre les sauts
                else:
                    del jump_spam_timers[obj_id]
        
        base_detection_distance = 200
        next_obstacle_x = float('inf')
        next_obstacle_type = None
        next_obstacle_id = None
        
        for obj in game_objects:
            obstacle_info = []
            obj_id = id(obj)  # Identifiant unique pour l'objet
            
            if isinstance(obj, Obstacle):
                obstacle_info.append((obj.x, "Obstacle", obj_id))
            elif isinstance(obj, Block):
                obstacle_info.append((obj.rect.x, "Block", obj_id))
            elif isinstance(obj, DoublePikes):
                obstacle_info.append((obj.x, "DoublePikes", obj_id))
            elif isinstance(obj, TriplePikes):
                obstacle_info.append((obj.x, "TriplePikes", obj_id))
            elif isinstance(obj, QuadruplePikes):
                obstacle_info.append((obj.x, "QuadruplePikes", obj_id))
            elif isinstance(obj, BlockGapBlockWithSpike):
                obstacle_info.append((obj.x, "BlockGapBlockWithSpike", obj_id))
            
            for pos, obj_type, obj_id in obstacle_info:
                speed_params = jump_params.get(current_speed, {})
                obstacle_params = speed_params.get(obj_type, {})
                
                if obstacle_params:
                    detection_multiplier = obstacle_params.get("detection", 20)
                    jump_timing_multiplier = obstacle_params.get("jump_timing", 12)
                    jump_count = obstacle_params.get("jump_count", 1)
                else:
                    detection_multiplier = 20
                    jump_timing_multiplier = 12
                    jump_count = 1
                
                detection_distance = base_detection_distance + (current_speed * detection_multiplier)
                
                distance = pos - player.rect.right
                
                if 0 < distance < detection_distance:
                    if pos < next_obstacle_x:
                        next_obstacle_x = pos
                        next_obstacle_type = obj_type
                        next_obstacle_id = obj_id
                        needs_to_jump = True
        
            if needs_to_jump and next_obstacle_id not in jump_spam_timers:
                speed_params = jump_params.get(current_speed, {})
                obstacle_params = speed_params.get(next_obstacle_type, {})
    
                if obstacle_params:
                    jump_timing_multiplier = obstacle_params.get("jump_timing", 15)
                    jump_count = obstacle_params.get("jump_count", 1)
                else:
                    jump_timing_multiplier = 15
                    jump_count = 1
    
                optimal_jump_distance = current_speed * jump_timing_multiplier
    
                if next_obstacle_x - player.rect.right <= optimal_jump_distance:
                    # Initialiser le spam de sauts
                    jump_spam_timers[next_obstacle_id] = {
                        "jumps_remaining": jump_count - 1,  # -1 car on va faire un saut immédiatement
                        "next_jump_time": current_time + 100  # 100ms après le premier saut
                    }
                    
                    # Premier saut immédiat
                    player.jump()

        if current_time - last_object > object_interval:
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
        
        player.update(game_objects)
        
        if not player.is_alive:
            print("Game Over! Score:", score)
            running = False
            return
        
        for obj in game_objects[:]:
            obj.update()
            
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
            
            # Suppression de l'obstacle une fois qu'il est sorti de l'écran
            # et suppression du timer correspondant s'il existe
            obj_id = id(obj)
            if ((isinstance(obj, Obstacle) and obj.x + obj.width < 0) or
                (isinstance(obj, Block) and obj.rect.right < 0) or
                (isinstance(obj, DoublePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, TriplePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, QuadruplePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, BlockGapBlockWithSpike) and obj.x + obj.width < 0)):
                game_objects.remove(obj)
                if obj_id in jump_spam_timers:
                    del jump_spam_timers[obj_id]
                score += 1
                
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
        
        screen.fill(WHITE)
        pygame.draw.rect(screen, BLACK, (0, GROUND_HEIGHT, WIDTH, HEIGHT - GROUND_HEIGHT))
        player.draw()
        
        for obj in game_objects:
            obj.draw()
            
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (20, 20))
        
        speed_text = font.render(f"Vitesse: {current_speed}", True, BLACK)
        screen.blit(speed_text, (20, 60))

        if score >= speed_threshold_random:
            next_change_text = font.render(f"Prochain changement à: {next_random_change}", True, BLACK)
            screen.blit(next_change_text, (20, 100))
        
        ai_text = font.render("Mode IA Test", True, (255, 0, 0))
        screen.blit(ai_text, (WIDTH - 150, 20))

        
        pygame.display.flip()
        clock.tick(FPS)
    
    return
