import pygame
<<<<<<< HEAD
import random

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_COLOR = 120, 170, 255

FLOOR_Y = 550

PLAYER_COLOR = 255, 255, 255

EVENT_OBSTACLE_SPAWN = pygame.USEREVENT + 1

FPS = 60


class Floor(pygame.sprite.Sprite):
    def __init__(self, y):
        super().__init__()
        self.image = pygame.Surface((SCREEN_RECT.w, 100))
        self.image.fill((100, 200, 80))
        self.rect = self.image.get_rect()
        self.rect.topleft = 0, y


class Player(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((50, 50))
        self.image.fill(PLAYER_COLOR)
        self.rect = self.image.get_rect()
        self.rect.bottomleft = x, y
        self.x = x
        self.y = y
        self.jumping = False
        self.jump_count = 20

    def move(self, amount):
        self.rect.x += amount

    def jump(self):
        self.jumping = True

    def collide(self, sprites):
        # Retourne les sprites en collision avec le joueur
        return tuple(sprites[i] for i in self.rect.collidelistall(sprites))

    def update(self):
        if self.jumping:
            # step = -(0.5 * self.jump_count ** 2)
            step = -(0.05 * self.jump_count ** 2) * (60 / FPS)
            if self.jump_count > 0:
                step *= -1
            self.rect.y -= step
            self.jump_count -= 1
            if self.rect.bottom >= self.y:
                self.rect.bottom = self.y
                self.jumping = False
                self.jump_count = 20


class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = pygame.Surface((width, height), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self.rect.bottomleft = x, y

    def update(self):
        self.rect.x -= 5


class Friend(Obstacle):
    points = 10
    def __init__(self, x, y):
        super().__init__(x, y, 60, 60)
        pygame.draw.circle(self.image, (0, 155, 155), (30, 30), 30)


class Ennemy(Obstacle):
    points = -15
    def __init__(self, x, y):
        super().__init__(x, y, 60, 60)
        pygame.draw.polygon(self.image, (0, 0, 0), ((30, 0), (0, 60), (60, 60)))


def obstacle_spawn():
    # Choix d'un obstacle (30% ami, 70% ennemi)
    obstacle_class = random.choices((Friend, Ennemy), (3, 7))[0]
    obstacle = obstacle_class(SCREEN_RECT.w, FLOOR_Y)
    draw_group.add(obstacle)
    obstacles_group.add(obstacle)


def spawn_timer():
    ms = random.randrange(500, 1100)
    pygame.time.set_timer(EVENT_OBSTACLE_SPAWN, ms)


def sprites_clear(surface, rect):
    surface.fill(SCREEN_COLOR, rect)


def pause():
    clock = pygame.time.Clock()
    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                return


screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
SCREEN_RECT = screen.get_rect()
pygame.mouse.set_visible(False)
pygame.display.set_caption("test jeu")

draw_group = pygame.sprite.RenderUpdates()
obstacles_group = pygame.sprite.Group()

player = Player(150, FLOOR_Y)
draw_group.add(player)

floor = Floor(FLOOR_Y)
draw_group.add(floor)

screen.fill(SCREEN_COLOR)
pygame.display.update()

clock = pygame.time.Clock()
running = True
points = 0
spawn_timer()
game_paused = False

while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and not player.jumping:
                player.jump()
            elif event.key == pygame.K_p:
                game_paused = not game_paused
                if game_paused:
                    pause()
                    game_paused = False
            elif event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == EVENT_OBSTACLE_SPAWN:
            obstacle_spawn()
            spawn_timer()
            draw_group.remove(player)
            draw_group.add(player)
        elif event.type == pygame.QUIT:
            running = False

    # Collision obstacles avec le joueur
    for sprite in player.collide(obstacles_group.sprites()):
        points += sprite.points
        obstacles_group.remove(sprite)
        draw_group.remove(sprite)
        print("points : ", points)

    # Suppression des sprites hors écran
    for sprite in obstacles_group.copy().sprites():
        if sprite.rect.right < SCREEN_RECT.x:
            obstacles_group.remove(sprite)
            draw_group.remove(sprite)

    draw_group.clear(screen, sprites_clear)
    draw_group.update()
    rects = draw_group.draw(screen)
    pygame.display.update(rects)


pygame.quit()
=======
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
        pygame.draw.rect(screen, BLACK, pygame.Rect(self.x, self.y, CUBE_SIZE, CUBE_SIZE))
        pygame.draw.rect(screen, BLACK, pygame.Rect(self.x + CUBE_SIZE * 3, self.y, CUBE_SIZE, CUBE_SIZE))
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
        8: [1100, 1800],
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
    
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
                
        needs_to_jump = False
        jump_distance = 180 + (current_speed - 6) * 20
        next_obstacle_x = float('inf')
        
        for obj in game_objects:
            obstacle_positions = []
            
            if isinstance(obj, Obstacle):
                obstacle_positions.append(obj.x)
            elif isinstance(obj, DoublePikes):
                obstacle_positions.append(obj.x)
            elif isinstance(obj, TriplePikes):
                obstacle_positions.append(obj.x)
            elif isinstance(obj, QuadruplePikes):
                obstacle_positions.append(obj.x)
            elif isinstance(obj, Block):
                obstacle_positions.append(obj.x)
            elif isinstance(obj, BlockGapBlockWithSpike):
                obstacle_positions.append(obj.x)
                obstacle_positions.append(obj.x + CUBE_SIZE * 3)

            for pos in obstacle_positions:
                distance = pos - player.rect.right
                
                if 0 < distance < jump_distance:
                    if pos < next_obstacle_x:
                        next_obstacle_x = pos
                        needs_to_jump = True
        
        if needs_to_jump and not player.is_jumping:
            if next_obstacle_x - player.rect.right <= 30 + (current_speed * 10):
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
            
            if ((isinstance(obj, Obstacle) and obj.x + obj.width < 0) or
                (isinstance(obj, Block) and obj.rect.right < 0) or
                (isinstance(obj, DoublePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, TriplePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, QuadruplePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, BlockGapBlockWithSpike) and obj.x + obj.width < 0)):
                game_objects.remove(obj)
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

def main():
    player = Player()
    game_objects = []
    score = 0
    last_object = pygame.time.get_ticks()
    
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
    
    space_pressed = False
    
    speed_threshold_7 = random.randint(10, 20)
    
    min_threshold_8 = max(25, 2 * speed_threshold_7 - 5)
    max_threshold_8 = 2 * speed_threshold_7 + 10
    speed_threshold_8 = random.randint(min_threshold_8, max_threshold_8)
    
    min_threshold_9 = max(40, 2 * speed_threshold_8 - 15)
    max_threshold_9 = 2 * speed_threshold_8 + 5
    speed_threshold_9 = random.randint(min_threshold_9, max_threshold_9)
    
    speed_threshold_random = 100
    
    next_random_change = speed_threshold_random + random.randint(25, 50)
    
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    space_pressed = False

        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_SPACE]:
            if not space_pressed or player.just_landed:
                if not player.is_jumping:
                    player.jump()
                space_pressed = True
        
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
        
        for obj in game_objects[:]:
            obj.update()
            
            if isinstance(obj, Obstacle) and player.rect.colliderect(obj.get_rect()):
                player.is_alive = False
                print("Game Over! Collision avec un obstacle")
                running = False
            
            elif isinstance(obj, DoublePikes):
                for rect in obj.get_rects():
                    if player.rect.colliderect(rect):
                        player.is_alive = False
                        print("Game Over! Collision avec un double pic")
                        running = False
                        break
            
            elif isinstance(obj, TriplePikes):
                for rect in obj.get_rects():
                    if player.rect.colliderect(rect):
                        player.is_alive = False
                        print("Game Over! Collision avec un triple pic")
                        running = False
                        break
                        
            elif isinstance(obj, QuadruplePikes):
                for rect in obj.get_rects():
                    if player.rect.colliderect(rect):
                        player.is_alive = False
                        print("Game Over! Collision avec un quadruple pic")
                        running = False
                        break
            
            if ((isinstance(obj, Obstacle) and obj.x + obj.width < 0) or
                (isinstance(obj, Block) and obj.rect.right < 0) or
                (isinstance(obj, DoublePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, TriplePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, QuadruplePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, BlockGapBlockWithSpike) and obj.x + obj.width < 0)):
                game_objects.remove(obj)
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
        
        pygame.display.flip()
        clock.tick(FPS)
    
    show_menu()

def show_menu():
    button_color = (200, 200, 200)
    hover_color = (150, 150, 150)
    
    button_width, button_height = 200, 50
    start_x = WIDTH // 2 - button_width // 2
    
    player_button = Button("Joueur", start_x, 200, button_width, button_height, button_color, hover_color)
    ai_reinforcement_button = Button("IA par Renforcement", start_x, 280, button_width, button_height, button_color, hover_color)
    ai_test_button = Button("IA Test", start_x, 360, button_width, button_height, button_color, hover_color)
    
    menu_running = True
    while menu_running:
        screen.fill(WHITE)
        
        font_title = pygame.font.SysFont(None, 60)
        title_text = font_title.render("Geometry Dash Clone", True, BLACK)
        title_rect = title_text.get_rect(center=(WIDTH // 2, 100))
        screen.blit(title_text, title_rect)
        
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_clicked = True
        
        player_button.update(mouse_pos)
        ai_reinforcement_button.update(mouse_pos)
        ai_test_button.update(mouse_pos)
        
        player_button.draw(screen)
        ai_reinforcement_button.draw(screen)
        ai_test_button.draw(screen)
        
        if player_button.check_clicked(mouse_pos, mouse_clicked):
            menu_running = False
            main()
        elif ai_reinforcement_button.check_clicked(mouse_pos, mouse_clicked):
            font = pygame.font.SysFont(None, 24)
            info_text = font.render("Fonctionnalité non implémentée", True, RED)
            screen.blit(info_text, (WIDTH // 2 - 120, 450))
            pygame.display.flip()
            pygame.time.wait(1000)
        elif ai_test_button.check_clicked(mouse_pos, mouse_clicked):
            menu_running = False
            ai_test_play()
            show_menu()
        
        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    show_menu()    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

>>>>>>> 7690fba68180c3839f96d7f13e7fb711e4cf7f85
