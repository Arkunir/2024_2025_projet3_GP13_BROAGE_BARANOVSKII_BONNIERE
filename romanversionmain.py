import pygame
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