import pygame

# Constantes
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRAVITY = 0.7
JUMP_FORCE = -14
GAME_SPEED = 5  # Vitesse 1 de Geometry Dash

COLORS = {
    "background": (50, 50, 70),
    "player": (255, 150, 0),
    "platform": (100, 100, 100),
    "obstacle": (255, 50, 50),
    "button": (80, 80, 200),
    "text": (255, 255, 255)
}

class Player:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y - 30, 30, 30)  # Carré de 30x30
        self.velocity_y = 0
        self.on_ground = False
        self.speed = GAME_SPEED
    
    def update(self, keys):
        # Gravité
        self.velocity_y += GRAVITY
        self.rect.y += self.velocity_y
        
        # Avancer automatiquement (comme dans Geometry Dash)
        self.rect.x += self.speed
        
        # Vérifier si le joueur est tombé hors de l'écran
        if self.rect.y > SCREEN_HEIGHT:
            self.rect.y = SCREEN_HEIGHT
            self.velocity_y = 0
            self.on_ground = True
    
    def jump(self):
        # Sauter seulement si au sol
        if self.on_ground:
            self.velocity_y = JUMP_FORCE
            self.on_ground = False
    
    def draw(self, screen, scroll_x):
        # Dessiner le joueur avec décalage de caméra
        draw_rect = pygame.Rect(self.rect.x - scroll_x, self.rect.y, self.rect.width, self.rect.height)
        pygame.draw.rect(screen, COLORS["player"], draw_rect)

class Platform:
    def __init__(self, x, y, width):
        self.rect = pygame.Rect(x, y, width, 20)
    
    def draw(self, screen, scroll_x):
        # Dessiner la plateforme avec décalage de caméra
        draw_rect = pygame.Rect(self.rect.x - scroll_x, self.rect.y, self.rect.width, self.rect.height)
        pygame.draw.rect(screen, COLORS["platform"], draw_rect)

class Spike:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y - 20, 20, 20)
        self.points = [
            (x, y),  # Base gauche
            (x + 10, y - 20),  # Sommet
            (x + 20, y)  # Base droite
        ]
    
    def draw(self, screen, scroll_x):
        # Dessiner le spike avec décalage de caméra
        adjusted_points = [(x - scroll_x, y) for x, y in self.points]
        pygame.draw.polygon(screen, COLORS["obstacle"], adjusted_points)

class Block:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y - height, width, height)
    
    def draw(self, screen, scroll_x):
        # Dessiner le bloc avec décalage de caméra
        draw_rect = pygame.Rect(self.rect.x - scroll_x, self.rect.y, self.rect.width, self.rect.height)
        pygame.draw.rect(screen, COLORS["obstacle"], draw_rect)

# Vous pouvez ajouter d'autres types d'obstacles ici
class JumpPad:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y - 10, 30, 10)
        self.boost_force = JUMP_FORCE * 1.5
    
    def activate(self, player):
        player.velocity_y = self.boost_force
        player.on_ground = False
    
    def draw(self, screen, scroll_x):
        draw_rect = pygame.Rect(self.rect.x - scroll_x, self.rect.y, self.rect.width, self.rect.height)
        pygame.draw.rect(screen, (50, 200, 50), draw_rect)

class Portal:
    def __init__(self, x, y, effect_type):
        self.rect = pygame.Rect(x, y - 50, 20, 50)
        self.effect_type = effect_type  # "gravity", "size", "speed", etc.
    
    def activate(self, player):
        if self.effect_type == "gravity":
            # Inverser la gravité
            global GRAVITY
            GRAVITY = -GRAVITY
        elif self.effect_type == "speed":
            # Changer la vitesse
            player.speed *= 1.5
    
    def draw(self, screen, scroll_x):
        draw_rect = pygame.Rect(self.rect.x - scroll_x, self.rect.y, self.rect.width, self.rect.height)
        color = (100, 100, 255) if self.effect_type == "gravity" else (200, 100, 200)
        pygame.draw.rect(screen, color, draw_rect)
