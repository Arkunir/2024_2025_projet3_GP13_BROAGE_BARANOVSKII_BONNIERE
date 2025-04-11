import pygame
import sys
import random
from testiacode import ai_test_play
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
            
            # Ajout de la détection des collisions avec BouncingObstacle
            elif isinstance(obj, BouncingObstacle):
                if self.rect.colliderect(obj.rect):
                    self.is_alive = False
                    print("Game Over! Collision avec un obstacle rebondissant")
            
            # Détection des collisions avec les obstacles standards (pics)
            elif isinstance(obj, Obstacle) or isinstance(obj, DoublePikes) or isinstance(obj, TriplePikes) or isinstance(obj, QuadruplePikes):
                if hasattr(obj, 'get_rect'):
                    if self.rect.colliderect(obj.get_rect()):
                        self.is_alive = False
                        print("Game Over! Collision avec un obstacle")
                elif hasattr(obj, 'get_rects'):
                    for rect in obj.get_rects():
                        if self.rect.colliderect(rect):
                            self.is_alive = False
                            print("Game Over! Collision avec un obstacle")
                            break
        
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
        
    def get_rects(self):
        return [
            # Ceiling spikes rects
            pygame.Rect(self.x, self.y - CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            pygame.Rect(self.x + CUBE_SIZE, self.y - CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            pygame.Rect(self.x + CUBE_SIZE * 2, self.y - CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            pygame.Rect(self.x + CUBE_SIZE * 3, self.y - CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            # Pads rects
            pygame.Rect(self.x + CUBE_SIZE/2, self.y + CUBE_SIZE, CUBE_SIZE/2, CUBE_SIZE/2),
            pygame.Rect(self.x + CUBE_SIZE + CUBE_SIZE/2, self.y + CUBE_SIZE, CUBE_SIZE/2, CUBE_SIZE/2),
            pygame.Rect(self.x + CUBE_SIZE * 2 + CUBE_SIZE/2, self.y + CUBE_SIZE, CUBE_SIZE/2, CUBE_SIZE/2)
        ]

    def activate_pads(self, player):
        if player.standing_on and player.standing_on in [
            pygame.Rect(self.x + CUBE_SIZE/2, self.y + CUBE_SIZE, CUBE_SIZE/2, CUBE_SIZE/2),
            pygame.Rect(self.x + CUBE_SIZE + CUBE_SIZE/2, self.y + CUBE_SIZE, CUBE_SIZE/2, CUBE_SIZE/2),
            pygame.Rect(self.x + CUBE_SIZE * 2 + CUBE_SIZE/2, self.y + CUBE_SIZE, CUBE_SIZE/2, CUBE_SIZE/2)
        ]:
            player.velocity_y = -22
            player.is_jumping = True

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
    
class BouncingObstacle(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.y = GROUND_HEIGHT - 3 * CUBE_SIZE  # Position plus haute pour commencer
        self.width = CUBE_SIZE
        self.height = CUBE_SIZE
        self.bounce_velocity = 5  # Vitesse initiale du rebond
        self.bounce_gravity = 0.3  # Gravité appliquée à l'obstacle rebondissant
        self.rotation_angle = 0  # Angle de rotation pour l'effet visuel
        self.rotation_speed = 4  # Vitesse de rotation en degrés par frame
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        
    def update(self):
        # Déplacement horizontal
        self.x -= self.scroll_speed
        
        # Physique de rebond
        self.y += self.bounce_velocity
        self.bounce_velocity += self.bounce_gravity
        
        # Limiter les rebonds au sol
        if self.y > GROUND_HEIGHT - CUBE_SIZE:
            self.y = GROUND_HEIGHT - CUBE_SIZE
            self.bounce_velocity = -abs(self.bounce_velocity) * 0.7  # Rebond avec perte d'énergie
        
        # Rotation pour effet visuel
        self.rotation_angle = (self.rotation_angle + self.rotation_speed) % 360
        
        # Mise à jour du rectangle de collision
        self.rect.x = self.x
        self.rect.y = self.y
        
    def draw(self):
        # Sauvegarder la surface originale
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Dessiner un motif à la fois dangereux et distinct
        pygame.draw.rect(surface, RED, (0, 0, self.width, self.height))
        pygame.draw.line(surface, WHITE, (0, 0), (self.width, self.height), 3)
        pygame.draw.line(surface, WHITE, (0, self.height), (self.width, 0), 3)
        
        # Rotation de l'obstacle
        rotated_surface = pygame.transform.rotate(surface, self.rotation_angle)
        rotated_rect = rotated_surface.get_rect(center=(self.x + self.width/2, self.y + self.height/2))
        
        # Affichage sur l'écran
        screen.blit(rotated_surface, rotated_rect)
        
    def get_rect(self):
        # Rectangle de collision
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def get_rects(self):
        # Pour compatibilité avec le système de collision du Player qui peut attendre get_rects()
        return [self.get_rect()]
    
    def check_collision(self, player):
        # Vérifier si le joueur entre en collision avec cet obstacle
        if player.rect.colliderect(self.rect):
            player.is_alive = False
            print("Game Over! Collision avec un obstacle rebondissant")
            return True
        return False