import pygame
import sys
import random

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

class Player:
    def __init__(self):
        self.rect = pygame.Rect(100, GROUND_HEIGHT - CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        self.velocity_y = 0
        self.is_jumping = False
        self.is_alive = True
        self.standing_on = None
        self.just_landed = False
        self.was_on_ground = True  # Pour suivre si le joueur était au sol à la frame précédente

    def update(self, game_objects):
        was_jumping = self.is_jumping
        previous_standing = self.standing_on is not None or self.rect.y >= GROUND_HEIGHT - CUBE_SIZE
        
        self.velocity_y += GRAVITY
        
        old_y = self.rect.y
        
        self.rect.y += self.velocity_y
        
        previous_standing_on = self.standing_on
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
            
            elif isinstance(obj, DoubleBlockPillar):
                # Gestion des collisions avec le pilier
                for pillar_rect in obj.get_rects():
                    if (old_y + CUBE_SIZE <= pillar_rect.top and 
                        self.rect.y + CUBE_SIZE >= pillar_rect.top and
                        self.rect.right > pillar_rect.left and
                        self.rect.left < pillar_rect.right):
                        
                        self.rect.bottom = pillar_rect.top
                        self.velocity_y = 0
                        
                        if self.is_jumping:
                            self.just_landed = True
                        
                        self.is_jumping = False
                        self.standing_on = obj
                        
                    elif (self.rect.colliderect(pillar_rect) and 
                          not (old_y + CUBE_SIZE <= pillar_rect.top)):
                        self.is_alive = False
                        print("Game Over! Collision latérale avec un pilier de blocs")
            
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
        
        on_ground = False
        if self.rect.y >= GROUND_HEIGHT - CUBE_SIZE and not self.standing_on:
            self.rect.y = GROUND_HEIGHT - CUBE_SIZE
            self.velocity_y = 0
            on_ground = True
            
            if self.is_jumping:
                self.just_landed = True
            
            self.is_jumping = False
            
        # Si le joueur n'est ni sur un objet ni sur le sol, il est considéré comme sautant
        if not self.standing_on and not on_ground:
            self.is_jumping = True
            
        # Conserve l'état précédent pour la frame suivante
        self.was_on_ground = self.standing_on is not None or on_ground
            
    def jump(self):
        if not self.is_jumping:
            self.velocity_y = JUMP_STRENGTH
            self.is_jumping = True
            
    def draw(self, screen):
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
        
    def draw(self, screen):
        pygame.draw.polygon(screen, RED, [
            (self.x, self.y + self.height),
            (self.x + self.width, self.y + self.height),
            (self.x + self.width/2, self.y)
        ])
    
    # Pour la classe Obstacle
    def get_rect(self):
        # Créer une hitbox inscrite dans le triangle au lieu d'utiliser tout le rectangle
        # Réduire la largeur et la hauteur et ajuster la position
        reduced_size = min(self.width, self.height) * 0.6  # Réduire à 60% de la taille
    
        # Centrer horizontalement et placer vers le haut du triangle
        x_offset = (self.width - reduced_size) / 2
        y_offset = self.height - reduced_size * 0.8  # Déplacer un peu vers le haut
    
        return pygame.Rect(
            self.x + x_offset,
            self.y + y_offset,
            reduced_size,
            reduced_size * 0.7  # Hitbox plus petite en hauteur
        )


class DoublePikes(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.y = GROUND_HEIGHT - CUBE_SIZE
        self.width = CUBE_SIZE * 2
        self.height = CUBE_SIZE
        
    def update(self):
        self.x -= self.scroll_speed
        
    def draw(self, screen):
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
        hitboxes = []
        spike_width = self.width / 2  # Largeur d'un pic
        reduced_size = spike_width * 0.6  # Réduire à 60% de la taille
    
        # Pour chaque pic, créer une hitbox inscrite
        for i in range(2):
            x_offset = (spike_width - reduced_size) / 2
            y_offset = self.height - reduced_size * 0.8
        
            hitboxes.append(pygame.Rect(
                self.x + i * spike_width + x_offset,
                self.y + y_offset,
                reduced_size,
                reduced_size * 0.7
            ))
        return hitboxes


class TriplePikes(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.y = GROUND_HEIGHT - CUBE_SIZE
        self.width = CUBE_SIZE * 3
        self.height = CUBE_SIZE
        
    def update(self):
        self.x -= self.scroll_speed
        
    def draw(self, screen):
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
        hitboxes = []
        spike_width = self.width / 3  # Largeur d'un pic
        reduced_size = spike_width * 0.6  # Réduire à 60% de la taille
    
        # Pour chaque pic, créer une hitbox inscrite
        for i in range(3):
            x_offset = (spike_width - reduced_size) / 2
            y_offset = self.height - reduced_size * 0.8
        
            hitboxes.append(pygame.Rect(
                self.x + i * spike_width + x_offset,
                self.y + y_offset,
                reduced_size,
                reduced_size * 0.7
            ))
        return hitboxes


class QuadruplePikes(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.y = GROUND_HEIGHT - CUBE_SIZE
        self.width = CUBE_SIZE * 4
        self.height = CUBE_SIZE
        
    def update(self):
        self.x -= self.scroll_speed
        
    def draw(self, screen):
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
        hitboxes = []
        spike_width = self.width / 4  # Largeur d'un pic
        reduced_size = spike_width * 0.6  # Réduire à 60% de la taille
    
        # Pour chaque pic, créer une hitbox inscrite
        for i in range(4):
            x_offset = (spike_width - reduced_size) / 2
            y_offset = self.height - reduced_size * 0.8
        
            hitboxes.append(pygame.Rect(
                self.x + i * spike_width + x_offset,
                self.y + y_offset,
                reduced_size,
                reduced_size * 0.7
            ))
        return hitboxes


class Block(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.rect = pygame.Rect(x, GROUND_HEIGHT - CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        
    def update(self):
        self.rect.x -= self.scroll_speed
        
    def draw(self, screen):
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
        
    def draw(self, screen):
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
        # Garder les rectangles des blocs tels quels
        block_rects = [
            pygame.Rect(self.x, self.y, CUBE_SIZE, self.height),
            pygame.Rect(self.x + CUBE_SIZE * 3, self.y, CUBE_SIZE, self.height)
        ]
    
        # Ajuster la hitbox du pic
        spike_width = CUBE_SIZE
        reduced_size = spike_width * 0.6
        x_offset = (spike_width - reduced_size) / 2
        y_offset = CUBE_SIZE * 0.2  # Le pic est orienté vers le haut, donc décalage différent
    
        spike_rect = pygame.Rect(
            self.x + CUBE_SIZE * 3 + x_offset,
            self.y - CUBE_SIZE + y_offset,
            reduced_size,
            reduced_size * 0.7
        )
    
        return [*block_rects, spike_rect]


        
    def activate_pads(self, player):
        pad_rects = [
            pygame.Rect(self.x + CUBE_SIZE/2, self.y + CUBE_SIZE, CUBE_SIZE/2, CUBE_SIZE/2),
            pygame.Rect(self.x + CUBE_SIZE + CUBE_SIZE/2, self.y + CUBE_SIZE, CUBE_SIZE/2, CUBE_SIZE/2),
            pygame.Rect(self.x + CUBE_SIZE * 2 + CUBE_SIZE/2, self.y + CUBE_SIZE, CUBE_SIZE/2, CUBE_SIZE/2)
        ]
        
        for pad_rect in pad_rects:
            if player.rect.colliderect(pad_rect):
                player.velocity_y = -22
                player.is_jumping = True
                break

class DoubleBlockPillar(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.width = CUBE_SIZE * 20  # Largeur totale mise à jour: 2 + 2 + 4 + 2 + 6 + 2 + 4
        
        # Liste pour stocker tous les rectangles de blocs
        self.block_rects = []
        
        # Premier pilier - 2 blocs de hauteur
        for i in range(2):
            self.block_rects.append(pygame.Rect(
                self.x, 
                GROUND_HEIGHT - CUBE_SIZE * (i + 1), 
                CUBE_SIZE, 
                CUBE_SIZE
            ))
        
        # Deuxième pilier - 4 blocs de hauteur (après 2 espaces vides)
        for i in range(4):
            self.block_rects.append(pygame.Rect(
                self.x + CUBE_SIZE * 3,  # Position x: 2 (premier pilier) + 1 (espaces)
                GROUND_HEIGHT - CUBE_SIZE * (i + 1), 
                CUBE_SIZE, 
                CUBE_SIZE
            ))
        
        # Troisième pilier - 6 blocs de hauteur (après 2 espaces vides supplémentaires)
        for i in range(6):
            self.block_rects.append(pygame.Rect(
                self.x + CUBE_SIZE * 6,  # Position x: 2 + 1 + 1 + 4 - 2
                GROUND_HEIGHT - CUBE_SIZE * (i + 1), 
                CUBE_SIZE, 
                CUBE_SIZE
            ))
        
        # Quatrième pilier - 4 blocs de hauteur (même que le 2ème pilier, après 2 espaces vides)
        for i in range(4):
            self.block_rects.append(pygame.Rect(
                self.x + CUBE_SIZE * 9,  # Position x: 2 + 1 + 1 + 4 - 2 + 1 + 6 - 3
                GROUND_HEIGHT - CUBE_SIZE * (i + 1), 
                CUBE_SIZE, 
                CUBE_SIZE
            ))
        
    def update(self, current_speed=INITIAL_SCROLL_SPEED):  # Ajout du paramètre current_speed avec valeur par défaut
        self.x -= self.scroll_speed
        # Mettre à jour les positions de tous les blocs
        for i, rect in enumerate(self.block_rects):
            if i < 2:  # Premier pilier (2 blocs)
                rect.x = self.x
            elif i < 6:  # Deuxième pilier (4 blocs)
                rect.x = self.x + CUBE_SIZE * (5 + current_speed - 6)
            elif i < 12:  # Troisième pilier (6 blocs)
                rect.x = self.x + CUBE_SIZE * (10 + current_speed - 6)
            else:  # Quatrième pilier (4 blocs)
                rect.x = self.x + CUBE_SIZE * (13 + current_speed - 5)
        
    def draw(self, screen):
        # Dessiner tous les blocs
        for rect in self.block_rects:
            pygame.draw.rect(screen, BLACK, rect)
    
    # Propriété pour la compatibilité avec le système de collision du joueur
    @property
    def rect(self):
        # On retourne le premier rectangle pour les collisions standards
        return self.block_rects[0] if self.block_rects else pygame.Rect(self.x, 0, 0, 0)
        
    def get_rects(self):
        return self.block_rects

class YellowOrb(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.y = GROUND_HEIGHT - CUBE_SIZE * 2  # Position au-dessus du sol
        self.width = CUBE_SIZE // 2
        self.height = CUBE_SIZE // 2
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.activated = False  # Pour éviter les activations multiples
        self.active_timer = 0   # Pour gérer l'animation d'activation
        self.pulse_size = 0     # Pour l'effet de pulsation
        self.pulse_alpha = 0    # Pour la transparence de la pulsation
        
    def update(self):
        self.x -= self.scroll_speed
        self.rect.x = self.x
        self.rect.y = self.y
        
        # Gestion de l'animation d'activation
        if self.activated:
            self.active_timer += 1
            self.pulse_size += 2
            self.pulse_alpha = max(0, 255 - self.pulse_size * 5)
            
            # Réinitialiser après l'animation
            if self.active_timer > 30:
                self.activated = False
                self.active_timer = 0
                self.pulse_size = 0
                self.pulse_alpha = 0
        
    def draw(self, screen):
        # Dessiner l'effet de pulsation si activé
        if self.activated:
            pulse_surface = pygame.Surface((self.width + self.pulse_size, self.height + self.pulse_size), pygame.SRCALPHA)
            pulse_color = (255, 255, 0, self.pulse_alpha)  # Jaune avec transparence
            pygame.draw.circle(pulse_surface, pulse_color, 
                              (pulse_surface.get_width() // 2, pulse_surface.get_height() // 2), 
                              (self.width + self.pulse_size) // 2)
            screen.blit(pulse_surface, 
                       (self.x - self.pulse_size // 2, self.y - self.pulse_size // 2))
        
        # Dessiner l'orbe jaune
        pygame.draw.circle(screen, (255, 255, 0), 
                          (self.x + self.width // 2, self.y + self.height // 2), 
                          self.width // 2)
        
        # Ajouter un petit détail à l'intérieur
        inner_color = (200, 200, 0) if not self.activated else (255, 255, 255)
        pygame.draw.circle(screen, inner_color, 
                          (self.x + self.width // 2, self.y + self.height // 2), 
                          self.width // 4)
    
    def check_activation(self, player, keys):
        # Vérifier si le joueur est en contact avec l'orbe et appuie sur espace
        if not self.activated and player.rect.colliderect(self.rect) and keys[pygame.K_SPACE]:
            self.activated = True
            player.velocity_y = JUMP_STRENGTH * 1.2  # Saut plus puissant que le saut normal
            player.is_jumping = True
            return True
        return False

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
        
    def draw(self, screen):
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
