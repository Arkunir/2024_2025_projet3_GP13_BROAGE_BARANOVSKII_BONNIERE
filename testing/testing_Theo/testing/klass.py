import pygame
import sys
import random
import os

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
        
        # Charger les skins disponibles
        self.skins = self.load_skins()
        # Choisir un skin aléatoire dès le début
        self.current_skin_index = random.randint(0, len(self.skins) - 1) if self.skins else 0
        self.image = self.skins[self.current_skin_index] if self.skins else None
        print(f"Skin aléatoire choisi au démarrage: index {self.current_skin_index}")
        
        # Ajouter des attributs pour la rotation
        self.rotation_angle = 0
        self.rotation_speed = -9  # Degrés par frame
        self.total_rotation = 0  # Pour suivre la rotation totale
        self.is_rotating = False  # Pour savoir si le joueur est en train de tourner
    
    # Ajout des propriétés pour accéder facilement aux coordonnées
    @property
    def x(self):
        return self.rect.x
        
    @property
    def y(self):
        return self.rect.y
        
    @property
    def width(self):
        return self.rect.width
        
    @property
    def height(self):
        return self.rect.height
    
    def load_skins(self):
        skins = []
        skins_dir = "skins"  # Dossier contenant les images des skins
        
        # Vérifier si le dossier existe
        if not os.path.exists(skins_dir):
            os.makedirs(skins_dir)
            print(f"Dossier {skins_dir} créé. Veuillez y ajouter des images de skins.")
            return skins
            
        # Charger toutes les images du dossier skins
        for filename in os.listdir(skins_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(skins_dir, filename)
                try:
                    # Charger l'image
                    image = pygame.image.load(path).convert_alpha()
                    # Redimensionner à une taille plus grande que CUBE_SIZE (4x)
                    new_size = int(CUBE_SIZE * 4)
                    image = pygame.transform.scale(image, (new_size, new_size))
                    skins.append(image)
                    print(f"Skin chargé: {filename}")
                except pygame.error:
                    print(f"Impossible de charger l'image: {filename}")
        
        # Si aucun skin n'est trouvé, ajouter un skin par défaut (un carré bleu)
        if not skins:
            print("Aucun skin trouvé, utilisation du skin par défaut")
            default_size = int(CUBE_SIZE * 4)
            default_skin = pygame.Surface((default_size, default_size), pygame.SRCALPHA)
            pygame.draw.rect(default_skin, BLUE, (0, 0, default_size, default_size))
            skins.append(default_skin)
            
        return skins
    
    def change_skin_randomly(self):
        if len(self.skins) > 1:
            # Choisir un skin différent du skin actuel
            new_index = self.current_skin_index
            while new_index == self.current_skin_index:
                new_index = random.randint(0, len(self.skins) - 1)
            self.current_skin_index = new_index
            self.image = self.skins[self.current_skin_index]
            print(f"Skin changé pour l'index {self.current_skin_index}")

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
            
            # Activer la rotation uniquement quand le joueur est en l'air
            if not self.is_rotating and self.is_jumping:
                self.is_rotating = True
                self.total_rotation = 0
        
        # Arrêter la rotation si le joueur touche le sol
        if (self.standing_on is not None or on_ground) and self.is_rotating:
            self.is_rotating = False
            self.rotation_angle = 0
            self.total_rotation = 0
            
        # Mettre à jour la rotation si le joueur est en l'air
        if self.is_rotating and self.is_jumping:
            self.total_rotation += self.rotation_speed
            self.rotation_angle = self.total_rotation % 360
            
        # Conserve l'état précédent pour la frame suivante
        self.was_on_ground = self.standing_on is not None or on_ground
            
    def jump(self):
        if not self.is_jumping:
            self.velocity_y = JUMP_STRENGTH
            self.is_jumping = True
            # Initialiser la rotation au début du saut
            self.is_rotating = True
            self.total_rotation = 0
            
    def draw(self, screen):
        if self.image:
            # Créer une copie de l'image pour la rotation
            image_to_draw = self.image
            
            # Appliquer la rotation si le joueur est en l'air
            if self.is_rotating and self.is_jumping:
                image_to_draw = pygame.transform.rotate(self.image, self.rotation_angle)
            
            # Calculer le décalage pour centrer l'image plus grande sur la hitbox
            image_width = image_to_draw.get_width()
            image_height = image_to_draw.get_height()
            offset_x = (image_width - CUBE_SIZE) // 2
            offset_y = (image_height - CUBE_SIZE) // 2
            
            # Dessiner l'image centrée sur la hitbox du joueur
            screen.blit(image_to_draw, (self.rect.x - offset_x, self.rect.y - offset_y))
        else:
            # Fallback sur le rectangle bleu original
            if self.is_rotating and self.is_jumping:
                # Créer une surface pour le cube
                surface = pygame.Surface((CUBE_SIZE, CUBE_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(surface, BLUE, (0, 0, CUBE_SIZE, CUBE_SIZE))
                
                # Appliquer la rotation
                rotated_surface = pygame.transform.rotate(surface, self.rotation_angle)
                rotated_rect = rotated_surface.get_rect(center=(self.rect.x + CUBE_SIZE/2, self.rect.y + CUBE_SIZE/2))
                
                # Afficher sur l'écran
                screen.blit(rotated_surface, rotated_rect)
            else:
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

class FivePikesWithOrb(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.y = GROUND_HEIGHT - CUBE_SIZE
        self.width = CUBE_SIZE * 5  # Ajusté pour 5 pics
        self.height = CUBE_SIZE
        self.orb_activated = False
        self.orb_active_timer = 0
        self.orb_pulse_size = 0
        self.orb_pulse_alpha = 0
        
    def update(self):
        self.x -= self.scroll_speed
        
        # Gestion de l'animation d'activation de l'orbe
        if self.orb_activated:
            self.orb_active_timer += 1
            self.orb_pulse_size += 2
            self.orb_pulse_alpha = max(0, 255 - self.orb_pulse_size * 5)
            
            # Réinitialiser après l'animation
            if self.orb_active_timer > 30:
                self.orb_activated = False
                self.orb_active_timer = 0
                self.orb_pulse_size = 0
                self.orb_pulse_alpha = 0
        
    def draw(self, screen):
        # Dessiner les 5 pics au sol
        for i in range(5):
            pygame.draw.polygon(screen, RED, [
                (self.x + CUBE_SIZE * i, self.y + self.height),
                (self.x + CUBE_SIZE * (i+1), self.y + self.height),
                (self.x + CUBE_SIZE * (i+0.5), self.y)
            ])
        
        # Position de l'orbe: 2 blocs au-dessus du pic central (3ème pic)
        orb_x = self.x + CUBE_SIZE * 2.5 - CUBE_SIZE // 4  # Centré sur le 3ème pic (indice 2)
        orb_y = self.y - CUBE_SIZE * 2  # 2 blocs au-dessus
        orb_width = CUBE_SIZE // 2
        orb_height = CUBE_SIZE // 2
        
        # Dessiner l'effet de pulsation si activé
        if self.orb_activated:
            pulse_surface = pygame.Surface((orb_width + self.orb_pulse_size, orb_height + self.orb_pulse_size), pygame.SRCALPHA)
            pulse_color = (255, 255, 0, self.orb_pulse_alpha)  # Jaune avec transparence
            pygame.draw.circle(pulse_surface, pulse_color, 
                              (pulse_surface.get_width() // 2, pulse_surface.get_height() // 2), 
                              (orb_width + self.orb_pulse_size) // 2)
            screen.blit(pulse_surface, 
                       (orb_x - self.orb_pulse_size // 2, orb_y - self.orb_pulse_size // 2))
        
        # Dessiner l'orbe jaune
        pygame.draw.circle(screen, (255, 255, 0), 
                          (orb_x + orb_width // 2, orb_y + orb_height // 2), 
                          orb_width // 2)
        
        # Ajouter un petit détail à l'intérieur de l'orbe
        inner_color = (200, 200, 0) if not self.orb_activated else (255, 255, 255)
        pygame.draw.circle(screen, inner_color, 
                          (orb_x + orb_width // 2, orb_y + orb_height // 2), 
                          orb_width // 4)
        
    def get_rects(self):
        # Hitboxes pour les 5 pics
        hitboxes = []
        spike_width = self.width / 5  # Largeur d'un pic
        reduced_size = spike_width * 0.6  # Réduire à 60% de la taille
    
        # Pour chaque pic, créer une hitbox inscrite
        for i in range(5):
            x_offset = (spike_width - reduced_size) / 2
            y_offset = self.height - reduced_size * 0.8
        
            hitboxes.append(pygame.Rect(
                self.x + i * spike_width + x_offset,
                self.y + y_offset,
                reduced_size,
                reduced_size * 0.7
            ))
        
        # Ajouter la hitbox de l'orbe (au-dessus du 3ème pic)
        orb_width = CUBE_SIZE // 2
        orb_x = self.x + CUBE_SIZE * 2.5 - orb_width // 2  # Centré sur le 3ème pic
        orb_y = self.y - CUBE_SIZE * 2  # 2 blocs au-dessus
        
        orb_rect = pygame.Rect(orb_x, orb_y, orb_width, orb_width)
        hitboxes.append(orb_rect)
        
        return hitboxes
    
    def check_activation(self, player, keys):
        # Obtenir la hitbox de l'orbe (le 6ème rectangle dans la liste)
        orb_rect = self.get_rects()[5]  # Indice 5 car il y a 5 pics + 1 orbe
        
        # Vérifier si le joueur est en contact avec l'orbe et appuie sur espace
        if not self.orb_activated and player.rect.colliderect(orb_rect) and keys[pygame.K_SPACE]:
            self.orb_activated = True
            player.velocity_y = JUMP_STRENGTH * 1  # Saut aussi puissant que le saut normal
            player.is_jumping = True
            return True
        return False

class PurpleOrb(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.y = GROUND_HEIGHT - CUBE_SIZE
        self.width = CUBE_SIZE * 2  # Largeur totale pour l'objet
        self.height = CUBE_SIZE
        self.orb_activated = False
        self.orb_active_timer = 0
        self.orb_pulse_size = 0
        self.orb_pulse_alpha = 0
        
    def update(self):
        self.x -= self.scroll_speed
        
        # Gestion de l'animation d'activation de l'orbe
        if self.orb_activated:
            self.orb_active_timer += 1
            self.orb_pulse_size += 2
            self.orb_pulse_alpha = max(0, 255 - self.orb_pulse_size * 5)
            
            # Réinitialiser après l'animation
            if self.orb_active_timer > 30:
                self.orb_activated = False
                self.orb_active_timer = 0
                self.orb_pulse_size = 0
                self.orb_pulse_alpha = 0
        
    def draw(self, screen):
        # Position de l'orbe: même hauteur que l'orbe jaune (2 blocs au-dessus du sol)
        orb_x = self.x + CUBE_SIZE - CUBE_SIZE // 4  # Centré sur l'obstacle
        orb_y = self.y - CUBE_SIZE * 2  # 2 blocs au-dessus (comme l'orbe jaune)
        orb_width = CUBE_SIZE // 2
        orb_height = CUBE_SIZE // 2
        
        # Dessiner l'effet de pulsation si activé
        if self.orb_activated:
            pulse_surface = pygame.Surface((orb_width + self.orb_pulse_size, orb_height + self.orb_pulse_size), pygame.SRCALPHA)
            pulse_color = (160, 32, 240, self.orb_pulse_alpha)  # Violet avec transparence
            pygame.draw.circle(pulse_surface, pulse_color, 
                              (pulse_surface.get_width() // 2, pulse_surface.get_height() // 2), 
                              (orb_width + self.orb_pulse_size) // 2)
            screen.blit(pulse_surface, 
                       (orb_x - self.orb_pulse_size // 2, orb_y - self.orb_pulse_size // 2))
        
        # Dessiner l'orbe violette
        pygame.draw.circle(screen, (160, 32, 240), 
                          (orb_x + orb_width // 2, orb_y + orb_height // 2), 
                          orb_width // 2)
        
        # Ajouter un petit détail à l'intérieur de l'orbe
        inner_color = (120, 24, 180) if not self.orb_activated else (255, 255, 255)
        pygame.draw.circle(screen, inner_color, 
                          (orb_x + orb_width // 2, orb_y + orb_height // 2), 
                          orb_width // 4)
        
    def get_rects(self):
        # Hitbox pour l'orbe (ajuster également la hitbox)
        orb_width = CUBE_SIZE // 2
        orb_x = self.x + CUBE_SIZE - orb_width // 4
        orb_y = self.y - CUBE_SIZE * 2  # Mise à jour de la position Y de la hitbox
        
        orb_rect = pygame.Rect(orb_x, orb_y, orb_width, orb_width)
        return [orb_rect]
    
    def check_activation(self, player, keys):
        # Obtenir la hitbox de l'orbe
        orb_rect = self.get_rects()[0]
        
        # Vérifier si le joueur est en contact avec l'orbe et appuie sur espace
        if not self.orb_activated and player.rect.colliderect(orb_rect) and keys[pygame.K_SPACE]:
            self.orb_activated = True
            player.velocity_y = JUMP_STRENGTH * 0.8  # Saut à 80% de la hauteur normale
            player.is_jumping = True
            return True
        return False

class JumpPad(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.y = GROUND_HEIGHT - CUBE_SIZE // 3  # Un tiers de la hauteur d'un cube
        self.width = CUBE_SIZE
        self.height = CUBE_SIZE // 3
        self.boost_strength = -25
        self.activated = False
        self.activation_timer = 0
        self.activation_duration = 10
        
    def update(self):
        self.x -= self.scroll_speed
        
        # Gestion de l'animation d'activation
        if self.activated:
            self.activation_timer += 1
            if self.activation_timer >= self.activation_duration:
                self.activated = False
                self.activation_timer = 0
    
    def draw(self, screen):
        # Simple rectangle jaune
        base_color = (255, 255, 0)  # Jaune
        
        # Si activé, changer brièvement la couleur
        if self.activated:
            base_color = (255, 255, 150)  # Jaune plus clair pendant l'activation
        
        # Dessiner un simple rectangle jaune
        pygame.draw.rect(screen, base_color, pygame.Rect(
            self.x, self.y, self.width, self.height))
        
        # Effet de pulsation original
        if self.activated:
            # Créer un effet de pulsation comme dans la version originale
            pulse_height = (self.activation_timer / self.activation_duration) * CUBE_SIZE * 2
            pulse_alpha = max(0, 255 - (pulse_height * 2))
            
            # Surface avec transparence pour l'effet
            pulse_surface = pygame.Surface((self.width, pulse_height), pygame.SRCALPHA)
            pulse_color = (255, 255, 0, pulse_alpha)  # Jaune avec transparence
            
            # Dessiner l'effet sur la surface
            pygame.draw.polygon(pulse_surface, pulse_color, [
                (0, pulse_height),
                (self.width, pulse_height),
                (self.width//2, 0)
            ])
            
            # Afficher l'effet au-dessus du pad
            screen.blit(pulse_surface, (self.x, self.y - pulse_height))
    
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def activate(self, player):
        if not self.activated:
            self.activated = True
            player.velocity_y = self.boost_strength
            player.is_jumping = True
            return True
        return False

class QuintuplePikesWithJumpPad(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.y = GROUND_HEIGHT - CUBE_SIZE
        self.width = CUBE_SIZE * 5
        self.height = CUBE_SIZE * 4  # Hauteur totale
        self.jump_pad_activated = False
        self.activation_timer = 0
        self.activation_duration = 10
        self.boost_strength = -25  # Force du saut
        
    def update(self):
        self.x -= self.scroll_speed
        
        # Gestion de l'animation d'activation du jumppad
        if self.jump_pad_activated:
            self.activation_timer += 1
            if self.activation_timer >= self.activation_duration:
                self.jump_pad_activated = False
                self.activation_timer = 0
        
    def draw(self, screen):
        # Position des blocs et des pics à 4 blocs du sol
        spike_y = self.y - CUBE_SIZE * 4
        
        # D'abord dessiner les blocs
        for i in range(5):
            pygame.draw.rect(screen, BLACK, pygame.Rect(
                self.x + i * CUBE_SIZE, 
                spike_y - CUBE_SIZE,  # Positionnés juste au-dessus des pics
                CUBE_SIZE, 
                CUBE_SIZE
            ))
        
        # Ensuite dessiner les pics orientés vers le bas
        for i in range(5):
            pygame.draw.polygon(screen, RED, [
                (self.x + i * CUBE_SIZE, spike_y),  # Point en haut à gauche
                (self.x + (i+1) * CUBE_SIZE, spike_y),  # Point en haut à droite
                (self.x + (i+0.5) * CUBE_SIZE, spike_y + CUBE_SIZE)  # Point en bas (pointe)
            ])
        
        # Dessiner le jumppad sous le 3ème pic
        pad_x = self.x + CUBE_SIZE * 2  # Position du 3ème pic
        pad_y = GROUND_HEIGHT - CUBE_SIZE // 3  # Hauteur du jumppad
        pad_width = CUBE_SIZE
        pad_height = CUBE_SIZE // 3
        
        # Couleur du jumppad (jaune ou jaune clair si activé)
        pad_color = (255, 255, 150) if self.jump_pad_activated else (255, 255, 0)
        
        # Dessiner le jumppad
        pygame.draw.rect(screen, pad_color, pygame.Rect(
            pad_x, pad_y, pad_width, pad_height))
        
        # Effet d'activation du jumppad
        if self.jump_pad_activated:
            # Effet de pulsation
            pulse_height = (self.activation_timer / self.activation_duration) * CUBE_SIZE * 2
            pulse_alpha = max(0, 255 - (pulse_height * 2))
            
            # Surface avec transparence pour l'effet
            pulse_surface = pygame.Surface((pad_width, pulse_height), pygame.SRCALPHA)
            pulse_color = (255, 255, 0, pulse_alpha)  # Jaune avec transparence
            
            # Dessiner l'effet sur la surface
            pygame.draw.polygon(pulse_surface, pulse_color, [
                (0, pulse_height),
                (pad_width, pulse_height),
                (pad_width//2, 0)
            ])
            
            # Afficher l'effet au-dessus du pad
            screen.blit(pulse_surface, (pad_x, pad_y - pulse_height))
    
    def get_rects(self):
        hitboxes = []
        spike_width = CUBE_SIZE
        reduced_size = spike_width * 0.6  # Réduire à 60% de la taille
        spike_y = self.y - CUBE_SIZE * 4
        
        # Ajouter des hitboxes pour les blocs
        for i in range(5):
            hitboxes.append(pygame.Rect(
                self.x + i * CUBE_SIZE,
                spike_y - CUBE_SIZE,  # Position des blocs
                CUBE_SIZE,
                CUBE_SIZE
            ))
        
        # Hitbox pour chaque pic
        for i in range(5):
            x_offset = (spike_width - reduced_size) / 2
            y_offset = CUBE_SIZE * 0.3
            
            hitboxes.append(pygame.Rect(
                self.x + i * spike_width + x_offset,
                spike_y + y_offset,
                reduced_size,
                reduced_size * 0.7
            ))
        
        # Hitbox pour le jumppad
        hitboxes.append(pygame.Rect(
            self.x + CUBE_SIZE * 2,
            GROUND_HEIGHT - CUBE_SIZE // 3,
            CUBE_SIZE,
            CUBE_SIZE // 3
        ))
        
        return hitboxes
    
    def check_collision(self, player):
        rects = self.get_rects()
        
        # Vérifier la collision avec le jumppad (dernier rectangle)
        jump_pad_rect = rects[-1]
        if player.rect.colliderect(jump_pad_rect):
            self.activate_jump_pad(player)
            return False  # Pas une collision mortelle
        
        # Vérifier les collisions avec les pics (5 rectangles après les 5 blocs)
        for i in range(5, 10):  # Indices 5 à 9 sont les pics
            if player.rect.colliderect(rects[i]):
                return True  # Collision mortelle
        
        # Si le joueur touche un bloc, c'est une surface sur laquelle il peut se tenir
        # Nous n'avons pas besoin de traiter ces collisions comme mortelles
                
        return False  # Pas de collision mortelle
            
    def activate_jump_pad(self, player):
        if not self.jump_pad_activated:
            self.jump_pad_activated = True
            player.velocity_y = self.boost_strength
            player.is_jumping = True
            return True
        return False
        
    def check_collision(self, player):
        rects = self.get_rects()
        
        # Vérifier la collision avec le jumppad (dernier rectangle)
        jump_pad_rect = rects[-1]
        if player.rect.colliderect(jump_pad_rect):
            self.activate_jump_pad(player)
            return False  # Pas une collision mortelle
        
        # Vérifier les collisions avec les pics (mortelles)
        for i in range(5):  # Les 5 premiers rectangles sont les pics
            if player.rect.colliderect(rects[i]):
                return True  # Collision mortelle
                
        return False  # Pas de collision

class JumppadOrbsObstacle(MovingObject):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.width = CUBE_SIZE * 20  # Maintenu à 20 pour garder la largeur totale
        self.height = CUBE_SIZE * 5   # Hauteur pour les calculs de collision
        
        # Créer une instance de JumpPad
        jumppad_pos = self.adjust_position_for_speed(1)
        self.jumppad = JumpPad(x + CUBE_SIZE * jumppad_pos)
        
        # États des orbes (séparés)
        self.yellow_orb1_activated = False
        self.yellow_orb2_activated = False
        self.purple_orb_activated = False
        
        # Timers et paramètres d'animation pour chaque orbe
        self.yellow_orb1_timer = 0
        self.yellow_orb1_pulse_size = 0
        self.yellow_orb1_pulse_alpha = 0
        
        self.yellow_orb2_timer = 0
        self.yellow_orb2_pulse_size = 0
        self.yellow_orb2_pulse_alpha = 0
        
        self.purple_orb_timer = 0
        self.purple_orb_pulse_size = 0
        self.purple_orb_pulse_alpha = 0
        
        # Orbes: force du saut
        self.yellow_orb_boost = JUMP_STRENGTH * 1.2
        self.purple_orb_boost = JUMP_STRENGTH * 0.8
    
    def adjust_position_for_speed(self, position):
        # Fonction utilitaire pour ajuster les positions selon la vitesse
        if self.scroll_speed == 9:
            return round(position * 9/8)
        else:
            return position
    
    def adjust_yellow_orb2_position(self, position):
        # Fonction spéciale pour la deuxième orbe jaune
        if self.scroll_speed == 9:
            # On décale d'un bloc vers la gauche (en soustrayant 1) avant d'appliquer le facteur 9/8
            return round((position - 1) * 9/8)
        else:
            return position
        
    def update(self):
        self.x -= self.scroll_speed
        jumppad_pos = self.adjust_position_for_speed(1)
        self.jumppad.x = self.x + CUBE_SIZE * jumppad_pos
        self.jumppad.update()  # Mettre à jour le jumppad
        
        # Gérer les animations de chaque orbe séparément
        self.update_orb_animation('yellow_orb1')
        self.update_orb_animation('yellow_orb2')
        self.update_orb_animation('purple_orb')
        
    def update_orb_animation(self, orb_name):
        # Vérifie si l'orbe est activée
        is_activated = getattr(self, f"{orb_name}_activated")
        if is_activated:
            # Incrémente le timer
            timer = getattr(self, f"{orb_name}_timer") + 1
            setattr(self, f"{orb_name}_timer", timer)
            
            # Mettre à jour les effets visuels
            pulse_size = getattr(self, f"{orb_name}_pulse_size") + 2
            setattr(self, f"{orb_name}_pulse_size", pulse_size)
            
            pulse_alpha = max(0, 255 - pulse_size * 5)
            setattr(self, f"{orb_name}_pulse_alpha", pulse_alpha)
            
            # Réinitialiser après l'animation
            if timer > 30:
                setattr(self, f"{orb_name}_activated", False)
                setattr(self, f"{orb_name}_timer", 0)
                setattr(self, f"{orb_name}_pulse_size", 0)
                setattr(self, f"{orb_name}_pulse_alpha", 0)
        
    def draw(self, screen):
        # 1. Dessiner le jumppad
        self.jumppad.draw(screen)
        
        # 2. Dessiner les pics sur le sol (toute la longueur sauf où se trouve le jumppad)
        # Note: Les positions des pics ne sont pas modifiées selon les instructions
        for i in range(18):  # 18 pics comme avant
            # Sauter le segment où se trouve le jumppad
            if i == 0 or i == 1:
                continue
                
            pygame.draw.polygon(screen, RED, [
                (self.x + i * CUBE_SIZE, GROUND_HEIGHT),
                (self.x + (i+1) * CUBE_SIZE, GROUND_HEIGHT),
                (self.x + (i+0.5) * CUBE_SIZE, GROUND_HEIGHT - CUBE_SIZE)
            ])
        
        # 3. Dessiner les pics verticaux en l'air avec des blocs à leur droite
        # Note: Les positions des pics ne sont pas modifiées selon les instructions
        air_spikes_x = self.x + CUBE_SIZE * 18  # Position inchangée à 18
        block_x = air_spikes_x + CUBE_SIZE  # Position des blocs à droite des pics
        
        # Dessiner 3 pics empilés verticalement avec des pointes vers la gauche
        for i in range(3):
            spike_y = GROUND_HEIGHT - CUBE_SIZE * (7 + i)
            
            # Dessiner le pic
            pygame.draw.polygon(screen, RED, [
                (air_spikes_x, spike_y),  # Point gauche
                (air_spikes_x + CUBE_SIZE, spike_y - CUBE_SIZE/2),  # Point haut
                (air_spikes_x + CUBE_SIZE, spike_y + CUBE_SIZE/2)   # Point bas
            ])
            
            # Dessiner le bloc à droite du pic (maintenant en noir)
            pygame.draw.rect(screen, (0, 0, 0), [  # Bloc noir au lieu de bleu
                block_x,
                spike_y - CUBE_SIZE/2,  # Aligner avec le centre du pic
                CUBE_SIZE,
                CUBE_SIZE
            ])
        
        # 4. Dessiner les orbes avec positions ajustées selon la vitesse
        # Première orbe jaune - Position ajustée si nécessaire
        yellow_orb1_pos = self.adjust_position_for_speed(8)
        yellow_orb1_height = self.adjust_position_for_speed(3)
        yellow_orb1_x = self.x + CUBE_SIZE * yellow_orb1_pos
        yellow_orb1_y = GROUND_HEIGHT - CUBE_SIZE * yellow_orb1_height
        orb_width = CUBE_SIZE // 2
        
        # Effet de pulsation de la première orbe jaune
        if self.yellow_orb1_activated:
            self.draw_specific_orb_pulse(screen, yellow_orb1_x, yellow_orb1_y, orb_width, 
                                        (255, 255, 0), self.yellow_orb1_pulse_size, self.yellow_orb1_pulse_alpha)
        
        # Dessiner la première orbe jaune
        pygame.draw.circle(screen, (255, 255, 0), 
                          (yellow_orb1_x, yellow_orb1_y), 
                          orb_width // 2)
        inner_color1 = (200, 200, 0) if not self.yellow_orb1_activated else (255, 255, 255)
        pygame.draw.circle(screen, inner_color1, 
                          (yellow_orb1_x, yellow_orb1_y), 
                          orb_width // 4)
        
        # Deuxième orbe jaune - Position ajustée si nécessaire avec décalage spécial
        yellow_orb2_pos = self.adjust_yellow_orb2_position(13)
        yellow_orb2_height = self.adjust_position_for_speed(7)
        yellow_orb2_x = self.x + CUBE_SIZE * yellow_orb2_pos
        yellow_orb2_y = GROUND_HEIGHT - CUBE_SIZE * yellow_orb2_height
        
        # Effet de pulsation de la deuxième orbe jaune
        if self.yellow_orb2_activated:
            self.draw_specific_orb_pulse(screen, yellow_orb2_x, yellow_orb2_y, orb_width, 
                                        (255, 255, 0), self.yellow_orb2_pulse_size, self.yellow_orb2_pulse_alpha)
            
        # Dessiner la deuxième orbe jaune
        pygame.draw.circle(screen, (255, 255, 0), 
                          (yellow_orb2_x, yellow_orb2_y), 
                          orb_width // 2)
        inner_color2 = (200, 200, 0) if not self.yellow_orb2_activated else (255, 255, 255)
        pygame.draw.circle(screen, inner_color2, 
                          (yellow_orb2_x, yellow_orb2_y), 
                          orb_width // 4)
        
        # Orbe violette (chemin correct) - Position ajustée si nécessaire
        purple_orb_pos = self.adjust_position_for_speed(14)
        purple_orb_height = self.adjust_position_for_speed(3)
        purple_orb_x = self.x + CUBE_SIZE * purple_orb_pos
        purple_orb_y = GROUND_HEIGHT - CUBE_SIZE * purple_orb_height
        
        # Effet de pulsation de l'orbe violette
        if self.purple_orb_activated:
            self.draw_specific_orb_pulse(screen, purple_orb_x, purple_orb_y, orb_width, 
                                        (160, 32, 240), self.purple_orb_pulse_size, self.purple_orb_pulse_alpha)
            
        # Dessiner l'orbe violette
        pygame.draw.circle(screen, (160, 32, 240), 
                          (purple_orb_x, purple_orb_y), 
                          orb_width // 2)
        inner_color3 = (120, 24, 180) if not self.purple_orb_activated else (255, 255, 255)
        pygame.draw.circle(screen, inner_color3, 
                          (purple_orb_x, purple_orb_y), 
                          orb_width // 4)
                          
    def draw_specific_orb_pulse(self, screen, x, y, orb_width, color, pulse_size, pulse_alpha):
        pulse_surface = pygame.Surface((orb_width + pulse_size, orb_width + pulse_size), pygame.SRCALPHA)
        r, g, b = color
        pulse_color = (r, g, b, pulse_alpha)
        pygame.draw.circle(pulse_surface, pulse_color, 
                          (pulse_surface.get_width() // 2, pulse_surface.get_height() // 2), 
                          (orb_width + pulse_size) // 2)
        screen.blit(pulse_surface, 
                   (x - orb_width//2 - pulse_size // 2, y - orb_width//2 - pulse_size // 2))
    
    def draw_orb_pulse(self, screen, x, y, orb_width, color):
        pass
    
    def get_rects(self):
        hitboxes = []
        orb_width = CUBE_SIZE // 2
        
        # 1. Hitbox du jumppad (utiliser celle de la classe JumpPad)
        jumppad_rect = self.jumppad.get_rect()
        hitboxes.append(jumppad_rect)
        
        # 2. Hitboxes des pics au sol (sauter le premier)
        # Note: Les positions des pics ne sont pas modifiées selon les instructions
        for i in range(18):  # Maintenu à 18 comme avant
            # Sauter le segment où se trouve le jumppad
            if i == 0 or i == 1:
                continue
                
            # Ajustement de la hitbox des pics pour être plus précise
            spike_width = CUBE_SIZE
            reduced_size = spike_width * 0.6
            x_offset = (spike_width - reduced_size) / 2
            y_offset = CUBE_SIZE * 0.2
            
            hitboxes.append(pygame.Rect(
                self.x + i * spike_width + x_offset,
                GROUND_HEIGHT - CUBE_SIZE + y_offset,
                reduced_size,
                reduced_size * 0.7
            ))
        
        # 3. Hitboxes des pics verticaux en l'air
        # Note: Les positions des pics ne sont pas modifiées selon les instructions
        air_spikes_x = self.x + CUBE_SIZE * 18  # Maintenu à 18 comme avant
        block_x = air_spikes_x + CUBE_SIZE  # Position des blocs à droite des pics
        
        # Hitboxes pour les 3 pics verticaux
        for i in range(3):
            spike_y = GROUND_HEIGHT - CUBE_SIZE * (7 + i)
            
            # Hitbox ajustée pour le pic horizontal pointant vers la gauche
            hitboxes.append(pygame.Rect(
                air_spikes_x + x_offset,
                spike_y - CUBE_SIZE/4,  # Centrer verticalement
                reduced_size * 0.7,
                reduced_size * 0.6
            ))
            
            # Hitbox pour le bloc à droite du pic
            hitboxes.append(pygame.Rect(
                block_x,
                spike_y - CUBE_SIZE/2,  # Aligner avec le centre du pic
                CUBE_SIZE,
                CUBE_SIZE
            ))
        
        # 4. Hitboxes des orbes avec positions ajustées selon la vitesse
        # Première orbe jaune - Position ajustée si nécessaire
        yellow_orb1_pos = self.adjust_position_for_speed(8)
        yellow_orb1_height = self.adjust_position_for_speed(3)
        yellow_orb1_x = self.x + CUBE_SIZE * yellow_orb1_pos - orb_width // 2
        yellow_orb1_y = GROUND_HEIGHT - CUBE_SIZE * yellow_orb1_height - orb_width // 2
        hitboxes.append(pygame.Rect(yellow_orb1_x, yellow_orb1_y, orb_width, orb_width))
        
        # Deuxième orbe jaune - Position ajustée si nécessaire avec décalage spécial
        yellow_orb2_pos = self.adjust_yellow_orb2_position(13)
        yellow_orb2_height = self.adjust_position_for_speed(7)
        yellow_orb2_x = self.x + CUBE_SIZE * yellow_orb2_pos - orb_width // 2
        yellow_orb2_y = GROUND_HEIGHT - CUBE_SIZE * yellow_orb2_height - orb_width // 2
        hitboxes.append(pygame.Rect(yellow_orb2_x, yellow_orb2_y, orb_width, orb_width))
        
        # Orbe violette - Position ajustée si nécessaire
        purple_orb_pos = self.adjust_position_for_speed(14)
        purple_orb_height = self.adjust_position_for_speed(3)
        purple_orb_x = self.x + CUBE_SIZE * purple_orb_pos - orb_width // 2
        purple_orb_y = GROUND_HEIGHT - CUBE_SIZE * purple_orb_height - orb_width // 2
        hitboxes.append(pygame.Rect(purple_orb_x, purple_orb_y, orb_width, orb_width))
        
        return hitboxes
    
    def check_collision(self, player, keys):
        rects = self.get_rects()
        
        # Vérifier la collision avec le jumppad (utiliser la méthode activate de JumpPad)
        jumppad_rect = rects[0]
        if player.rect.colliderect(jumppad_rect) and not player.is_jumping:
            self.jumppad.activate(player)
            return False  # Pas une collision mortelle
        
        # Nombre de pics au sol (18 - 2 pour le jumppad = 16)
        ground_spikes_count = 16
        
        # Nombre total de pics en l'air et leur blocs (3 pics + 3 blocs = 6)
        air_obstacles_count = 6
        
        # Vérifier les collisions avec les pics au sol
        for i in range(1, ground_spikes_count + 1):
            if player.rect.colliderect(rects[i]):
                player.is_alive = False
                print("Game Over! Collision avec un pic au sol")
                return True  # Collision mortelle
        
        # Indice de départ pour les obstacles en l'air
        air_obstacles_start = 1 + ground_spikes_count
        
        # Vérifier les collisions avec les pics en l'air et leurs blocs
        for i in range(air_obstacles_start, air_obstacles_start + air_obstacles_count):
            rect = rects[i]
            if player.rect.colliderect(rect):
                # Vérifier si c'est un bloc (indices pairs après le premier pic)
                if (i - air_obstacles_start) % 2 == 1:  # Les blocs sont aux indices impairs
                    # Vérifier si le joueur atterrit sur le dessus du bloc
                    if (player.rect.bottom <= rect.top + 10 and 
                        player.velocity_y > 0):
                        player.rect.bottom = rect.top
                        player.velocity_y = 0
                        player.is_jumping = False
                        player.standing_on = self
                        return False
                    else:
                        player.is_alive = False
                        print("Game Over! Collision latérale avec un bloc en l'air")
                        return True
                else:  # C'est un pic
                    player.is_alive = False
                    print("Game Over! Collision avec un pic en l'air")
                    return True
        
        # Indice des orbes dans rects: dernières 3 positions
        orbs_start = 1 + ground_spikes_count + air_obstacles_count
        
        # Vérifier la collision avec la première orbe jaune
        if player.rect.colliderect(rects[orbs_start]) and keys[pygame.K_SPACE] and not self.yellow_orb1_activated:
            self.activate_yellow_orb1(player)
            return False
            
        # Vérifier la collision avec la deuxième orbe jaune
        if player.rect.colliderect(rects[orbs_start + 1]) and keys[pygame.K_SPACE] and not self.yellow_orb2_activated:
            self.activate_yellow_orb2(player)
            return False
            
        # Vérifier la collision avec l'orbe violette
        if player.rect.colliderect(rects[orbs_start + 2]) and keys[pygame.K_SPACE] and not self.purple_orb_activated:
            self.activate_purple_orb(player)
            return False
                
        return False  # Pas de collision
        
    def activate_yellow_orb1(self, player):
        self.yellow_orb1_activated = True
        player.velocity_y = self.yellow_orb_boost
        player.is_jumping = True
        
    def activate_yellow_orb2(self, player):
        self.yellow_orb2_activated = True
        player.velocity_y = self.yellow_orb_boost
        player.is_jumping = True
        
    def activate_purple_orb(self, player):
        self.purple_orb_activated = True
        player.velocity_y = self.purple_orb_boost
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