import pygame
import sys
import random

# Initialisation de Pygame
pygame.init()

# Constantes
WIDTH, HEIGHT = 800, 600
FPS = 60
GRAVITY = 1
JUMP_STRENGTH = -18
GROUND_HEIGHT = 500
CUBE_SIZE = 50
INITIAL_SCROLL_SPEED = 6  # Vitesse initiale à 6

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Création de la fenêtre
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Geometry Dash Clone")
clock = pygame.time.Clock()

# Classe Joueur
class Player:
    def __init__(self):
        self.rect = pygame.Rect(100, GROUND_HEIGHT - CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        self.velocity_y = 0
        self.is_jumping = False
        self.is_alive = True
        self.standing_on = None  # Le bloc sur lequel le joueur se tient
        # Nouvelle variable pour détecter quand le joueur vient juste d'atterrir
        self.just_landed = False

    def update(self, game_objects):
        # Sauvegarder l'état de saut avant mise à jour
        was_jumping = self.is_jumping
        
        # Appliquer la gravité
        self.velocity_y += GRAVITY
        
        # Sauvegarder la position précédente pour la détection de collision
        old_y = self.rect.y
        
        # Mettre à jour la position verticale
        self.rect.y += self.velocity_y
        
        # Réinitialiser l'objet sur lequel le joueur se tient
        self.standing_on = None
        
        # Par défaut, on n'a pas juste atterri
        self.just_landed = False
        
        # Vérifier les collisions avec les objets du jeu
        for obj in game_objects:
            # Collisions avec les blocs normaux
            if isinstance(obj, Block):
                # Collision avec le dessus du bloc
                if (old_y + CUBE_SIZE <= obj.rect.top and 
                    self.rect.y + CUBE_SIZE >= obj.rect.top and
                    self.rect.right > obj.rect.left and
                    self.rect.left < obj.rect.right):
                    
                    self.rect.bottom = obj.rect.top
                    self.velocity_y = 0
                    
                    # Si on était en train de sauter et maintenant on ne l'est plus
                    if self.is_jumping:
                        self.just_landed = True
                    
                    self.is_jumping = False
                    self.standing_on = obj
                    
                # Collision latérale avec le bloc
                elif (self.rect.colliderect(obj.rect) and 
                      not (old_y + CUBE_SIZE <= obj.rect.top)):
                    self.is_alive = False
                    print("Game Over! Collision latérale avec un bloc")
                    
            # Collisions avec la structure spéciale (bloc-espace-bloc+pic)
            elif isinstance(obj, BlockGapBlockWithSpike):
                # Récupérer les rectangles des blocs (sans le pic)         
                bloc_rects = [obj.get_rects()[0], obj.get_rects()[1]]
                pic_rect = obj.get_rects()[2]
                
                # Vérifier la collision avec le pic
                if self.rect.colliderect(pic_rect):
                    self.is_alive = False
                    print("Game Over! Collision avec un pic")
                
                # Vérifier les collisions avec les blocs
                for bloc_rect in bloc_rects:
                    # Collision avec le dessus du bloc
                    if (old_y + CUBE_SIZE <= bloc_rect.top and 
                        self.rect.y + CUBE_SIZE >= bloc_rect.top and
                        self.rect.right > bloc_rect.left and
                        self.rect.left < bloc_rect.right):
                        
                        self.rect.bottom = bloc_rect.top
                        self.velocity_y = 0
                        
                        # Si on était en train de sauter et maintenant on ne l'est plus
                        if self.is_jumping:
                            self.just_landed = True
                        
                        self.is_jumping = False
                        self.standing_on = obj
                        
                    # Collision latérale avec le bloc
                    elif (self.rect.colliderect(bloc_rect) and 
                          not (old_y + CUBE_SIZE <= bloc_rect.top)):
                        self.is_alive = False
                        print("Game Over! Collision latérale avec un bloc de la structure")
        
        # Vérifier si le joueur touche le sol
        if self.rect.y >= GROUND_HEIGHT - CUBE_SIZE and not self.standing_on:
            self.rect.y = GROUND_HEIGHT - CUBE_SIZE
            self.velocity_y = 0
            
            # Si on était en train de sauter et maintenant on ne l'est plus
            if self.is_jumping:
                self.just_landed = True
            
            self.is_jumping = False
            
    def jump(self):
        if not self.is_jumping:
            self.velocity_y = JUMP_STRENGTH
            self.is_jumping = True
            
    def draw(self):
        pygame.draw.rect(screen, BLUE, self.rect)

# Classe de base pour les objets en mouvement
class MovingObject:
    def __init__(self, x):
        self.scroll_speed = INITIAL_SCROLL_SPEED
        
    def set_speed(self, speed):
        self.scroll_speed = speed

# Classe Obstacle (Triangle rouge - dangereux)
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
        # Dessiner un triangle rouge
        pygame.draw.polygon(screen, RED, [
            (self.x, self.y + self.height),  # Bas gauche
            (self.x + self.width, self.y + self.height),  # Bas droite
            (self.x + self.width/2, self.y)  # Sommet
        ])
        
    def get_rect(self):
        # Créer un rectangle pour la détection de collision
        return pygame.Rect(self.x, self.y, self.width, self.height)

# Classe pour une structure avec deux pics consécutifs
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
        # Dessiner deux triangles rouges consécutifs
        # Premier pic
        pygame.draw.polygon(screen, RED, [
            (self.x, self.y + self.height),  # Bas gauche
            (self.x + CUBE_SIZE, self.y + self.height),  # Bas droite
            (self.x + CUBE_SIZE/2, self.y)  # Sommet
        ])
        
        # Deuxième pic
        pygame.draw.polygon(screen, RED, [
            (self.x + CUBE_SIZE, self.y + self.height),  # Bas gauche
            (self.x + CUBE_SIZE*2, self.y + self.height),  # Bas droite
            (self.x + CUBE_SIZE*1.5, self.y)  # Sommet
        ])
        
    def get_rects(self):
        # Créer deux rectangles pour la détection de collision
        return [
            pygame.Rect(self.x, self.y, CUBE_SIZE, self.height),
            pygame.Rect(self.x + CUBE_SIZE, self.y, CUBE_SIZE, self.height)
        ]

# Classe pour une structure avec trois pics consécutifs
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
        # Dessiner trois triangles rouges consécutifs
        # Premier pic
        pygame.draw.polygon(screen, RED, [
            (self.x, self.y + self.height),  # Bas gauche
            (self.x + CUBE_SIZE, self.y + self.height),  # Bas droite
            (self.x + CUBE_SIZE/2, self.y)  # Sommet
        ])
        
        # Deuxième pic
        pygame.draw.polygon(screen, RED, [
            (self.x + CUBE_SIZE, self.y + self.height),  # Bas gauche
            (self.x + CUBE_SIZE*2, self.y + self.height),  # Bas droite
            (self.x + CUBE_SIZE*1.5, self.y)  # Sommet
        ])
        
        # Troisième pic
        pygame.draw.polygon(screen, RED, [
            (self.x + CUBE_SIZE*2, self.y + self.height),  # Bas gauche
            (self.x + CUBE_SIZE*3, self.y + self.height),  # Bas droite
            (self.x + CUBE_SIZE*2.5, self.y)  # Sommet
        ])
        
    def get_rects(self):
        # Créer trois rectangles pour la détection de collision
        return [
            pygame.Rect(self.x, self.y, CUBE_SIZE, self.height),
            pygame.Rect(self.x + CUBE_SIZE, self.y, CUBE_SIZE, self.height),
            pygame.Rect(self.x + CUBE_SIZE*2, self.y, CUBE_SIZE, self.height)
        ]

# Nouvelle classe pour une structure avec quatre pics consécutifs
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
        # Dessiner quatre triangles rouges consécutifs
        # Premier pic
        pygame.draw.polygon(screen, RED, [
            (self.x, self.y + self.height),  # Bas gauche
            (self.x + CUBE_SIZE, self.y + self.height),  # Bas droite
            (self.x + CUBE_SIZE/2, self.y)  # Sommet
        ])
        
        # Deuxième pic
        pygame.draw.polygon(screen, RED, [
            (self.x + CUBE_SIZE, self.y + self.height),  # Bas gauche
            (self.x + CUBE_SIZE*2, self.y + self.height),  # Bas droite
            (self.x + CUBE_SIZE*1.5, self.y)  # Sommet
        ])
        
        # Troisième pic
        pygame.draw.polygon(screen, RED, [
            (self.x + CUBE_SIZE*2, self.y + self.height),  # Bas gauche
            (self.x + CUBE_SIZE*3, self.y + self.height),  # Bas droite
            (self.x + CUBE_SIZE*2.5, self.y)  # Sommet
        ])
        
        # Quatrième pic
        pygame.draw.polygon(screen, RED, [
            (self.x + CUBE_SIZE*3, self.y + self.height),  # Bas gauche
            (self.x + CUBE_SIZE*4, self.y + self.height),  # Bas droite
            (self.x + CUBE_SIZE*3.5, self.y)  # Sommet
        ])
        
    def get_rects(self):
        # Créer quatre rectangles pour la détection de collision
        return [
            pygame.Rect(self.x, self.y, CUBE_SIZE, self.height),
            pygame.Rect(self.x + CUBE_SIZE, self.y, CUBE_SIZE, self.height),
            pygame.Rect(self.x + CUBE_SIZE*2, self.y, CUBE_SIZE, self.height),
            pygame.Rect(self.x + CUBE_SIZE*3, self.y, CUBE_SIZE, self.height)
        ]

# Classe Block (Blocs noirs - inoffensifs si touchés par le dessus)
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
        # Structure de 3 éléments: bloc, espace élargi (2 cubes), bloc+pic
        self.width = CUBE_SIZE * 4  # Augmenté de 3 à 4 pour tenir compte de l'espace supplémentaire
        self.height = CUBE_SIZE
        
    def update(self):
        self.x -= self.scroll_speed
        
    def draw(self):
        # Premier bloc
        pygame.draw.rect(screen, BLACK, pygame.Rect(
            self.x, self.y, CUBE_SIZE, CUBE_SIZE))
        
        # Deuxième bloc (à droite avec un espace de 2 cubes entre les deux)
        pygame.draw.rect(screen, BLACK, pygame.Rect(
            self.x + CUBE_SIZE * 3, self.y, CUBE_SIZE, CUBE_SIZE))  # Position modifiée de 2 à 3
        
        # Pic sur le deuxième bloc
        pygame.draw.polygon(screen, RED, [
            (self.x + CUBE_SIZE * 3, self.y),  # Position modifiée de 2 à 3
            (self.x + CUBE_SIZE * 4, self.y),  # Position modifiée de 3 à 4
            (self.x + CUBE_SIZE * 3.5, self.y - CUBE_SIZE)  # Position modifiée de 2.5 à 3.5
        ])
        
    def get_rects(self):
        # Créer trois rectangles pour la détection de collision
        return [
            # Premier bloc
            pygame.Rect(self.x, self.y, CUBE_SIZE, self.height),
            # Deuxième bloc
            pygame.Rect(self.x + CUBE_SIZE * 3, self.y, CUBE_SIZE, self.height),  # Position modifiée de 2 à 3
            # Pic sur le deuxième bloc
            pygame.Rect(self.x + CUBE_SIZE * 3, self.y - CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)  # Position modifiée de 2 à 3
        ]

# Fonction principale
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
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
