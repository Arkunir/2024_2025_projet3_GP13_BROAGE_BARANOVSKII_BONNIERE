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
SCROLL_SPEED = 5

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

    def update(self, game_objects):
        # Appliquer la gravité
        self.velocity_y += GRAVITY
        
        # Sauvegarder la position précédente pour la détection de collision
        old_y = self.rect.y
        
        # Mettre à jour la position verticale
        self.rect.y += self.velocity_y
        
        # Réinitialiser l'objet sur lequel le joueur se tient
        self.standing_on = None
        
        # Vérifier les collisions avec les blocs
        for obj in game_objects:
            if isinstance(obj, Block):
                # Collision avec le dessus du bloc
                if (old_y + CUBE_SIZE <= obj.rect.top and 
                    self.rect.y + CUBE_SIZE >= obj.rect.top and
                    self.rect.right > obj.rect.left and
                    self.rect.left < obj.rect.right):
                    
                    self.rect.bottom = obj.rect.top
                    self.velocity_y = 0
                    self.is_jumping = False
                    self.standing_on = obj
                    
                # Collision latérale avec le bloc
                elif (self.rect.colliderect(obj.rect) and 
                      not (old_y + CUBE_SIZE <= obj.rect.top)):
                    self.is_alive = False
                    print("Game Over! Collision latérale avec un bloc")
        
        # Vérifier si le joueur touche le sol
        if self.rect.y >= GROUND_HEIGHT - CUBE_SIZE and not self.standing_on:
            self.rect.y = GROUND_HEIGHT - CUBE_SIZE
            self.velocity_y = 0
            self.is_jumping = False
            
    def jump(self):
        if not self.is_jumping:
            self.velocity_y = JUMP_STRENGTH
            self.is_jumping = True
            
    def draw(self):
        pygame.draw.rect(screen, BLUE, self.rect)

# Classe Obstacle (Triangles rouges - dangereux)
class Obstacle:
    def __init__(self, x):
        self.x = x
        self.y = GROUND_HEIGHT - CUBE_SIZE
        self.width = CUBE_SIZE
        self.height = CUBE_SIZE
        
    def update(self):
        self.x -= SCROLL_SPEED
        
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

# Classe Block (Blocs noirs - inoffensifs si touchés par le dessus)
class Block:
    def __init__(self, x):
        self.rect = pygame.Rect(x, GROUND_HEIGHT - CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        
    def update(self):
        self.rect.x -= SCROLL_SPEED
        
    def draw(self):
        pygame.draw.rect(screen, BLACK, self.rect)

# Fonction principale
def main():
    player = Player()
    game_objects = []  # Liste combinée pour les obstacles et les blocs
    score = 0
    last_object = pygame.time.get_ticks()
    object_interval = 2000  # millisecondes
    
    # Boucle de jeu
    running = True
    while running:
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    player.jump()
        
        # Ajouter de nouveaux obstacles ou blocs
        current_time = pygame.time.get_ticks()
        if current_time - last_object > object_interval:
            # 70% de chance d'avoir un obstacle, 30% d'avoir un bloc
            if random.random() < 0.7:
                game_objects.append(Obstacle(WIDTH))
            else:
                game_objects.append(Block(WIDTH))
                
            last_object = current_time
            object_interval = random.randint(1500, 3000)
        
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
                
            # Supprimer les objets hors écran
            if (isinstance(obj, Obstacle) and obj.x + obj.width < 0) or \
               (isinstance(obj, Block) and obj.rect.right < 0):
                game_objects.remove(obj)
                score += 1
        
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
        
        # Mettre à jour l'écran
        pygame.display.flip()
        
        # Contrôler la vitesse du jeu
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 