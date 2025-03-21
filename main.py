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
PLAYER_SIZE = 40
OBSTACLE_WIDTH = 50
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
        self.rect = pygame.Rect(100, GROUND_HEIGHT - PLAYER_SIZE, PLAYER_SIZE, PLAYER_SIZE)
        self.velocity_y = 0
        self.is_jumping = False
        self.is_alive = True

    def update(self):
        # Appliquer la gravité
        self.velocity_y += GRAVITY
        
        # Mettre à jour la position verticale
        self.rect.y += self.velocity_y
        
        # Vérifier si le joueur touche le sol
        if self.rect.y >= GROUND_HEIGHT - PLAYER_SIZE:
            self.rect.y = GROUND_HEIGHT - PLAYER_SIZE
            self.velocity_y = 0
            self.is_jumping = False
            
    def jump(self):
        if not self.is_jumping:
            self.velocity_y = JUMP_STRENGTH
            self.is_jumping = True
            
    def draw(self):
        pygame.draw.rect(screen, BLUE, self.rect)

# Classe Obstacle
class Obstacle:
    def __init__(self, x):
        self.rect = pygame.Rect(x, GROUND_HEIGHT - OBSTACLE_WIDTH, OBSTACLE_WIDTH, OBSTACLE_WIDTH)
        
    def update(self):
        self.rect.x -= SCROLL_SPEED
        
    def draw(self):
        pygame.draw.rect(screen, RED, self.rect)

# Fonction principale
def main():
    player = Player()
    obstacles = []
    score = 0
    last_obstacle = pygame.time.get_ticks()
    obstacle_interval = 2000  # millisecondes
    
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
        
        # Ajouter de nouveaux obstacles
        current_time = pygame.time.get_ticks()
        if current_time - last_obstacle > obstacle_interval:
            obstacles.append(Obstacle(WIDTH))
            last_obstacle = current_time
            obstacle_interval = random.randint(1500, 3000)
        
        # Mettre à jour le joueur
        player.update()
        
        # Mettre à jour les obstacles
        for obstacle in obstacles[:]:
            obstacle.update()
            
            # Vérifier les collisions
            if player.rect.colliderect(obstacle.rect):
                player.is_alive = False
                print("Game Over! Score:", score)
                running = False
                
            # Supprimer les obstacles hors écran
            if obstacle.rect.right < 0:
                obstacles.remove(obstacle)
                score += 1
        
        # Dessiner l'arrière-plan
        screen.fill(WHITE)
        
        # Dessiner le sol
        pygame.draw.rect(screen, BLACK, (0, GROUND_HEIGHT, WIDTH, HEIGHT - GROUND_HEIGHT))
        
        # Dessiner le joueur
        player.draw()
        
        # Dessiner les obstacles
        for obstacle in obstacles:
            obstacle.draw()
            
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