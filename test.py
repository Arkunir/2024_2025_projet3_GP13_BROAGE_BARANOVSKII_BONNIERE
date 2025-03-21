import pygame
import sys

# Initialiser Pygame
pygame.init()

# Dimensions de la fenêtre
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# Titre de la fenêtre
pygame.display.set_caption('Rotation d\'image avec Pygame')

# Charger l'image
image = pygame.image.load('votre_image.png')  # Remplacez par le chemin de votre image
rect = image.get_rect(center=(screen_width // 2, screen_height // 2))

# Variable pour l'angle de rotation
angle = 0

# Boucle principale du jeu
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Remplir l'écran avec une couleur de fond
    screen.fill((255, 255, 255))

    # Appliquer la rotation (l'image tourne de 1 degré à chaque frame)
    rotated_image = pygame.transform.rotate(image, angle)
    
    # Récupérer le nouveau rectangle pour l'image pivotée (important pour garder l'image centrée)
    rotated_rect = rotated_image.get_rect(center=rect.center)

    # Afficher l'image pivotée
    screen.blit(rotated_image, rotated_rect)

    # Augmenter l'angle de rotation
    angle += 1  # Vous pouvez ajuster la vitesse de rotation ici
    
    # Mettre à jour l'écran
    pygame.display.flip()

    # Limiter le taux de rafraîchissement (environ 60 FPS)
    pygame.time.Clock().tick(60)

# Quitter Pygame
pygame.quit()
sys.exit()
