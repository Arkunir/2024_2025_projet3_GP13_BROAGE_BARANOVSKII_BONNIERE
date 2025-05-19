import pygame
import sys
import os
from klass import Button
from klass import Player
from klass import MovingObject
from klass import Obstacle
from klass import DoublePikes
from klass import TriplePikes   
from klass import QuadruplePikes
from klass import Block
from klass import BlockGapBlockWithSpike
from klass import BouncingObstacle
from klass import DoubleBlockPillar
from klass import FivePikesWithOrb
from klass import JumpPad
from klass import QuintuplePikesWithJumpPad
from klass import PurpleOrb
from klass import JumppadOrbsObstacle  
# Nous allons créer nos propres versions des fonctions IA
# from ia_reinforcement import ai_reinforcement_play, best_ai_play

def ai_reinforcement_play_v2():
    """Version 2.0 de l'IA par renforcement utilisant le main 2.0"""
    # Import local pour éviter les conflits
    import ia_reinforcement
    
    # Remplacer temporairement la fonction main dans le module ia_reinforcement
    original_main = getattr(ia_reinforcement, 'main', None)
    ia_reinforcement.main = main
    
    try:
        # Exécuter l'IA par renforcement avec notre main 2.0
        ia_reinforcement.ai_reinforcement_play()
    finally:
        # Restaurer la fonction main originale
        if original_main:
            ia_reinforcement.main = original_main

def best_ai_play_v2():
    """Version 2.0 de la meilleure IA utilisant le main 2.0"""
    # Import local pour éviter les conflits
    import ia_reinforcement
    
    # Remplacer temporairement la fonction main dans le module ia_reinforcement
    original_main = getattr(ia_reinforcement, 'main', None)
    ia_reinforcement.main = main
    
    try:
        # Exécuter la meilleure IA avec notre main 2.0
        ia_reinforcement.best_ai_play()
    finally:
        # Restaurer la fonction main originale
        if original_main:
            ia_reinforcement.main = original_main

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
pygame.display.set_caption("Geometry Dash Clone 2.0")
clock = pygame.time.Clock()

# Patterns d'obstacles prédéfinis par vitesse
OBSTACLE_PATTERNS = {
    6: [
        ('Obstacle', 200),
        ('JumpPad', 180),
        ('Block', 220),
        ('PurpleOrb', 190),
        ('Obstacle', 210),
        ('JumpPad', 170),
        ('Block', 200),
        ('PurpleOrb', 180),
        ('Obstacle', 190),
        ('JumpPad', 200)
    ],
    7: [
        ('DoublePikes', 250),
        ('Block', 200),
        ('QuintuplePikesWithJumpPad', 280),
        ('FivePikesWithOrb', 230),
        ('PurpleOrb', 180),
        ('Obstacle', 220),
        ('DoublePikes', 240),
        ('QuintuplePikesWithJumpPad', 270),
        ('FivePikesWithOrb', 220),
        ('Block', 190)
    ],
    8: [
        ('DoubleBlockPillar', 300),
        ('BlockGapBlockWithSpike', 280),
        ('BouncingObstacle', 250),
        ('JumppadOrbsObstacle', 320),
        ('DoublePikes', 230),
        ('FivePikesWithOrb', 260),
        ('PurpleOrb', 200),
        ('DoubleBlockPillar', 290),
        ('BlockGapBlockWithSpike', 270),
        ('BouncingObstacle', 240)
    ],
    9: [
        ('TriplePikes', 280),
        ('BlockGapBlockWithSpike', 320),
        ('DoubleBlockPillar', 300),
        ('QuadruplePikes', 350),
        ('JumppadOrbsObstacle', 330),
        ('PurpleOrb', 220),
        ('TriplePikes', 270),
        ('BlockGapBlockWithSpike', 310),
        ('QuadruplePikes', 340),
        ('DoubleBlockPillar', 290)
    ],
    10: [
        ('QuadruplePikes', 380),
        ('BlockGapBlockWithSpike', 350),
        ('TriplePikes', 320),
        ('FivePikesWithOrb', 360),
        ('JumppadOrbsObstacle', 390),
        ('PurpleOrb', 250),
        ('QuadruplePikes', 370),
        ('BlockGapBlockWithSpike', 340),
        ('TriplePikes', 310),
        ('FivePikesWithOrb', 350)
    ],
    11: [
        ('QuadruplePikes', 400),
        ('FivePikesWithOrb', 380),
        ('JumppadOrbsObstacle', 420),
        ('BlockGapBlockWithSpike', 360),
        ('TriplePikes', 340),
        ('PurpleOrb', 280),
        ('QuadruplePikes', 390),
        ('FivePikesWithOrb', 370),
        ('JumppadOrbsObstacle', 410),
        ('TriplePikes', 330)
    ]
}

# Seuils de vitesse prédéfinis
SPEED_THRESHOLDS = {
    7: 15,   # Passage à la vitesse 7 au score 15
    8: 35,   # Passage à la vitesse 8 au score 35
    9: 60,   # Passage à la vitesse 9 au score 60
    10: 90,  # Passage à la vitesse 10 au score 90
    11: 125  # Passage à la vitesse 11 au score 125
}

def create_obstacle(obstacle_type, x_position):
    """Crée un obstacle du type spécifié à la position donnée"""
    if obstacle_type == 'Obstacle':
        return Obstacle(x_position)
    elif obstacle_type == 'DoublePikes':
        return DoublePikes(x_position)
    elif obstacle_type == 'TriplePikes':
        return TriplePikes(x_position)
    elif obstacle_type == 'QuadruplePikes':
        return QuadruplePikes(x_position)
    elif obstacle_type == 'Block':
        return Block(x_position)
    elif obstacle_type == 'BlockGapBlockWithSpike':
        return BlockGapBlockWithSpike(x_position)
    elif obstacle_type == 'BouncingObstacle':
        return BouncingObstacle(x_position)
    elif obstacle_type == 'DoubleBlockPillar':
        return DoubleBlockPillar(x_position)
    elif obstacle_type == 'FivePikesWithOrb':
        return FivePikesWithOrb(x_position)
    elif obstacle_type == 'JumpPad':
        return JumpPad(x_position)
    elif obstacle_type == 'QuintuplePikesWithJumpPad':
        return QuintuplePikesWithJumpPad(x_position)
    elif obstacle_type == 'PurpleOrb':
        return PurpleOrb(x_position)
    elif obstacle_type == 'JumppadOrbsObstacle':
        return JumppadOrbsObstacle(x_position)
    else:
        return Obstacle(x_position)  # Défaut

def main():
    player = Player()
    game_objects = []
    score = 0
    
    current_speed = INITIAL_SCROLL_SPEED
    previous_speed = current_speed
    
    # Variables pour gérer les patterns d'obstacles
    current_pattern_index = 0
    next_obstacle_position = WIDTH + 200  # Position du prochain obstacle
    pattern = OBSTACLE_PATTERNS[current_speed]
    
    space_pressed = False
    
    running = True
    while running:
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

        # Créer des obstacles selon le pattern prédéfini
        can_create_obstacle = False
        if len(game_objects) == 0:
            can_create_obstacle = True
        else:
            last_obj = game_objects[-1]
            # Gérer les différents types d'objets selon leur attribut de position
            if isinstance(last_obj, Block):
                last_obj_x = last_obj.rect.x
            elif hasattr(last_obj, 'x'):
                last_obj_x = last_obj.x
            else:
                last_obj_x = 0  # Fallback
            
            if last_obj_x + 100 < next_obstacle_position:
                can_create_obstacle = True
        
        if can_create_obstacle:
            # Récupérer le pattern pour la vitesse actuelle
            pattern = OBSTACLE_PATTERNS[current_speed]
            obstacle_type, distance = pattern[current_pattern_index % len(pattern)]
            
            # Créer l'obstacle
            obj = create_obstacle(obstacle_type, next_obstacle_position)
            if obj:
                obj.set_speed(current_speed)
                game_objects.append(obj)
                
                # Calculer la position du prochain obstacle
                next_obstacle_position += distance
                current_pattern_index += 1

        player.update(game_objects)
        
        if not player.is_alive:
            print("Game Over! Score:", score)
            running = False
        
        # Créer une copie de la liste pour éviter les problèmes de modification pendant l'itération
        objects_to_remove = []
        
        # Mettre à jour les objets du jeu et vérifier les collisions
        for obj in game_objects[:]:
            obj.update()
            
            # Gestion spécifique pour le JumppadOrbsObstacle
            if isinstance(obj, JumppadOrbsObstacle):
                if obj.check_collision(player, keys):
                    player.is_alive = False
                    print("Game Over! Collision avec un JumppadOrbsObstacle")
                    running = False
                    break
                
                if obj.x + obj.width < 0:
                    objects_to_remove.append(obj)
            
            # Gestion spécifique pour le FivePikesWithOrb
            elif isinstance(obj, FivePikesWithOrb):
                obj.check_activation(player, keys)
                
                for i, rect in enumerate(obj.get_rects()):
                    if i < 5:  # Les 5 premiers rectangles sont les pics
                        if player.rect.colliderect(rect):
                            player.is_alive = False
                            print("Game Over! Collision avec un pic du FivePikesWithOrb")
                            running = False
                            break
                
                if obj.x + obj.width < 0:
                    objects_to_remove.append(obj)
            
            # Gestion spécifique pour l'orbe violette
            elif isinstance(obj, PurpleOrb):
                obj.check_activation(player, keys)
                
                if obj.x + obj.width < 0:
                    objects_to_remove.append(obj)

            elif isinstance(obj, Obstacle) and player.rect.colliderect(obj.get_rect()):
                player.is_alive = False
                print("Game Over! Collision avec un obstacle")
                running = False
                
                if not running:
                    break
                
                if hasattr(obj, 'get_rects') and callable(getattr(obj, 'get_rects')):
                    rects = obj.get_rects()
                    if len(rects) > 4:
                        for pad_rect in rects[4:]:
                            if player.rect.colliderect(pad_rect) and hasattr(obj, 'activate_pads'):
                                obj.activate_pads(player)
            
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
                    
            elif isinstance(obj, DoubleBlockPillar):
                for rect in obj.get_rects():
                    if player.rect.colliderect(rect):
                        player.is_alive = False
                        print("Game Over! Collision avec un pilier de blocs")
                        running = False
                        break
                    
            elif isinstance(obj, JumpPad):
                if hasattr(obj, 'get_rect') and callable(getattr(obj, 'get_rect')):
                    pad_rect = obj.get_rect()
                    if (player.rect.bottom >= pad_rect.top and 
                        player.rect.right > pad_rect.left and 
                        player.rect.left < pad_rect.right):
                        if hasattr(obj, 'activate') and callable(getattr(obj, 'activate')):
                            obj.activate(player)
            
            elif isinstance(obj, QuintuplePikesWithJumpPad):
                if hasattr(obj, 'get_rects') and callable(getattr(obj, 'get_rects')):
                    rects = obj.get_rects()
                    if len(rects) > 0:
                        jumppad_rect = rects[-1]
                        
                        if player.rect.colliderect(jumppad_rect):
                            if hasattr(obj, 'activate_jump_pad') and callable(getattr(obj, 'activate_jump_pad')):
                                obj.activate_jump_pad(player)
                        
                        for i in range(5, min(10, len(rects))):
                            if player.rect.colliderect(rects[i]):
                                player.is_alive = False
                                print("Game Over! Collision avec un pic quintuple")
                                running = False
                                break
                    
            # Vérifier si l'objet est hors de l'écran pour le marquer pour suppression
            obj_right_edge = 0
            if isinstance(obj, Block):
                obj_right_edge = obj.rect.right
            elif hasattr(obj, 'x') and hasattr(obj, 'width'):
                obj_right_edge = obj.x + obj.width
            elif hasattr(obj, 'x'):
                obj_right_edge = obj.x
                
            if obj_right_edge < 0:
                objects_to_remove.append(obj)
        
        # Supprimer les objets marqués et incrémenter le score
        for obj in objects_to_remove:
            if obj in game_objects:
                game_objects.remove(obj)
                score += 1
                
                # Gestion des changements de vitesse selon les seuils prédéfinis
                old_speed = current_speed
                
                # Vérifier chaque seuil de vitesse
                for speed, threshold in SPEED_THRESHOLDS.items():
                    if score == threshold and current_speed < speed:
                        current_speed = speed
                        print(f"Passage à la vitesse {speed} au score {score}!")
                        # Mettre à jour la vitesse de tous les objets existants
                        for game_obj in game_objects:
                            game_obj.set_speed(current_speed)
                        # Réinitialiser le pattern pour la nouvelle vitesse
                        current_pattern_index = 0
                        break
                
                # Vérifier si la vitesse a changé
                if current_speed != previous_speed:
                    print(f"Changement de vitesse détecté: {previous_speed} -> {current_speed}")
                    if hasattr(player, 'change_skin_randomly') and callable(getattr(player, 'change_skin_randomly')):
                        player.change_skin_randomly()
                    previous_speed = current_speed
        
        # Affichage
        screen.fill(WHITE)
        
        pygame.draw.rect(screen, BLACK, (0, GROUND_HEIGHT, WIDTH, HEIGHT - GROUND_HEIGHT))
        
        player.draw(screen)
        
        for obj in game_objects:
            obj.draw(screen)
            
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (20, 20))
        
        speed_text = font.render(f"Vitesse: {current_speed}", True, BLACK)
        screen.blit(speed_text, (20, 60))
        
        # Afficher le prochain seuil de vitesse
        next_threshold = None
        for speed, threshold in SPEED_THRESHOLDS.items():
            if threshold > score:
                next_threshold = threshold
                break
        
        if next_threshold:
            threshold_text = font.render(f"Prochain niveau à: {next_threshold}", True, BLACK)
            screen.blit(threshold_text, (20, 100))
        else:
            max_text = font.render("Vitesse maximale atteinte!", True, BLACK)
            screen.blit(max_text, (20, 100))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    show_menu()
    
def show_menu():
    button_color = (200, 200, 200)
    hover_color = (150, 150, 150)
    
    button_width, button_height = 200, 50
    start_x = WIDTH // 2 - button_width // 2
    
    player_button = Button("Joueur", start_x, 200, button_width, button_height, button_color, hover_color)
    reinforcement_ai_button = Button("IA par Renforcement", start_x, 280, button_width, button_height, button_color, hover_color)
    best_ai_button = Button("Meilleure IA", start_x, 360, button_width, button_height, button_color, hover_color)
    
    menu_running = True
    while menu_running:
        screen.fill(WHITE)
        
        font_title = pygame.font.SysFont(None, 60)
        title_text = font_title.render("Geometry Dash Clone 2.0", True, BLACK)
        title_rect = title_text.get_rect(center=(WIDTH // 2, 100))
        screen.blit(title_text, title_rect)
        
        # Afficher les informations sur les patterns
        info_font = pygame.font.SysFont(None, 24)
        info_text = info_font.render("Version avec obstacles prédéfinis", True, BLACK)
        info_rect = info_text.get_rect(center=(WIDTH // 2, 140))
        screen.blit(info_text, info_rect)
        
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
        reinforcement_ai_button.update(mouse_pos)
        best_ai_button.update(mouse_pos)
        
        player_button.draw(screen)
        reinforcement_ai_button.draw(screen)
        best_ai_button.draw(screen)
        
        if player_button.check_clicked(mouse_pos, mouse_clicked):
            menu_running = False
            main()
        elif reinforcement_ai_button.check_clicked(mouse_pos, mouse_clicked):
            menu_running = False
            ai_reinforcement_play_v2()
            show_menu()
        elif best_ai_button.check_clicked(mouse_pos, mouse_clicked):
            menu_running = False
            best_ai_play_v2()
            show_menu()
        
        pygame.display.flip()
        clock.tick(30)

# Démarrer le menu principal si exécuté directement
if __name__ == "__main__":
    show_menu()