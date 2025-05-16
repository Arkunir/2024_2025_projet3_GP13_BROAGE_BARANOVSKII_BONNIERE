import pygame
import sys
import random
import os
# Suppression de l'import problématique - à remplacer par votre propre code d'IA
# from testiacode import ai_test_play
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
from ia_reinforcement import ai_reinforcement_play, best_ai_play

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

# Suppression des fonctions temporaires pour utiliser celles importées du module ia_reinforcement

def main():
    player = Player()
    game_objects = []
    score = 0
    last_object = pygame.time.get_ticks()
    
    # Distances minimales entre obstacles en fonction de la vitesse
    min_obstacle_distances = {
        6: 150,  # Espacement minimal à vitesse 6
        7: 175,  # Espacement minimal à vitesse 7
        8: 500,  # Espacement minimal à vitesse 8
        9: 250,  # Espacement minimal à vitesse 9
        10: 275, # Espacement minimal à vitesse 10
        11: 300  # Espacement minimal à vitesse 11
    }
    
    obstacle_intervals = {
        6: [800, 1400],
        7: [900, 1600],
        8: [1200, 1800],
        9: [1300, 2000],
        10: [1400, 2100],
        11: [1500, 2200]
    }
    
    object_interval = random.randint(*obstacle_intervals[INITIAL_SCROLL_SPEED])
    
    current_speed = INITIAL_SCROLL_SPEED
    previous_speed = current_speed  # Pour détecter les changements de vitesse
    
    space_pressed = False
    
    speed_threshold_7 = random.randint(10, 20)
    
    min_threshold_8 = max(25, 2 * speed_threshold_7 - 5)
    max_threshold_8 = 2 * speed_threshold_7 + 10
    speed_threshold_8 = random.randint(min_threshold_8, max_threshold_8)
    
    min_threshold_9 = max(40, 2 * speed_threshold_8 - 15)
    max_threshold_9 = 2 * speed_threshold_8 + 5             
    speed_threshold_9 = random.randint(min_threshold_9, max_threshold_9)
    
    speed_threshold_random = 100
    
    next_random_change = speed_threshold_random + random.randint(25, 50)
    
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
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

        # Vérifier s'il faut créer un nouvel obstacle
        can_spawn_obstacle = True
        min_distance = min_obstacle_distances[current_speed]
        
        # Si des obstacles existent déjà, vérifier l'espacement
        if game_objects:
            last_obstacle = game_objects[-1]
            
            # Calculer la position de fin du dernier obstacle
            if isinstance(last_obstacle, Obstacle):
                last_obstacle_right = last_obstacle.x + last_obstacle.width
            elif isinstance(last_obstacle, Block):
                last_obstacle_right = last_obstacle.rect.right
            elif isinstance(last_obstacle, (DoublePikes, TriplePikes, QuadruplePikes, FivePikesWithOrb, PurpleOrb)):
                last_obstacle_right = last_obstacle.x + last_obstacle.width
            elif isinstance(last_obstacle, BouncingObstacle):
                last_obstacle_right = last_obstacle.x + last_obstacle.width
            elif isinstance(last_obstacle, DoubleBlockPillar):
                last_obstacle_right = last_obstacle.x + last_obstacle.width
            elif isinstance(last_obstacle, BlockGapBlockWithSpike):
                last_obstacle_right = last_obstacle.x + last_obstacle.width
            elif isinstance(last_obstacle, JumpPad):
                last_obstacle_right = last_obstacle.x + last_obstacle.width
            elif isinstance(last_obstacle, QuintuplePikesWithJumpPad):
                last_obstacle_right = last_obstacle.x + last_obstacle.width
            elif isinstance(last_obstacle, JumppadOrbsObstacle):
                last_obstacle_right = last_obstacle.x + last_obstacle.width
            
            # Vérifier si l'espace est suffisant pour un nouvel obstacle
            if WIDTH - last_obstacle_right < min_distance:
                can_spawn_obstacle = False
        
        # Créer un nouvel obstacle si les conditions sont remplies
        if can_spawn_obstacle and current_time - last_object > object_interval:
            obj = None  # Initialisation pour éviter des problèmes
            
            if current_speed == 6:
                choice = random.random()
                if choice < 0.35:  # Réduire la probabilité pour faire place à notre nouvel obstacle
                    obj = Obstacle(WIDTH)
                elif choice < 0.5:
                    obj = JumpPad(WIDTH)
                elif choice < 0.65:
                    obj = Block(WIDTH)
                else:
                    obj = PurpleOrb(WIDTH)
            elif current_speed == 7:
                choice = random.random()
                if choice < 0.15:
                    obj = Obstacle(WIDTH)
                elif choice < 0.3:
                    obj = Block(WIDTH)
                elif choice < 0.5:
                    obj = DoublePikes(WIDTH)
                elif choice < 0.7:
                    obj = QuintuplePikesWithJumpPad(WIDTH)
                elif choice < 0.85:
                    obj = FivePikesWithOrb(WIDTH)
                else:
                    obj = PurpleOrb(WIDTH)
            elif current_speed == 8:
                choice = random.random()
                if choice < 0.1:
                    obj = Obstacle(WIDTH)
                elif choice < 0.2:
                    obj = DoublePikes(WIDTH)
                elif choice < 0.35:
                    obj = DoubleBlockPillar(WIDTH)
                elif choice < 0.5:
                    obj = BlockGapBlockWithSpike(WIDTH)
                elif choice < 0.6:
                    obj = BouncingObstacle(WIDTH)
                elif choice < 0.7:
                    obj = FivePikesWithOrb(WIDTH)
                elif choice < 0.85:
                    obj = PurpleOrb(WIDTH)
                else:
                    obj = JumppadOrbsObstacle(WIDTH)  # Ajout du nouvel obstacle avec 15% de chance
            elif current_speed == 9:
                choice = random.random()
                if choice < 0.05:
                    obj = Obstacle(WIDTH)
                elif choice < 0.1:
                    obj = Block(WIDTH)
                elif choice < 0.15:
                    obj = DoublePikes(WIDTH)
                elif choice < 0.3:
                    obj = BlockGapBlockWithSpike(WIDTH)
                elif choice < 0.4:
                    obj = TriplePikes(WIDTH)
                elif choice < 0.5:
                    obj = DoubleBlockPillar(WIDTH)
                elif choice < 0.6:
                    obj = QuadruplePikes(WIDTH)
                elif choice < 0.75:
                    obj = PurpleOrb(WIDTH)
                else:
                    obj = JumppadOrbsObstacle(WIDTH)  # Ajout du nouvel obstacle avec 25% de chance
            elif current_speed >= 10:
                choice = random.random()
                if choice < 0.3:
                    obj = DoublePikes(WIDTH)
                elif choice < 0.45:
                    obj = BlockGapBlockWithSpike(WIDTH)
                elif choice < 0.55:
                    obj = TriplePikes(WIDTH)
                elif choice < 0.65:
                    obj = QuadruplePikes(WIDTH)
                elif choice < 0.75:
                    obj = FivePikesWithOrb(WIDTH)
                elif choice < 0.85:
                    obj = PurpleOrb(WIDTH)
                else:
                    obj = JumppadOrbsObstacle(WIDTH)  # Ajout du nouvel obstacle avec 15% de chance
                
            # Vérifier que obj a bien été défini avant de l'utiliser
            if obj:
                obj.set_speed(current_speed)
                game_objects.append(obj)
                
                last_object = current_time
                object_interval = random.randint(*obstacle_intervals[current_speed])
        
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
                # Vérifier les collisions avec cet obstacle
                if obj.check_collision(player, keys):
                    player.is_alive = False
                    print("Game Over! Collision avec un JumppadOrbsObstacle")
                    running = False
                    break
                
                # Si l'objet est hors de l'écran, le marquer pour suppression
                if obj.x + obj.width < 0:
                    objects_to_remove.append(obj)
            
            # Gestion spécifique pour le FivePikesWithOrb
            elif isinstance(obj, FivePikesWithOrb):
                # Vérifier l'activation de l'orbe
                obj.check_activation(player, keys)
                
                # Vérifier les collisions avec les pics
                for i, rect in enumerate(obj.get_rects()):
                    if i < 5:  # Les 5 premiers rectangles sont les pics
                        if player.rect.colliderect(rect):
                            player.is_alive = False
                            print("Game Over! Collision avec un pic du FivePikesWithOrb")
                            running = False
                            break
                
                # Si l'objet est hors de l'écran, le marquer pour suppression
                if obj.x + obj.width < 0:
                    objects_to_remove.append(obj)
            
            # Gestion spécifique pour l'orbe violette
            elif isinstance(obj, PurpleOrb):
                # Vérifier l'activation de l'orbe
                obj.check_activation(player, keys)
                
                # Si l'objet est hors de l'écran, le marquer pour suppression
                if obj.x + obj.width < 0:
                    objects_to_remove.append(obj)

            elif isinstance(obj, Obstacle) and player.rect.colliderect(obj.get_rect()):
                player.is_alive = False
                print("Game Over! Collision avec un obstacle")
                running = False
                
                if not running:
                    break
                
                # Correction: vérifier si la méthode get_rects existe et traiter en conséquence
                if hasattr(obj, 'get_rects') and callable(getattr(obj, 'get_rects')):
                    rects = obj.get_rects()
                    if len(rects) > 4:  # Vérifier qu'il y a suffisamment de rectangles
                        for pad_rect in rects[4:]:  # Pad rects
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
                # Vérifier si la méthode get_rect existe
                if hasattr(obj, 'get_rect') and callable(getattr(obj, 'get_rect')):
                    pad_rect = obj.get_rect()
                    if (player.rect.bottom >= pad_rect.top and 
                        player.rect.right > pad_rect.left and 
                        player.rect.left < pad_rect.right):
                        # Vérifier si la méthode activate existe
                        if hasattr(obj, 'activate') and callable(getattr(obj, 'activate')):
                            obj.activate(player)  # Activer le pad et appliquer le boost
            
            elif isinstance(obj, QuintuplePikesWithJumpPad):
                if hasattr(obj, 'get_rects') and callable(getattr(obj, 'get_rects')):
                    rects = obj.get_rects()
                    if len(rects) > 0:  # S'assurer qu'il y a des rectangles
                        jumppad_rect = rects[-1]  # Le dernier rectangle est le jumppad
                        
                        # Vérifier la collision avec le jumppad
                        if player.rect.colliderect(jumppad_rect):
                            if hasattr(obj, 'activate_jump_pad') and callable(getattr(obj, 'activate_jump_pad')):
                                obj.activate_jump_pad(player)
                        
                        # Vérifier les collisions avec les pics (indices 5 à 9 sont les pics après les 5 blocs)
                        for i in range(5, min(10, len(rects))):
                            if player.rect.colliderect(rects[i]):
                                player.is_alive = False
                                print("Game Over! Collision avec un pic quintuple")
                                running = False
                                break
                    
            # Vérifier si l'objet est hors de l'écran pour le marquer pour suppression
            if ((isinstance(obj, Obstacle) and obj.x + obj.width < 0) or
                (isinstance(obj, Block) and obj.rect.right < 0) or
                (isinstance(obj, (DoublePikes, TriplePikes, QuadruplePikes)) and obj.x + obj.width < 0) or
                (isinstance(obj, BouncingObstacle) and obj.x + obj.width < 0) or
                (isinstance(obj, DoubleBlockPillar) and obj.x + obj.width < 0) or
                (isinstance(obj, JumpPad) and obj.x + obj.width < 0) or
                (isinstance(obj, BlockGapBlockWithSpike) and obj.x + obj.width < 0) or
                (isinstance(obj, QuintuplePikesWithJumpPad) and obj.x + obj.width < 0)):
                objects_to_remove.append(obj)
        
        # Maintenant, supprimons tous les objets marqués et incrémentons le score
        for obj in objects_to_remove:
            if obj in game_objects:  # Vérifier que l'objet est toujours dans la liste
                game_objects.remove(obj)
                score += 1
                
                # Gestion des changements de vitesse en fonction du score
                old_speed = current_speed  # Sauvegarder l'ancienne vitesse
                
                if score < speed_threshold_random:
                    if score == speed_threshold_7 and current_speed < 7:
                        current_speed = 7
                        print(f"Passage à la vitesse 7 à {score} points!")
                        for game_obj in game_objects:
                            game_obj.set_speed(current_speed)
                    elif score == speed_threshold_8 and current_speed < 8:
                        current_speed = 8
                        print(f"Passage à la vitesse 8 à {score} points!")
                        for game_obj in game_objects:
                            game_obj.set_speed(current_speed)
                    elif score == speed_threshold_9 and current_speed < 9:
                        current_speed = 9
                        print(f"Passage à la vitesse 9 à {score} points!")
                        for game_obj in game_objects:
                            game_obj.set_speed(current_speed)
                elif score == speed_threshold_random:
                    # Modification: s'assurer que la nouvelle vitesse est différente de la vitesse actuelle
                    new_speed = random.randint(9, 11)
                    if score >= 100:  # À partir du score 100, on s'assure que la vitesse change
                        while new_speed == current_speed:
                            new_speed = random.randint(9, 11)
                    
                    current_speed = new_speed
                    print(f"Passage à la vitesse aléatoire {new_speed} à {score} points!")
                    for game_obj in game_objects:
                        game_obj.set_speed(current_speed)
                    next_random_change = score + random.randint(25, 50)
                    print(f"Prochain changement à {next_random_change} points")
                elif score >= speed_threshold_random and score == next_random_change:
                    # Modification: s'assurer que la nouvelle vitesse est différente de la vitesse actuelle
                    new_speed = random.randint(9, 11)
                    if score >= 100:  # À partir du score 100, on s'assure que la vitesse change
                        while new_speed == current_speed:
                            new_speed = random.randint(9, 11)
                    
                    current_speed = new_speed
                    print(f"Nouveau changement à la vitesse aléatoire {new_speed} à {score} points!")
                    for game_obj in game_objects:
                        game_obj.set_speed(current_speed)
                    next_random_change = score + random.randint(25, 50)
                    print(f"Prochain changement à {next_random_change} points")
                
                # Vérifier si la vitesse a changé
                if current_speed != previous_speed:
                    print(f"Changement de vitesse détecté: {previous_speed} -> {current_speed}")
                    if hasattr(player, 'change_skin_randomly') and callable(getattr(player, 'change_skin_randomly')):
                        player.change_skin_randomly()
                    previous_speed = current_speed
        
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

        if score >= speed_threshold_random:
            next_change_text = font.render(f"Prochain changement à: {next_random_change}", True, BLACK)
            screen.blit(next_change_text, (20, 100))
        
        pygame.display.flip()
         
        clock.tick(FPS)
    
    show_menu()
    
def show_menu():
    button_color = (200, 200, 200)
    hover_color = (150, 150, 150)
    
    button_width, button_height = 200, 50
    start_x = WIDTH // 2 - button_width // 2
    
    # On ne garde que les trois boutons demandés
    player_button = Button("Joueur", start_x, 200, button_width, button_height, button_color, hover_color)
    reinforcement_ai_button = Button("IA par Renforcement", start_x, 280, button_width, button_height, button_color, hover_color)
    best_ai_button = Button("Meilleure IA", start_x, 360, button_width, button_height, button_color, hover_color)
    
    menu_running = True
    while menu_running:
        screen.fill(WHITE)
        
        font_title = pygame.font.SysFont(None, 60)
        title_text = font_title.render("Geometry Dash Clone", True, BLACK)
        title_rect = title_text.get_rect(center=(WIDTH // 2, 100))
        screen.blit(title_text, title_rect)
        
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
            # Utilisation de la fonction importée du module ia_reinforcement
            ai_reinforcement_play()
            show_menu()
        elif best_ai_button.check_clicked(mouse_pos, mouse_clicked):
            menu_running = False
            # Utilisation de la fonction importée du module ia_reinforcement
            best_ai_play()
            show_menu()
        
        pygame.display.flip()
        clock.tick(30)

# Démarrer le menu principal si exécuté directement
if __name__ == "__main__":
    show_menu()