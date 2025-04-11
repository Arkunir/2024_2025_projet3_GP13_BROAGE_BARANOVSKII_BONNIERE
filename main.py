import pygame
import sys
import random
from testiacode import ai_test_play
from klass import Player
from klass import MovingObject
from klass import Obstacle
from klass import DoublePikes
from klass import TriplePikes
from klass import QuadruplePikes
from klass import Block
from klass import BlockGapBlockWithSpike
from klass import Button
from klass import BouncingObstacle
from klass import DoubleBlockPillar

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

def main():
    player = Player()
    game_objects = []
    score = 0
    last_object = pygame.time.get_ticks()
    
    # Distances minimales entre obstacles en fonction de la vitesse
    min_obstacle_distances = {
        6: 150,  # Espacement minimal à vitesse 6
        7: 175,  # Espacement minimal à vitesse 7
        8: 225,  # Espacement minimal à vitesse 8
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
            elif isinstance(last_obstacle, DoublePikes) or isinstance(last_obstacle, TriplePikes) or isinstance(last_obstacle, QuadruplePikes):
                last_obstacle_right = last_obstacle.x + last_obstacle.width
            elif isinstance(last_obstacle, BouncingObstacle):
                last_obstacle_right = last_obstacle.x + last_obstacle.width
            elif isinstance(last_obstacle, DoubleBlockPillar):
                last_obstacle_right = last_obstacle.x + last_obstacle.width
            elif isinstance(last_obstacle, BlockGapBlockWithSpike):
                last_obstacle_right = last_obstacle.x + last_obstacle.width
            
            # Vérifier si l'espace est suffisant pour un nouvel obstacle
            if WIDTH - last_obstacle_right < min_distance:
                can_spawn_obstacle = False
        
        # Créer un nouvel obstacle si les conditions sont remplies
        if can_spawn_obstacle and current_time - last_object > object_interval:
            if current_speed == 6:
                choice = random.random()
                if choice < 0.6:
                    obj = Obstacle(WIDTH)
                else:
                    obj = Block(WIDTH)
            elif current_speed == 7:
                choice = random.random()
                if choice < 0.3:
                    obj = Obstacle(WIDTH)
                elif choice < 0.6:
                    obj = Block(WIDTH)
                else:
                    obj = DoublePikes(WIDTH)
            elif current_speed == 8:
                choice = random.random()
                if choice < 0.1:
                    obj = Obstacle(WIDTH)
                elif choice < 0.2:
                    obj = DoublePikes(WIDTH)
                elif choice < 0.5:
                    obj = DoubleBlockPillar(WIDTH)
                elif choice < 0.7:
                    obj = BlockGapBlockWithSpike(WIDTH)
                else:
                    obj = BouncingObstacle(WIDTH) 
            elif current_speed == 9:
                choice = random.random()
                if choice < 0.08:
                    obj = Obstacle(WIDTH)
                elif choice < 0.16:
                    obj = Block(WIDTH)
                elif choice < 0.36:
                    obj = DoublePikes(WIDTH)
                elif choice < 0.66:
                    obj = BlockGapBlockWithSpike(WIDTH)
                elif choice < 0.96:
                    obj = TriplePikes(WIDTH)
                else:
                    obj = QuadruplePikes(WIDTH)
            elif current_speed >= 10:
                choice = random.random()
                if choice < 0.5:
                    obj = DoublePikes(WIDTH)
                elif choice < 0.8:
                    obj = BlockGapBlockWithSpike(WIDTH)
                elif choice < 0.95:
                    obj = TriplePikes(WIDTH)
                else:
                    obj = QuadruplePikes(WIDTH)
                
            obj.set_speed(current_speed)
            game_objects.append(obj)
                
            last_object = current_time
            object_interval = random.randint(*obstacle_intervals[current_speed])
        
        player.update(game_objects)
        
        if not player.is_alive:
            print("Game Over! Score:", score)
            running = False
        
        for obj in game_objects[:]:
            obj.update()
            
            if isinstance(obj, Obstacle) and player.rect.colliderect(obj.get_rect()):
                player.is_alive = False
                print("Game Over! Collision avec un obstacle")
                running = False
                
                if not running:
                    break
                
                for pad_rect in obj.get_rects()[4:]:  # Pad rects
                    if player.rect.colliderect(pad_rect):
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
            
            if ((isinstance(obj, Obstacle) and obj.x + obj.width < 0) or
                (isinstance(obj, Block) and obj.rect.right < 0) or
                (isinstance(obj, DoublePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, TriplePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, QuadruplePikes) and obj.x + obj.width < 0) or
                (isinstance(obj, BouncingObstacle) and obj.x + obj.width < 0) or
                (isinstance(obj, DoubleBlockPillar) and obj.x + obj.width < 0) or
                (isinstance(obj, BlockGapBlockWithSpike) and obj.x + obj.width < 0)):
                game_objects.remove(obj)
                score += 1
                
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
        
        screen.fill(WHITE)
        
        pygame.draw.rect(screen, BLACK, (0, GROUND_HEIGHT, WIDTH, HEIGHT - GROUND_HEIGHT))
        
        player.draw(screen)  # Correction ici: ajout du paramètre screen
        
        for obj in game_objects:
            obj.draw(screen)  # Correction ici: s'assurer que tous les objets reçoivent aussi screen
            
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
    
    player_button = Button("Joueur", start_x, 200, button_width, button_height, button_color, hover_color)
    ai_reinforcement_button = Button("IA par Renforcement", start_x, 280, button_width, button_height, button_color, hover_color)
    ai_test_button = Button("IA Test", start_x, 360, button_width, button_height, button_color, hover_color)
    
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
        ai_reinforcement_button.update(mouse_pos)
        ai_test_button.update(mouse_pos)
        
        player_button.draw(screen)
        ai_reinforcement_button.draw(screen)
        ai_test_button.draw(screen)
        
        if player_button.check_clicked(mouse_pos, mouse_clicked):
            menu_running = False
            main()
        elif ai_reinforcement_button.check_clicked(mouse_pos, mouse_clicked):
            font = pygame.font.SysFont(None, 24)
            info_text = font.render("Fonctionnalité non implémentée", True, RED)
            screen.blit(info_text, (WIDTH // 2 - 120, 450))
            pygame.display.flip()
            pygame.time.wait(1000)
        elif ai_test_button.check_clicked(mouse_pos, mouse_clicked):
            menu_running = False
            ai_test_play()
            show_menu()
        
        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    show_menu()    
    pygame.quit()
