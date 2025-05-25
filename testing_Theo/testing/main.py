import pygame
import sys
import random
import os
import pickle
import matplotlib.pyplot as plt
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

def main():
    player = Player()
    game_objects = []
    score = 0
    last_object = pygame.time.get_ticks()
    
    min_obstacle_distances = {
        6: 150,
        7: 175,
        8: 500,
        9: 250,
        10: 275,
        11: 300
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
    previous_speed = current_speed
    
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

        can_spawn_obstacle = True
        min_distance = min_obstacle_distances[current_speed]
        
        if game_objects:
            last_obstacle = game_objects[-1]
            
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
            
            if WIDTH - last_obstacle_right < min_distance:
                can_spawn_obstacle = False
        
        if can_spawn_obstacle and current_time - last_object > object_interval:
            obj = None
            
            if current_speed == 6:
                choice = random.random()
                if choice < 0.35:
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
                    obj = JumppadOrbsObstacle(WIDTH)
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
                    obj = JumppadOrbsObstacle(WIDTH)
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
                    obj = JumppadOrbsObstacle(WIDTH)
                
            if obj:
                obj.set_speed(current_speed)
                game_objects.append(obj)
                
                last_object = current_time
                object_interval = random.randint(*obstacle_intervals[current_speed])
        
        player.update(game_objects)
        
        if not player.is_alive:
            print("Game Over! Score:", score)
            running = False
        
        objects_to_remove = []
        
        for obj in game_objects[:]:
            obj.update()
            
            if isinstance(obj, JumppadOrbsObstacle):
                if obj.check_collision(player, keys):
                    player.is_alive = False
                    print("Game Over! Collision avec un JumppadOrbsObstacle")
                    running = False
                    break
                
                if obj.x + obj.width < 0:
                    objects_to_remove.append(obj)
            
            elif isinstance(obj, FivePikesWithOrb):
                obj.check_activation(player, keys)
                
                for i, rect in enumerate(obj.get_rects()):
                    if i < 5:
                        if player.rect.colliderect(rect):
                            player.is_alive = False
                            print("Game Over! Collision avec un pic du FivePikesWithOrb")
                            running = False
                            break
                
                if obj.x + obj.width < 0:
                    objects_to_remove.append(obj)
            
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
                    
            if ((isinstance(obj, Obstacle) and obj.x + obj.width < 0) or
                (isinstance(obj, Block) and obj.rect.right < 0) or
                (isinstance(obj, (DoublePikes, TriplePikes, QuadruplePikes)) and obj.x + obj.width < 0) or
                (isinstance(obj, BouncingObstacle) and obj.x + obj.width < 0) or
                (isinstance(obj, DoubleBlockPillar) and obj.x + obj.width < 0) or
                (isinstance(obj, JumpPad) and obj.x + obj.width < 0) or
                (isinstance(obj, BlockGapBlockWithSpike) and obj.x + obj.width < 0) or
                (isinstance(obj, QuintuplePikesWithJumpPad) and obj.x + obj.width < 0)):
                objects_to_remove.append(obj)
        
        for obj in objects_to_remove:
            if obj in game_objects:
                game_objects.remove(obj)
                score += 1
                
                old_speed = current_speed
                
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
                    new_speed = random.randint(9, 11)
                    if score >= 100:
                        while new_speed == current_speed:
                            new_speed = random.randint(9, 11)
                    
                    current_speed = new_speed
                    print(f"Passage à la vitesse aléatoire {new_speed} à {score} points!")
                    for game_obj in game_objects:
                        game_obj.set_speed(current_speed)
                    next_random_change = score + random.randint(25, 50)
                    print(f"Prochain changement à {next_random_change} points")
                elif score >= speed_threshold_random and score == next_random_change:
                    new_speed = random.randint(9, 11)
                    if score >= 100:
                        while new_speed == current_speed:
                            new_speed = random.randint(9, 11)
                    
                    current_speed = new_speed
                    print(f"Nouveau changement à la vitesse aléatoire {new_speed} à {score} points!")
                    for game_obj in game_objects:
                        game_obj.set_speed(current_speed)
                    next_random_change = score + random.randint(25, 50)
                    print(f"Prochain changement à {next_random_change} points")
                
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
def show_training_graph():
    """Affiche un graphique de l'évolution du score moyen à partir du fichier de modèle."""
    model_path = 'geometry_dash_ai_modelv3.pkl'
    
    if not os.path.exists(model_path):
        print(f"Le fichier {model_path} n'existe pas.")
        return
    
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, dict) and 'training_scores' in data:
            scores = data['training_scores']
        else:
            print("Aucune donnée de scores d'entraînement trouvée dans le fichier.")
            return
            
        if not scores:
            print("Aucun score d'entraînement disponible.")
            return
            
        # Calculer les moyennes mobiles pour lisser la courbe
        window_size = min(10, len(scores))
        moving_averages = []
        
        for i in range(len(scores)):
            start_idx = max(0, i - window_size + 1)
            window_scores = scores[start_idx:i+1]
            moving_averages.append(sum(window_scores) / len(window_scores))
        
        plt.figure(figsize=(12, 8))
        
        # Graphique des scores bruts
        plt.subplot(2, 1, 1)
        plt.plot(scores, alpha=0.3, color='lightblue', label='Scores bruts')
        plt.plot(moving_averages, color='blue', linewidth=2, label=f'Moyenne mobile ({window_size} épisodes)')
        plt.title('Évolution des scores d\'entraînement')
        plt.xlabel('Épisode')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graphique des statistiques
        plt.subplot(2, 1, 2)
        episodes_100 = [i for i in range(99, len(scores), 100)]
        avg_per_100 = [sum(scores[max(0, i-99):i+1]) / min(100, i+1) for i in episodes_100]
        
        plt.plot(episodes_100, avg_per_100, 'ro-', linewidth=2, markersize=4)
        plt.title('Score moyen par tranche de 100 épisodes')
        plt.xlabel('Épisode')
        plt.ylabel('Score moyen')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Afficher quelques statistiques
        print(f"\n=== Statistiques d'entraînement ===")
        print(f"Nombre total d'épisodes: {len(scores)}")
        print(f"Score maximum: {max(scores)}")
        print(f"Score moyen global: {sum(scores)/len(scores):.2f}")
        print(f"Score moyen des 100 derniers épisodes: {sum(scores[-100:])/min(100, len(scores)):.2f}")
        
    except Exception as e:
        print(f"Erreur lors du chargement du graphique: {e}")

def show_menu():
    button_color = (200, 200, 200)
    hover_color = (150, 150, 150)
    
    button_width, button_height = 200, 50
    start_x = WIDTH // 2 - button_width // 2
    
    player_button = Button("Joueur", start_x, 180, button_width, button_height, button_color, hover_color)
    reinforcement_ai_button = Button("IA par Renforcement", start_x, 240, button_width, button_height, button_color, hover_color)
    best_ai_button = Button("Meilleure IA", start_x, 300, button_width, button_height, button_color, hover_color)
    graph_button = Button("Voir Graphique", start_x, 360, button_width, button_height, button_color, hover_color)
    
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
        graph_button.update(mouse_pos)
        
        player_button.draw(screen)
        reinforcement_ai_button.draw(screen)
        best_ai_button.draw(screen)
        graph_button.draw(screen)
        
        if player_button.check_clicked(mouse_pos, mouse_clicked):
            menu_running = False
            main()
        elif reinforcement_ai_button.check_clicked(mouse_pos, mouse_clicked):
            menu_running = False
            ai_reinforcement_play()
            show_menu()
        elif best_ai_button.check_clicked(mouse_pos, mouse_clicked):
            menu_running = False
            best_ai_play()
            show_menu()
        elif graph_button.check_clicked(mouse_pos, mouse_clicked):
            show_training_graph()
        
        pygame.display.flip()
        clock.tick(30)
if __name__ == "__main__":
    show_menu()