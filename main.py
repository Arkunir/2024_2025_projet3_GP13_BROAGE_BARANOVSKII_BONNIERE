import pygame
import sys
import random
import os
import pickle
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
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (100, 100, 100)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Geometry Dash Clone")
clock = pygame.time.Clock()

def load_training_data():
    """Charge les données d'entraînement depuis le fichier pkl."""
    model_path = 'geometry_dash_ai_modelv5.pkl'
    
    if not os.path.exists(model_path):
        print(f"Le fichier {model_path} n'existe pas.")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            
        # Gestion de différents formats de données possibles
        if isinstance(data, dict):
            # Si c'est un dictionnaire, chercher les scores
            if 'training_scores' in data:
                return data['training_scores']
            elif 'scores' in data:
                return data['scores']
            elif 'episode_scores' in data:
                return data['episode_scores']
            else:
                # Si aucune clé connue, essayer de prendre la première liste trouvée
                for key, value in data.items():
                    if isinstance(value, list) and value:
                        return value
        elif isinstance(data, list):
            # Si c'est directement une liste
            return data
        elif hasattr(data, 'training_scores'):
            # Si c'est un objet avec un attribut training_scores
            return data.training_scores
            
        print("Format de données non reconnu dans le fichier pkl.")
        print(f"Type de données: {type(data)}")
        if isinstance(data, dict):
            print(f"Clés disponibles: {list(data.keys())}")
        return None
        
    except Exception as e:
        print(f"Erreur lors du chargement du fichier: {e}")
        return None

def calculate_moving_average(scores, window_size=10):
    """Calcule la moyenne mobile des scores."""
    if not scores:
        return []
    
    window_size = min(window_size, len(scores))
    moving_averages = []
    
    for i in range(len(scores)):
        start_idx = max(0, i - window_size + 1)
        window_scores = scores[start_idx:i+1]
        moving_averages.append(sum(window_scores) / len(window_scores))
    
    return moving_averages

def draw_graph(surface, scores, x, y, width, height):
    """Dessine un graphique des scores sur la surface Pygame."""
    if not scores:
        # Afficher un message si pas de données
        font = pygame.font.SysFont(None, 24)
        text = font.render("Aucune donnée disponible", True, BLACK)
        text_rect = text.get_rect(center=(x + width//2, y + height//2))
        surface.blit(text, text_rect)
        return
    
    # Dessiner le cadre du graphique
    pygame.draw.rect(surface, WHITE, (x, y, width, height))
    pygame.draw.rect(surface, BLACK, (x, y, width, height), 2)
    
    # Calculer les valeurs min/max pour l'échelle
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score if max_score != min_score else 1
    
    # Dessiner la grille
    grid_color = LIGHT_GRAY
    for i in range(1, 5):
        # Lignes horizontales
        grid_y = y + (height * i // 5)
        pygame.draw.line(surface, grid_color, (x, grid_y), (x + width, grid_y))
        
        # Lignes verticales
        grid_x = x + (width * i // 5)
        pygame.draw.line(surface, grid_color, (grid_x, y), (grid_x, y + height))
    
    # Dessiner les scores bruts (en gris clair)
    if len(scores) > 1:
        points = []
        for i, score in enumerate(scores):
            px = x + (i * width // (len(scores) - 1))
            py = y + height - int(((score - min_score) / score_range) * height)
            points.append((px, py))
        
        if len(points) > 1:
            pygame.draw.lines(surface, DARK_GRAY, False, points, 1)
    
    # Dessiner la moyenne mobile (en bleu)
    moving_avg = calculate_moving_average(scores, min(10, len(scores)))
    if len(moving_avg) > 1:
        points = []
        for i, avg in enumerate(moving_avg):
            px = x + (i * width // (len(moving_avg) - 1))
            py = y + height - int(((avg - min_score) / score_range) * height)
            points.append((px, py))
        
        if len(points) > 1:
            pygame.draw.lines(surface, BLUE, False, points, 3)
    
    # Dessiner les étiquettes
    font_small = pygame.font.SysFont(None, 18)
    font_medium = pygame.font.SysFont(None, 20)
    
    # Étiquettes des axes
    max_text = font_small.render(f"{max_score:.0f}", True, BLACK)
    surface.blit(max_text, (x - 35, y - 5))
    
    min_text = font_small.render(f"{min_score:.0f}", True, BLACK)
    surface.blit(min_text, (x - 35, y + height - 10))
    
    mid_score = (max_score + min_score) / 2
    mid_text = font_small.render(f"{mid_score:.0f}", True, BLACK)
    surface.blit(mid_text, (x - 35, y + height//2 - 5))
    
    # Nombre d'épisodes
    episodes_text = font_small.render(f"0", True, BLACK)
    surface.blit(episodes_text, (x - 5, y + height + 5))
    
    episodes_end = font_small.render(f"{len(scores)}", True, BLACK)
    surface.blit(episodes_end, (x + width - 20, y + height + 5))

def show_training_graph():
    """Affiche l'écran de graphique des données d'entraînement."""
    scores = load_training_data()
    
    graph_running = True
    while graph_running:
        screen.fill(WHITE)
        
        # Titre
        font_title = pygame.font.SysFont(None, 48)
        title_text = font_title.render("Graphique d'Entraînement", True, BLACK)
        title_rect = title_text.get_rect(center=(WIDTH // 2, 40))
        screen.blit(title_text, title_rect)
        
        if scores:
            # Dessiner le graphique principal
            draw_graph(screen, scores, 80, 80, WIDTH - 160, 300)
            
            # Afficher les statistiques
            font_stats = pygame.font.SysFont(None, 24)
            stats_y = 400
            
            stats = [
                f"Nombre d'épisodes: {len(scores)}",
                f"Score maximum: {max(scores)}",
                f"Score moyen: {sum(scores)/len(scores):.2f}",
                f"Score moyen (100 derniers): {sum(scores[-100:])/min(100, len(scores)):.2f}"
            ]
            
            for i, stat in enumerate(stats):
                stat_text = font_stats.render(stat, True, BLACK)
                screen.blit(stat_text, (80, stats_y + i * 25))
            
            # Légende
            legend_y = 520
            font_legend = pygame.font.SysFont(None, 20)
            
            # Ligne grise pour scores bruts
            pygame.draw.line(screen, DARK_GRAY, (80, legend_y), (110, legend_y), 2)
            legend_text1 = font_legend.render("Scores bruts", True, BLACK)
            screen.blit(legend_text1, (120, legend_y - 8))
            
            # Ligne bleue pour moyenne mobile
            pygame.draw.line(screen, BLUE, (250, legend_y), (280, legend_y), 3)
            legend_text2 = font_legend.render("Moyenne mobile", True, BLACK)
            screen.blit(legend_text2, (290, legend_y - 8))
        else:
            # Message d'erreur si pas de données
            font_error = pygame.font.SysFont(None, 32)
            error_text = font_error.render("Impossible de charger les données d'entraînement", True, RED)
            error_rect = error_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            screen.blit(error_text, error_rect)
            
            font_help = pygame.font.SysFont(None, 24)
            help_text = font_help.render("Vérifiez que le fichier 'geometry_dash_ai_modelv5.pkl' existe", True, BLACK)
            help_rect = help_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 40))
            screen.blit(help_text, help_rect)
        
        # Bouton retour
        back_button = Button("Retour", WIDTH // 2 - 50, HEIGHT - 80, 100, 40, (200, 200, 200), (150, 150, 150))
        
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    graph_running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_clicked = True
        
        back_button.update(mouse_pos)
        back_button.draw(screen)
        
        if back_button.check_clicked(mouse_pos, mouse_clicked):
            graph_running = False
        
        pygame.display.flip()
        clock.tick(30)

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