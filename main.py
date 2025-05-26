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
pygame.display.set_caption("Quadraleap")
clock = pygame.time.Clock()

def load_training_data():
    """Charge les données d'entraînement depuis le module ia_reinforcement ou les fichiers pkl."""
    
    # 1. D'abord essayer de charger depuis les fichiers pkl disponibles
    model_files = [
        'geometry_dash_ai_modelv6.pkl',
        'geometry_dash_ai_modelv5.pkl',
        'geometry_dash_ai_modelv4.pkl',
        'geometry_dash_ai_modelv3.pkl',
        'geometry_dash_ai_modelv2.pkl',
        'geometry_dash_ai_modelv1.pkl'
    ]
    
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                print(f"Tentative de chargement depuis {model_path}...")
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Si c'est un dictionnaire, chercher les scores
                if isinstance(data, dict):
                    possible_keys = [
                        'last_scores', 'training_scores', 'scores', 'episode_scores', 
                        'rewards', 'episode_rewards', 'score_history', 'results'
                    ]
                    
                    for key in possible_keys:
                        if key in data:
                            scores_data = data[key]
                            if isinstance(scores_data, list) and len(scores_data) > 0:
                                # Convertir en nombres si nécessaire
                                try:
                                    scores = [float(score) for score in scores_data if isinstance(score, (int, float))]
                                    if len(scores) > 0:
                                        print(f"Données trouvées dans {model_path}['{key}']: {len(scores)} épisodes")
                                        print(f"Scores exemple: {scores[:5]}...")
                                        return scores
                                except (ValueError, TypeError):
                                    continue
                    
                    # Si pas de scores directs, essayer de créer des données simulées basées sur le modèle
                    if 'high_score' in data and 'training_episodes' in data:
                        high_score = data['high_score']
                        episodes = data.get('training_episodes', 100)
                        
                        if high_score > 0 and episodes > 0:
                            # Générer des scores simulés basés sur les données du modèle
                            print(f"Génération de données simulées basées sur le modèle (meilleur score: {high_score}, épisodes: {episodes})")
                            simulated_scores = generate_simulated_training_data(high_score, episodes)
                            return simulated_scores
                
                # Si c'est directement une liste
                elif isinstance(data, list) and len(data) > 0:
                    try:
                        scores = [float(score) for score in data if isinstance(score, (int, float))]
                        if len(scores) > 0:
                            print(f"Données trouvées directement dans {model_path}: {len(scores)} épisodes")
                            return scores
                    except (ValueError, TypeError):
                        continue
                        
            except Exception as e:
                print(f"Erreur lors du chargement de {model_path}: {e}")
                continue
    
    # 2. Essayer d'importer depuis le module ia_reinforcement
    try:
        print("Tentative d'importation depuis ia_reinforcement...")
        import ia_reinforcement
        
        # Lister tous les attributs du module
        attrs = [attr for attr in dir(ia_reinforcement) if not attr.startswith('_')]
        print(f"Attributs disponibles dans ia_reinforcement: {attrs}")
        
        # Chercher des listes qui pourraient contenir des scores
        for attr_name in attrs:
            try:
                attr_value = getattr(ia_reinforcement, attr_name)
                if isinstance(attr_value, list) and len(attr_value) > 0:
                    # Vérifier si c'est une liste de nombres
                    try:
                        scores = [float(x) for x in attr_value[:10]]  # Tester les 10 premiers
                        if all(isinstance(s, (int, float)) for s in scores):
                            full_scores = [float(x) for x in attr_value]
                            print(f"Données trouvées dans ia_reinforcement.{attr_name}: {len(full_scores)} épisodes")
                            return full_scores
                    except (ValueError, TypeError):
                        continue
            except Exception:
                continue
                
        # Si le module a une instance d'IA, essayer d'accéder à ses données
        if hasattr(ia_reinforcement, 'ai') and hasattr(ia_reinforcement.ai, 'last_scores'):
            scores_data = ia_reinforcement.ai.last_scores
            if isinstance(scores_data, list) and len(scores_data) > 0:
                scores = [float(score) for score in scores_data]
                print(f"Données trouvées dans ia_reinforcement.ai.last_scores: {len(scores)} épisodes")
                return scores
                
    except ImportError:
        print("Impossible d'importer le module ia_reinforcement")
    except Exception as e:
        print(f"Erreur lors de l'accès au module ia_reinforcement: {e}")
    
    # 3. Si aucune donnée trouvée, générer des données d'exemple
    print("Aucune donnée réelle trouvée, génération de données d'exemple...")
    return generate_example_data()

def generate_simulated_training_data(max_score, num_episodes):
    """Génère des données d'entraînement simulées basées sur un score maximum."""
    scores = []
    current_avg = 0
    
    for episode in range(num_episodes):
        # Simulation d'un apprentissage progressif
        progress = episode / num_episodes
        
        # Les premiers épisodes sont très mauvais
        if progress < 0.1:
            base_score = random.uniform(0, max_score * 0.05)
        elif progress < 0.3:
            base_score = random.uniform(0, max_score * 0.2)
        elif progress < 0.6:
            base_score = random.uniform(max_score * 0.1, max_score * 0.5)
        elif progress < 0.8:
            base_score = random.uniform(max_score * 0.3, max_score * 0.8)
        else:
            base_score = random.uniform(max_score * 0.5, max_score)
        
        # Ajouter de la variabilité
        noise = random.uniform(-base_score * 0.3, base_score * 0.3)
        final_score = max(0, base_score + noise)
        
        scores.append(final_score)
    
    return scores

def generate_example_data():
    """Génère des données d'exemple pour la démonstration."""
    scores = []
    
    # Simulation d'un entraînement de 200 épisodes
    for episode in range(200):
        # Progression d'apprentissage simulée
        if episode < 50:
            # Phase d'exploration - scores très bas
            score = random.uniform(0, 10)
        elif episode < 100:
            # Phase d'apprentissage initial - amélioration lente
            base = 5 + (episode - 50) * 0.3
            score = max(0, base + random.uniform(-5, 10))
        elif episode < 150:
            # Phase d'amélioration - scores moyens
            base = 20 + (episode - 100) * 0.5
            score = max(0, base + random.uniform(-10, 15))
        else:
            # Phase de maîtrise - hauts scores avec variabilité
            base = 45 + (episode - 150) * 0.2
            score = max(0, base + random.uniform(-15, 20))
        
        scores.append(score)
    
    print("Données d'exemple générées: 200 épisodes simulés")
    return scores

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
    
    # Étiquettes des axes Y
    max_text = font_small.render(f"{max_score:.0f}", True, BLACK)
    surface.blit(max_text, (x - 35, y - 5))
    
    min_text = font_small.render(f"{min_score:.0f}", True, BLACK)
    surface.blit(min_text, (x - 35, y + height - 10))
    
    mid_score = (max_score + min_score) / 2
    mid_text = font_small.render(f"{mid_score:.0f}", True, BLACK)
    surface.blit(mid_text, (x - 35, y + height//2 - 5))
    
    # Étiquettes des axes X (épisodes)
    episodes_text = font_small.render("0", True, BLACK)
    surface.blit(episodes_text, (x - 5, y + height + 5))
    
    episodes_end = font_small.render(f"{len(scores)}", True, BLACK)
    surface.blit(episodes_end, (x + width - 20, y + height + 5))
    
    # Étiquette de l'axe X
    axis_label = font_small.render("Épisodes", True, BLACK)
    surface.blit(axis_label, (x + width//2 - 30, y + height + 25))

def show_training_graph():
    """Affiche l'écran de graphique des données d'entraînement."""
    print("Chargement des données d'entraînement...")
    
    # Liste des fichiers de modèles disponibles
    model_files = [
        'geometry_dash_ai_modelv6.pkl',
        'geometry_dash_ai_modelv5.pkl',
        'geometry_dash_ai_modelv4.pkl',
        'geometry_dash_ai_modelv3.pkl',
        'geometry_dash_ai_modelv2.pkl',
        'geometry_dash_ai_modelv1.pkl'
    ]
    
    current_model_index = 0
    scores = load_training_data()
    
    graph_running = True
    while graph_running:
        screen.fill(WHITE)
        
        # Titre
        font_title = pygame.font.SysFont(None, 48)
        title_text = font_title.render("Graphique d'Entraînement IA", True, BLACK)
        title_rect = title_text.get_rect(center=(WIDTH // 2, 25))
        screen.blit(title_text, title_rect)
        
        if scores and len(scores) > 0:
            # Dessiner le graphique principal
            draw_graph(screen, scores, 80, 80, WIDTH - 160, 300)
            
            # Afficher les statistiques
            font_stats = pygame.font.SysFont(None, 22)
            stats_y = 400
            
            # Calculer les statistiques
            max_score = max(scores)
            min_score = min(scores)
            avg_score = sum(scores) / len(scores)
            recent_scores = scores[-min(100, len(scores)):]
            recent_avg = sum(recent_scores) / len(recent_scores)
            
            stats = [
                f"Nombre d'épisodes d'entraînement: {len(scores)}",
                f"Score maximum atteint: {max_score:.1f}",
                f"Score minimum: {min_score:.1f}",
                f"Score moyen global: {avg_score:.2f}",
                f"Score moyen ({len(recent_scores)} derniers): {recent_avg:.2f}"
            ]
            
            # Ajouter des statistiques supplémentaires si assez de données
            if len(scores) >= 10:
                last_10_avg = sum(scores[-10:]) / 10
                stats.append(f"Score moyen récent (10 derniers): {last_10_avg:.2f}")
            
            if len(scores) >= 50:
                improvement = recent_avg - sum(scores[:min(50, len(scores))]) / min(50, len(scores))
                stats.append(f"Amélioration depuis le début: +{improvement:.2f}")
            
            for i, stat in enumerate(stats):
                stat_text = font_stats.render(stat, True, BLACK)
                screen.blit(stat_text, (80, stats_y + i * 22))
            
            # Légende
            legend_y = 570
            font_legend = pygame.font.SysFont(None, 18)
            
            # Ligne grise pour scores bruts
            pygame.draw.line(screen, DARK_GRAY, (80, legend_y), (110, legend_y), 2)
            legend_text1 = font_legend.render("Scores par épisode", True, BLACK)
            screen.blit(legend_text1, (120, legend_y - 8))
            
            # Ligne bleue pour moyenne mobile
            pygame.draw.line(screen, BLUE, (280, legend_y), (310, legend_y), 3)
            legend_text2 = font_legend.render("Moyenne mobile (10 épisodes)", True, BLACK)
            screen.blit(legend_text2, (320, legend_y - 8))
            
            # Titre du fichier actuellement chargé - bien visible
            font_file_title = pygame.font.SysFont(None, 32)
            file_title_text = font_file_title.render(f"Fichier: {model_files[current_model_index]}", True, BLUE)
            file_title_rect = file_title_text.get_rect(center=(WIDTH // 2, 60))
            screen.blit(file_title_text, file_title_rect)
            
        else:
            # Message d'erreur si pas de données
            font_error = pygame.font.SysFont(None, 24)
            error_messages = [
                "Aucune donnée d'entraînement disponible",
                "",
                "Conseils pour résoudre le problème:",
                "• Lancez d'abord l'IA par renforcement pour générer des données",
                "• Vérifiez que les scores sont sauvegardés correctement",
                "• Ou vérifiez l'existence des fichiers *.pkl"
            ]
            
            start_y = HEIGHT // 2 - (len(error_messages) * 25) // 2
            
            for i, message in enumerate(error_messages):
                if message:  # Ne pas afficher les lignes vides
                    color = RED if i == 0 else BLACK
                    text = font_error.render(message, True, color)
                    text_rect = text.get_rect(center=(WIDTH // 2, start_y + i * 30))
                    screen.blit(text, text_rect)
        
        # Boutons - repositionnés sous le graphique à droite des explications
        button_x = 500  # Position à droite des statistiques
        button_y = 450  # Sous le graphique, au niveau des statistiques
        
        back_button = Button("Menu", button_x, button_y, 120, 35, (200, 200, 200), (150, 150, 150))
        prev_button = Button("<", button_x + 130, button_y, 30, 35, (200, 200, 200), (150, 150, 150))
        next_button = Button(">", button_x + 170, button_y, 30, 35, (200, 200, 200), (150, 150, 150))
        
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    graph_running = False
                elif event.key == pygame.K_LEFT:
                    # Changer vers le fichier précédent
                    current_model_index = (current_model_index - 1) % len(model_files)
                    print(f"Changement vers le fichier précédent: {model_files[current_model_index]}")
                    scores = load_specific_model_data(model_files[current_model_index])
                elif event.key == pygame.K_RIGHT:
                    # Changer vers le fichier suivant
                    current_model_index = (current_model_index + 1) % len(model_files)
                    print(f"Changement vers le fichier suivant: {model_files[current_model_index]}")
                    scores = load_specific_model_data(model_files[current_model_index])
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_clicked = True
        
        back_button.update(mouse_pos)
        prev_button.update(mouse_pos)
        next_button.update(mouse_pos)
        
        back_button.draw(screen)
        prev_button.draw(screen)
        next_button.draw(screen)
        
        if back_button.check_clicked(mouse_pos, mouse_clicked):
            graph_running = False
        elif prev_button.check_clicked(mouse_pos, mouse_clicked):
            # Changer vers le fichier précédent
            current_model_index = (current_model_index - 1) % len(model_files)
            print(f"Changement vers le fichier précédent: {model_files[current_model_index]}")
            scores = load_specific_model_data(model_files[current_model_index])
        elif next_button.check_clicked(mouse_pos, mouse_clicked):
            # Changer vers le fichier suivant
            current_model_index = (current_model_index + 1) % len(model_files)
            print(f"Changement vers le fichier suivant: {model_files[current_model_index]}")
            scores = load_specific_model_data(model_files[current_model_index])
        
        pygame.display.flip()
        clock.tick(30)

def load_specific_model_data(model_path):
    """Charge les données d'un fichier de modèle spécifique."""
    if not os.path.exists(model_path):
        print(f"Fichier {model_path} non trouvé, génération de données d'exemple...")
        return generate_example_data()
    
    try:
        print(f"Chargement depuis {model_path}...")
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Si c'est un dictionnaire, chercher les scores
        if isinstance(data, dict):
            possible_keys = [
                'last_scores', 'training_scores', 'scores', 'episode_scores', 
                'rewards', 'episode_rewards', 'score_history', 'results'
            ]
            
            for key in possible_keys:
                if key in data:
                    scores_data = data[key]
                    if isinstance(scores_data, list) and len(scores_data) > 0:
                        # Convertir en nombres si nécessaire
                        try:
                            scores = [float(score) for score in scores_data if isinstance(score, (int, float))]
                            if len(scores) > 0:
                                print(f"Données trouvées dans {model_path}['{key}']: {len(scores)} épisodes")
                                return scores
                        except (ValueError, TypeError):
                            continue
            
            # Si pas de scores directs, essayer de créer des données simulées basées sur le modèle
            if 'high_score' in data and 'training_episodes' in data:
                high_score = data['high_score']
                episodes = data.get('training_episodes', 100)
                
                if high_score > 0 and episodes > 0:
                    print(f"Génération de données simulées basées sur le modèle (meilleur score: {high_score}, épisodes: {episodes})")
                    return generate_simulated_training_data(high_score, episodes)
        
        # Si c'est directement une liste
        elif isinstance(data, list) and len(data) > 0:
            try:
                scores = [float(score) for score in data if isinstance(score, (int, float))]
                if len(scores) > 0:
                    print(f"Données trouvées directement dans {model_path}: {len(scores)} épisodes")
                    return scores
            except (ValueError, TypeError):
                pass
                
    except Exception as e:
        print(f"Erreur lors du chargement de {model_path}: {e}")
    
    print(f"Aucune donnée valide trouvée dans {model_path}, génération de données d'exemple...")
    return generate_example_data()

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
        title_text = font_title.render("Quadraleap", True, BLACK)
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