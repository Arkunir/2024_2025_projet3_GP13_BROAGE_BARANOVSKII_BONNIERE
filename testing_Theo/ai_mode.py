import pygame
import torch
import numpy as np
import os
import sys
import time
from ai_agent import RLAgent

def ai_reinforcement_play(WIDTH=800, HEIGHT=600, FPS=60, GROUND_HEIGHT=500):
    """
    Fonction pour le mode de jeu utilisant l'apprentissage par renforcement
    """
    # Importation des classes nécessaires (pour éviter les importations circulaires)
    from paste import Player, GROUND_HEIGHT, INITIAL_SCROLL_SPEED
    
    # Initialiser Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Geometry Dash - Mode IA par Renforcement")
    clock = pygame.time.Clock()
    
    # Couleurs
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    
    # Initialiser le joueur et les objets du jeu
    player = Player()
    game_objects = []
    score = 0
    last_object = pygame.time.get_ticks()
    
    # Configuration des distances et intervalles (identique à main())
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
    
    object_interval = np.random.randint(*obstacle_intervals[INITIAL_SCROLL_SPEED])
    
    current_speed = INITIAL_SCROLL_SPEED
    previous_speed = current_speed
    
    # Configurations pour les changements de vitesse
    speed_threshold_7 = np.random.randint(10, 20)
    min_threshold_8 = max(25, 2 * speed_threshold_7 - 5)
    max_threshold_8 = 2 * speed_threshold_7 + 10
    speed_threshold_8 = np.random.randint(min_threshold_8, max_threshold_8)
    min_threshold_9 = max(40, 2 * speed_threshold_8 - 15)
    max_threshold_9 = 2 * speed_threshold_8 + 5
    speed_threshold_9 = np.random.randint(min_threshold_9, max_threshold_9)
    speed_threshold_random = 100
    next_random_change = speed_threshold_random + np.random.randint(25, 50)
    
    # Initialiser l'agent d'apprentissage par renforcement
    # État: 12 valeurs (4 caractéristiques pour chacun des 3 obstacles les plus proches)
    # Actions: 2 (sauter ou ne pas sauter)
    state_size = 12
    action_size = 2
    
    # Utiliser CUDA si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = RLAgent(state_size, action_size, device)
    
    # Variables pour l'apprentissage
    done = False
    batch_size = 32
    episode_rewards = 0
    frame_count = 0
    last_action_time = 0
    action_cooldown = 5  # Nombre de frames minimum entre deux actions
    actions_history = []
    
    # Pour le mode d'entraînement ou d'inférence
    TRAINING_MODE = True  # Mettre à False pour le mode démonstration uniquement
    SAVE_INTERVAL = 1000  # Intervalle de sauvegarde du modèle (en frames)
    
    # Initialiser l'état
    state = agent.get_state(player, game_objects, WIDTH)
    
    # Boucle principale
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        frame_count += 1
        
        # Gérer les événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_t:  # Touche pour basculer le mode entraînement
                    TRAINING_MODE = not TRAINING_MODE
                    print(f"Mode d'entraînement: {'activé' if TRAINING_MODE else 'désactivé'}")
        
        # Créer un nouvel obstacle si nécessaire (même logique que dans main())
        can_spawn_obstacle = True
        min_distance = min_obstacle_distances[current_speed]
        
        if game_objects:
            last_obstacle = game_objects[-1]
            
            # Calculer la position de fin du dernier obstacle
            if hasattr(last_obstacle, 'rect'):
                last_obstacle_right = last_obstacle.rect.right
            else:
                last_obstacle_right = last_obstacle.x + getattr(last_obstacle, 'width', 50)
            
            # Vérifier si l'espace est suffisant pour un nouvel obstacle
            if WIDTH - last_obstacle_right < min_distance:
                can_spawn_obstacle = False
        
        # Créer un nouvel obstacle si les conditions sont remplies
        if can_spawn_obstacle and current_time - last_object > object_interval:
            # Importer la fonction pour créer un obstacle de manière cohérente avec le jeu principal
            from paste import create_obstacle_based_on_speed
            obj = create_obstacle_based_on_speed(WIDTH, current_speed)
            
            if obj:
                obj.set_speed(current_speed)
                game_objects.append(obj)
                
                last_object = current_time
                object_interval = np.random.randint(*obstacle_intervals[current_speed])
        
        # Obtenir l'état actuel
        state = agent.get_state(player, game_objects, WIDTH)
        
        # Sélectionner une action avec un cooldown
        if frame_count - last_action_time >= action_cooldown:
            action = agent.act(state)
            last_action_time = frame_count
            
            # Enregistrer l'action pour l'analyse
            actions_history.append(action)
            
            # Exécuter l'action
            if action == 1:  # Sauter
                if not player.is_jumping:
                    player.jump()
        
        # Mettre à jour le joueur et les objets
        player.update(game_objects)
        
        # Vérifier si le jeu est terminé
        reward = 0.1  # Petite récompense pour survivre
        done = False
        
        if not player.is_alive:
            reward = -10  # Pénalité pour avoir perdu
            done = True
            print(f"Game Over! Score: {score}")
        
        # Variables pour suivre les collisions et objets à supprimer
        objects_to_remove = []
        
        # Mettre à jour les objets et vérifier les collisions
        for obj in game_objects[:]:
            obj.update()
            
            # Vérifier les collisions selon le type d'obstacle
            collision = False
            
            if hasattr(obj, 'check_collision') and callable(getattr(obj, 'check_collision')):
                # Utiliser la méthode de collision spécifique de l'objet
                if obj.check_collision(player, pygame.key.get_pressed()):
                    collision = True
            elif hasattr(obj, 'get_rect') and callable(getattr(obj, 'get_rect')):
                # Utiliser un simple test de collision rectangle
                if player.rect.colliderect(obj.get_rect()):
                    collision = True
            elif hasattr(obj, 'get_rects') and callable(getattr(obj, 'get_rects')):
                # Pour les obstacles avec plusieurs rectangles de collision
                for rect in obj.get_rects():
                    if player.rect.colliderect(rect):
                        collision = True
                        break
            
            # Gérer la collision
            if collision:
                player.is_alive = False
                reward = -10
                done = True
                print(f"Game Over! Collision avec {obj.__class__.__name__}")
                break
            
            # Vérifier si l'objet est hors de l'écran
            if hasattr(obj, 'rect') and obj.rect.right < 0:
                objects_to_remove.append(obj)
            elif hasattr(obj, 'x') and obj.x + getattr(obj, 'width', 50) < 0:
                objects_to_remove.append(obj)
        
        # Supprimer les objets hors écran et incrémenter le score
        for obj in objects_to_remove:
            if obj in game_objects:
                game_objects.remove(obj)
                score += 1
                reward += 1  # Récompense supplémentaire pour chaque obstacle passé
                
                # Gestion des changements de vitesse (même que dans main())
                old_speed = current_speed
                
                if score < speed_threshold_random:
                    if score == speed_threshold_7 and current_speed < 7:
                        current_speed = 7
                        print(f"Passage à la vitesse 7 à {score} points!")
                    elif score == speed_threshold_8 and current_speed < 8:
                        current_speed = 8
                        print(f"Passage à la vitesse 8 à {score} points!")
                    elif score == speed_threshold_9 and current_speed < 9:
                        current_speed = 9
                        print(f"Passage à la vitesse 9 à {score} points!")
                elif score == speed_threshold_random:
                    new_speed = np.random.randint(9, 11)
                    if score >= 100:
                        while new_speed == current_speed:
                            new_speed = np.random.randint(9, 11)
                    
                    current_speed = new_speed
                    print(f"Passage à la vitesse aléatoire {new_speed} à {score} points!")
                    next_random_change = score + np.random.randint(25, 50)
                elif score >= speed_threshold_random and score == next_random_change:
                    new_speed = np.random.randint(9, 11)
                    if score >= 100:
                        while new_speed == current_speed:
                            new_speed = np.random.randint(9, 11)
                    
                    current_speed = new_speed
                    print(f"Nouveau changement à la vitesse aléatoire {new_speed} à {score} points!")
                    next_random_change = score + np.random.randint(25, 50)
                
                # Appliquer le changement de vitesse à tous les objets
                for game_obj in game_objects:
                    game_obj.set_speed(current_speed)
                
                # Changement de skin si la vitesse change
                if current_speed != previous_speed:
                    if hasattr(player, 'change_skin_randomly'):
                        player.change_skin_randomly()
                    previous_speed = current_speed
        
        # Calculer le nouvel état
        next_state = agent.get_state(player, game_objects, WIDTH)
        
        # Stocker l'expérience dans la mémoire de l'agent
        if TRAINING_MODE:
            agent.remember(state, action if 'action' in locals() else 0, reward, next_state, done)
            
            # Entraîner l'agent
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            # Sauvegarder le modèle périodiquement
            if frame_count % SAVE_INTERVAL == 0:
                agent.save_model()
        
        # Accumuler les récompenses
        episode_rewards += reward
        
        # Affichage
        screen.fill(WHITE)
        pygame.draw.rect(screen, BLACK, (0, GROUND_HEIGHT, WIDTH, HEIGHT - GROUND_HEIGHT))
        
        player.draw(screen)
        
        for obj in game_objects:
            obj.draw(screen)
        
        # Afficher les informations
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (20, 20))
        
        speed_text = font.render(f"Vitesse: {current_speed}", True, BLACK)
        screen.blit(speed_text, (20, 60))
        
        mode_text = font.render(f"Mode: {'Entraînement' if TRAINING_MODE else 'Démo'}", True, BLACK)
        screen.blit(mode_text, (20, 100))
        
        epsilon_text = font.render(f"Exploration: {agent.epsilon:.2f}", True, BLACK)
        screen.blit(epsilon_text, (20, 140))
        
        next_action_text = font.render(f"Prochaine action dans: {max(0, action_cooldown - (frame_count - last_action_time))}", True, BLACK)
        screen.blit(next_action_text, (20, 180))
        
        if actions_history:
            recent_actions = actions_history[-min(10, len(actions_history)):]
            action_text = font.render(f"Actions récentes: {''.join(['J' if a == 1 else '-' for a in recent_actions])}", True, BLACK)
            screen.blit(action_text, (20, 220))
        
        if score >= speed_threshold_random:
            next_change_text = font.render(f"Prochain changement à: {next_random_change}", True, BLACK)
            screen.blit(next_change_text, (20, 260))
        
        pygame.display.flip()
        clock.tick(FPS)
        
        # Si le jeu est terminé, on recommence
        if done:
            print(f"Épisode terminé: Score = {score}, Récompense totale = {episode_rewards:.2f}")
            
            # Réinitialiser pour un nouvel épisode
            player = Player()
            game_objects = []
            score = 0
            last_object = pygame.time.get_ticks()
            current_speed = INITIAL_SCROLL_SPEED
            previous_speed = current_speed
            episode_rewards = 0
            actions_history = []
            
            # Reconfigurer les seuils pour les changements de vitesse
            speed_threshold_7 = np.random.randint(10, 20)
            min_threshold_8 = max(25, 2 * speed_threshold_7 - 5)
            max_threshold_8 = 2 * speed_threshold_7 + 10
            speed_threshold_8 = np.random.randint(min_threshold_8, max_threshold_8)
            min_threshold_9 = max(40, 2 * speed_threshold_8 - 15)
            max_threshold_9 = 2 * speed_threshold_8 + 5
            speed_threshold_9 = np.random.randint(min_threshold_9, max_threshold_9)
            
            # Attendre un court instant avant de recommencer
            time.sleep(1)
    
    # Sauvegarder le modèle avant de quitter
    if TRAINING_MODE:
        agent.save_model()
    
    pygame.quit()

def best_ai_play(WIDTH=800, HEIGHT=600, FPS=60, GROUND_HEIGHT=500):
    """
    Fonction pour le mode utilisant la meilleure IA (mode démonstration)
    """
    # Importation des classes nécessaires
    from paste import Player, GROUND_HEIGHT, INITIAL_SCROLL_SPEED
    
    # Initialiser Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Geometry Dash - Mode Meilleure IA")
    clock = pygame.time.Clock()
    
    # Couleurs
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    
    # Initialiser le joueur et les objets du jeu
    player = Player()
    game_objects = []
    score = 0
    last_object = pygame.time.get_ticks()
    
    # Configuration identique au mode principal
    min_obstacle_distances = {
        6: 150, 7: 175, 8: 500, 9: 250, 10: 275, 11: 300
    }
    
    obstacle_intervals = {
        6: [800, 1400], 7: [900, 1600], 8: [1200, 1800],
        9: [1300, 2000], 10: [1400, 2100], 11: [1500, 2200]
    }
    
    object_interval = np.random.randint(*obstacle_intervals[INITIAL_SCROLL_SPEED])
    
    current_speed = INITIAL_SCROLL_SPEED
    previous_speed = current_speed
    
    # Configurations pour les changements de vitesse
    speed_threshold_7 = np.random.randint(10, 20)
    min_threshold_8 = max(25, 2 * speed_threshold_7 - 5)
    max_threshold_8 = 2 * speed_threshold_7 + 10
    speed_threshold_8 = np.random.randint(min_threshold_8, max_threshold_8)
    min_threshold_9 = max(40, 2 * speed_threshold_8 - 15)
    max_threshold_9 = 2 * speed_threshold_8 + 5
    speed_threshold_9 = np.random.randint(min_threshold_9, max_threshold_9)
    speed_threshold_random = 100
    next_random_change = speed_threshold_random + np.random.randint(25, 50)
    
    # Initialiser l'agent RL avec le meilleur modèle
    state_size = 12
    action_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = RLAgent(state_size, action_size, device)
    
    # Forcer le chargement du meilleur modèle
    agent.load_model("model/best_model.pth")
    
    # Désactiver l'exploration pour une démonstration pure
    agent.epsilon = 0
    
    # Variables pour le jeu
    frame_count = 0
    last_action_time = 0
    action_cooldown = 3  # Cooldown plus court pour le meilleur modèle
    actions_history = []
    
    # Obtenir l'état initial
    state = agent.get_state(player, game_objects, WIDTH)
    
    # Variables pour les statistiques
    total_episodes = 0
    best_score = 0
    current_episode_start_time = time.time()
    
    # Boucle principale
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        frame_count += 1
        
        # Gérer les événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Créer un nouvel obstacle si nécessaire
        can_spawn_obstacle = True
        min_distance = min_obstacle_distances[current_speed]
        
        if game_objects:
            last_obstacle = game_objects[-1]
            
            # Calculer la position de fin du dernier obstacle
            if hasattr(last_obstacle, 'rect'):
                last_obstacle_right = last_obstacle.rect.right
            else:
                last_obstacle_right = last_obstacle.x + getattr(last_obstacle, 'width', 50)
            
            # Vérifier si l'espace est suffisant pour un nouvel obstacle
            if WIDTH - last_obstacle_right < min_distance:
                can_spawn_obstacle = False
        
        # Créer un nouvel obstacle si les conditions sont remplies
        if can_spawn_obstacle and current_time - last_object > object_interval:
            from paste import create_obstacle_based_on_speed
            obj = create_obstacle_based_on_speed(WIDTH, current_speed)
            
            if obj:
                obj.set_speed(current_speed)
                game_objects.append(obj)
                
                last_object = current_time
                object_interval = np.random.randint(*obstacle_intervals[current_speed])
        
        # Obtenir l'état actuel
        state = agent.get_state(player, game_objects, WIDTH)
        
        # Sélectionner une action (sans exploration)
        if frame_count - last_action_time >= action_cooldown:
            action = agent.act(state, use_epsilon=False)  # Forcer l'utilisation du modèle sans exploration
            last_action_time = frame_count
            
            # Enregistrer l'action pour l'analyse
            actions_history.append(action)
            
            # Exécuter l'action
            if action == 1:  # Sauter
                if not player.is_jumping:
                    player.jump()
        
        # Mettre à jour le joueur et les objets
        player.update(game_objects)
        
        # Vérifier si le jeu est terminé
        done = False
        
        if not player.is_alive:
            done = True
            total_episodes += 1
            episode_duration = time.time() - current_episode_start_time
            print(f"Partie terminée: Score = {score}, Durée = {episode_duration:.1f}s")
            
            # Mettre à jour le meilleur score
            if score > best_score:
                best_score = score
                print(f"Nouveau record: {best_score}!")
        
        # Variables pour suivre les collisions et objets à supprimer
        objects_to_remove = []
        
        # Mettre à jour les objets et vérifier les collisions
        for obj in game_objects[:]:
            obj.update()
            
            # Vérifier les collisions selon le type d'obstacle
            collision = False
            
            if hasattr(obj, 'check_collision') and callable(getattr(obj, 'check_collision')):
                if obj.check_collision(player, pygame.key.get_pressed()):
                    collision = True
            elif hasattr(obj, 'get_rect') and callable(getattr(obj, 'get_rect')):
                if player.rect.colliderect(obj.get_rect()):
                    collision = True
            elif hasattr(obj, 'get_rects') and callable(getattr(obj, 'get_rects')):
                for rect in obj.get_rects():
                    if player.rect.colliderect(rect):
                        collision = True
                        break
            
            # Gérer la collision
            if collision:
                player.is_alive = False
                done = True
                print(f"Game Over! Collision avec {obj.__class__.__name__}")
                break
            
            # Vérifier si l'objet est hors de l'écran
            if hasattr(obj, 'rect') and obj.rect.right < 0:
                objects_to_remove.append(obj)
            elif hasattr(obj, 'x') and obj.x + getattr(obj, 'width', 50) < 0:
                objects_to_remove.append(obj)
        
        # Supprimer les objets hors écran et incrémenter le score
        for obj in objects_to_remove:
            if obj in game_objects:
                game_objects.remove(obj)
                score += 1
                
                # Gestion des changements de vitesse
                old_speed = current_speed
                
                if score < speed_threshold_random:
                    if score == speed_threshold_7 and current_speed < 7:
                        current_speed = 7
                        print(f"Passage à la vitesse 7 à {score} points!")
                    elif score == speed_threshold_8 and current_speed < 8:
                        current_speed = 8
                        print(f"Passage à la vitesse 8 à {score} points!")
                    elif score == speed_threshold_9 and current_speed < 9:
                        current_speed = 9
                        print(f"Passage à la vitesse 9 à {score} points!")
                elif score == speed_threshold_random:
                    new_speed = np.random.randint(9, 11)
                    if score >= 100:
                        while new_speed == current_speed:
                            new_speed = np.random.randint(9, 11)
                    
                    current_speed = new_speed
                    print(f"Passage à la vitesse aléatoire {new_speed} à {score} points!")
                    next_random_change = score + np.random.randint(25, 50)
                elif score >= speed_threshold_random and score == next_random_change:
                    new_speed = np.random.randint(9, 11)
                    if score >= 100:
                        while new_speed == current_speed:
                            new_speed = np.random.randint(9, 11)
                    
                    current_speed = new_speed
                    print(f"Nouveau changement à la vitesse aléatoire {new_speed} à {score} points!")
                    next_random_change = score + np.random.randint(25, 50)
                
                # Appliquer le changement de vitesse à tous les objets
                for game_obj in game_objects:
                    game_obj.set_speed(current_speed)
                
                # Changement de skin si la vitesse change
                if current_speed != previous_speed:
                    if hasattr(player, 'change_skin_randomly'):
                        player.change_skin_randomly()
                    previous_speed = current_speed
        
        # Affichage
        screen.fill(WHITE)
        pygame.draw.rect(screen, BLACK, (0, GROUND_HEIGHT, WIDTH, HEIGHT - GROUND_HEIGHT))
        
        player.draw(screen)
        
        for obj in game_objects:
            obj.draw(screen)
        
        # Afficher les informations
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (20, 20))
        
        speed_text = font.render(f"Vitesse: {current_speed}", True, BLACK)
        screen.blit(speed_text, (20, 60))
        
        best_score_text = font.render(f"Meilleur score: {best_score}", True, BLACK)
        screen.blit(best_score_text, (20, 100))
        
        episodes_text = font.render(f"Parties jouées: {total_episodes}", True, BLACK)
        screen.blit(episodes_text, (20, 140))
        
        time_text = font.render(f"Temps écoulé: {time.time() - current_episode_start_time:.1f}s", True, BLACK)
        screen.blit(time_text, (20, 180))
        
        if actions_history:
            recent_actions = actions_history[-min(10, len(actions_history)):]
            action_text = font.render(f"Actions récentes: {''.join(['J' if a == 1 else '-' for a in recent_actions])}", True, BLACK)
            screen.blit(action_text, (20, 220))
        
        if score >= speed_threshold_random:
            next_change_text = font.render(f"Prochain changement à: {next_random_change}", True, BLACK)
            screen.blit(next_change_text, (20, 260))
        
        pygame.display.flip()
        clock.tick(FPS)
        
        # Si le jeu est terminé, on recommence
        if done:
            # Réinitialiser pour un nouvel épisode
            player = Player()
            game_objects = []
            score = 0
            last_object = pygame.time.get_ticks()
            current_speed = INITIAL_SCROLL_SPEED
            previous_speed = current_speed
            actions_history = []
            current_episode_start_time = time.time()
            
            # Reconfigurer les seuils pour les changements de vitesse
            speed_threshold_7 = np.random.randint(10, 20)
            min_threshold_8 = max(25, 2 * speed_threshold_7 - 5)
            max_threshold_8 = 2 * speed_threshold_7 + 10
            speed_threshold_8 = np.random.randint(min_threshold_8, max_threshold_8)
            min_threshold_9 = max(40, 2 * speed_threshold_8 - 15)
            max_threshold_9 = 2 * speed_threshold_8 + 5
            speed_threshold_9 = np.random.randint(min_threshold_9, max_threshold_9)
            
            # Attendre un court instant avant de recommencer
            time.sleep(1)
    
    pygame.quit()

def human_ai_collaborative_play(WIDTH=800, HEIGHT=600, FPS=60, GROUND_HEIGHT=500):
    """
    Fonction pour le mode de jeu collaboratif Humain-IA
    où l'IA suggère des actions que le joueur peut choisir de suivre
    """
    # Importation des classes nécessaires
    from paste import Player, GROUND_HEIGHT, INITIAL_SCROLL_SPEED
    
    # Initialiser Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Geometry Dash