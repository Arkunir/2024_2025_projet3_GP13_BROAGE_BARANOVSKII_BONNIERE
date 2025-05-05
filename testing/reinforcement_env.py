# # reinforcement_env.py
# import pygame
# import numpy as np
# import random
# import sys
# import os
# from klass import Player
# from klass import Obstacle, DoublePikes, TriplePikes, QuadruplePikes
# from klass import Block, BlockGapBlockWithSpike, BouncingObstacle
# from klass import DoubleBlockPillar, FivePikesWithOrb, JumpPad
# from klass import QuintuplePikesWithJumpPad, PurpleOrb, JumppadOrbsObstacle

# class GeometryDashEnv:
#     """Environnement d'apprentissage par renforcement pour Geometry Dash Clone"""
    
#     def __init__(self, render_mode=None):
#         # Paramètres du jeu
#         self.WIDTH, self.HEIGHT = 800, 600
#         self.GROUND_HEIGHT = 500
#         self.FPS = 60
#         self.INITIAL_SCROLL_SPEED = 6
        
#         # Couleurs
#         self.WHITE = (255, 255, 255)
#         self.BLACK = (0, 0, 0)
#         self.RED = (255, 0, 0)
        
#         # Mode de rendu (pour visualiser ou non le jeu pendant l'entraînement)
#         self.render_mode = render_mode
#         if render_mode:
#             pygame.init()
#             self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
#             pygame.display.set_caption("Geometry Dash RL Environment")
#             self.clock = pygame.time.Clock()
#             self.font = pygame.font.SysFont(None, 36)
        
#         # Initialisation de l'environnement
#         self.reset()
        
#         # Espaces d'action et d'observation
#         # Action: 0 = ne pas sauter, 1 = sauter
#         self.action_space = 2
        
#         # Observation: voir méthode _get_state()
#         # Format: [player_y, player_velocity, distances aux 3 prochains obstacles, hauteurs des 3 prochains obstacles]
#         self.observation_space = 8
    
#     def reset(self):
#         """Réinitialise l'environnement et retourne l'état initial"""
#         self.player = Player()
#         self.game_objects = []
#         self.score = 0
#         self.steps = 0
#         self.current_speed = self.INITIAL_SCROLL_SPEED
#         self.last_object = pygame.time.get_ticks() if pygame.get_init() else 0
#         self.object_interval = random.randint(800, 1400)
#         self.game_over = False
        
#         # Créer quelques obstacles initiaux pour avoir un état valide
#         self._add_initial_obstacles()
        
#         return self._get_state()
    
#     def _add_initial_obstacles(self):
#         """Ajoute quelques obstacles initiaux à des positions aléatoires"""
#         # Ajouter 3 obstacles simples à des positions espacées
#         for i in range(3):
#             x_pos = self.WIDTH + i * 400  # Espacer les obstacles
#             obstacle = Obstacle(x_pos)
#             obstacle.set_speed(self.current_speed)
#             self.game_objects.append(obstacle)
    
#     def step(self, action):
#         """
#         Exécute une étape de l'environnement avec l'action donnée.
        
#         Args:
#             action: 0 (ne pas sauter) ou 1 (sauter)
            
#         Returns:
#             state: le nouvel état
#             reward: la récompense
#             done: si l'épisode est terminé
#             info: informations supplémentaires
#         """
#         self.steps += 1
#         reward = 0.1  # Petite récompense pour survivre
        
#         # Appliquer l'action
#         keys = [False] * 323  # Simuler le tableau de pygame.key.get_pressed()
#         if action == 1:  # Sauter
#             keys[pygame.K_SPACE] = True
#             # Si le joueur est sur le sol ou vient d'atterrir, sauter
#             if not self.player.is_jumping or self.player.just_landed:
#                 self.player.jump()
        
#         # Mettre à jour le joueur
#         self.player.update(self.game_objects, keys_pressed=keys)
        
#         # Vérifier si le joueur est mort
#         if not self.player.is_alive:
#             reward = -10.0  # Grosse pénalité pour mourir
#             self.game_over = True
        
#         # Gestion des obstacles
#         self._manage_obstacles()
        
#         # Mettre à jour tous les objets de jeu
#         objects_to_remove = []
#         for obj in self.game_objects:
#             obj.update()
            
#             # Vérifier si l'objet est sorti de l'écran
#             if self._is_object_offscreen(obj):
#                 objects_to_remove.append(obj)
        
#         # Supprimer les obstacles sortis de l'écran et augmenter le score
#         for obj in objects_to_remove:
#             if obj in self.game_objects:
#                 self.game_objects.remove(obj)
#                 self.score += 1
#                 reward += 1.0  # Récompense pour avoir passé un obstacle
        
#         # Afficher le jeu si demandé
#         if self.render_mode:
#             self._render_frame()
        
#         return self._get_state(), reward, self.game_over, {"score": self.score}
    
#     def _is_object_offscreen(self, obj):
#         """Vérifie si un objet est sorti de l'écran"""
#         if hasattr(obj, 'rect'):
#             return obj.rect.right < 0
#         elif hasattr(obj, 'x') and hasattr(obj, 'width'):
#             return obj.x + obj.width < 0
#         return False
    
#     def _manage_obstacles(self):
#         """Gère la création de nouveaux obstacles"""
#         current_time = pygame.time.get_ticks() if pygame.get_init() else self.steps * (1000 / self.FPS)
        
#         # Vérifier s'il faut créer un nouvel obstacle
#         can_spawn = True
#         min_distance = 150 + (self.current_speed - 6) * 25  # Distance minimale basée sur la vitesse
        
#         # Vérifier l'espacement par rapport au dernier obstacle
#         if self.game_objects:
#             last_obj = self.game_objects[-1]
#             last_right = 0
            
#             if hasattr(last_obj, 'rect'):
#                 last_right = last_obj.rect.right
#             elif hasattr(last_obj, 'x') and hasattr(last_obj, 'width'):
#                 last_right = last_obj.x + last_obj.width
            
#             if self.WIDTH - last_right < min_distance:
#                 can_spawn = False
        
#         # Créer un nouvel obstacle si possible
#         if can_spawn and current_time - self.last_object > self.object_interval:
#             obj = self._create_random_obstacle()
#             obj.set_speed(self.current_speed)
#             self.game_objects.append(obj)
            
#             self.last_object = current_time
#             self.object_interval = random.randint(800, 1400)
    
#     def _create_random_obstacle(self):
#         """Crée un obstacle aléatoire adapté à la vitesse actuelle"""
#         choice = random.random()
        
#         if self.current_speed <= 7:
#             if choice < 0.4:
#                 return Obstacle(self.WIDTH)
#             elif choice < 0.7:
#                 return Block(self.WIDTH)
#             else:
#                 return JumpPad(self.WIDTH)
#         else:
#             if choice < 0.2:
#                 return Obstacle(self.WIDTH)
#             elif choice < 0.4:
#                 return DoublePikes(self.WIDTH)
#             elif choice < 0.6:
#                 return Block(self.WIDTH)
#             elif choice < 0.8:
#                 return JumpPad(self.WIDTH)
#             else:
#                 return TriplePikes(self.WIDTH)
    
#     def _get_state(self):
#         """
#         Récupère l'état actuel de l'environnement sous forme de tableau numpy.
        
#         L'état comprend:
#         - Position Y du joueur (normalisée)
#         - Vitesse Y du joueur (normalisée)
#         - Distance aux 3 prochains obstacles (normalisées)
#         - Hauteur/type des 3 prochains obstacles (normalisée)
#         """
#         # Position et vitesse du joueur
#         player_y = self.player.y / self.HEIGHT
#         player_vel = self.player.velocity / 20.0  # Normaliser par une vitesse maximale approximative
        
#         # Trouver les 3 prochains obstacles
#         next_obstacles = sorted([obj for obj in self.game_objects], 
#                                key=lambda obj: obj.x if hasattr(obj, 'x') else obj.rect.x)
        
#         # Distances et hauteurs pour les 3 prochains obstacles (ou padding avec des valeurs par défaut)
#         distances = []
#         heights = []
        
#         for i in range(3):
#             if i < len(next_obstacles):
#                 obj = next_obstacles[i]
                
#                 # Distance à l'obstacle (normalisée par la largeur de l'écran)
#                 if hasattr(obj, 'x'):
#                     dist = (obj.x - self.player.x) / self.WIDTH
#                 else:
#                     dist = (obj.rect.x - self.player.x) / self.WIDTH
#                 distances.append(max(0, min(1, dist)))  # Clamp entre 0 et 1
                
#                 # Hauteur/type d'obstacle (normalisée)
#                 # Utiliser une valeur différente selon le type d'obstacle
#                 if isinstance(obj, Obstacle):
#                     heights.append(0.2)
#                 elif isinstance(obj, Block):
#                     heights.append(0.4)
#                 elif isinstance(obj, DoublePikes):
#                     heights.append(0.5)
#                 elif isinstance(obj, TriplePikes):
#                     heights.append(0.6)
#                 elif isinstance(obj, JumpPad):
#                     heights.append(0.3)
#                 else:
#                     heights.append(0.7)  # Valeur par défaut pour autres obstacles
#             else:
#                 # Padding avec des valeurs par défaut si moins de 3 obstacles
#                 distances.append(1.0)  # Distance maximale
#                 heights.append(0.0)    # Hauteur minimale/absence d'obstacle
        
#         # Créer et retourner le vecteur d'état
#         state = np.array([
#             player_y,
#             player_vel,
#             *distances,
#             *heights
#         ], dtype=np.float32)
        
#         return state
    
#     def _render_frame(self):
#         """Affiche l'état actuel du jeu à l'écran"""
#         self.screen.fill(self.WHITE)
        
#         # Dessiner le sol
#         pygame.draw.rect(self.screen, self.BLACK, (0, self.GROUND_HEIGHT, self.WIDTH, self.HEIGHT - self.GROUND_HEIGHT))
        
#         # Dessiner le joueur
#         self.player.draw(self.screen)
        
#         # Dessiner tous les objets
#         for obj in self.game_objects:
#             obj.draw(self.screen)
        
#         # Afficher le score et la vitesse
#         score_text = self.font.render(f"Score: {self.score}", True, self.BLACK)
#         speed_text = self.font.render(f"Vitesse: {self.current_speed}", True, self.BLACK)
#         steps_text = self.font.render(f"Étapes: {self.steps}", True, self.BLACK)
        
#         self.screen.blit(score_text, (20, 20))
#         self.screen.blit(speed_text, (20, 60))
#         self.screen.blit(steps_text, (20, 100))
        
#         pygame.display.flip()
#         self.clock.tick(self.FPS)
    
#     def render(self):
#         """Méthode publique pour forcer le rendu hors de step()"""
#         if self.render_mode:
#             self._render_frame()
    
#     def close(self):
#         """Ferme proprement l'environnement"""
#         if self.render_mode and pygame.get_init():
#             pygame.quit()