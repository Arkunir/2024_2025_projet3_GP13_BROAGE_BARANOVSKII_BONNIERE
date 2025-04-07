import pygame
import sys
from tools import Player, Platform, Spike, SCREEN_WIDTH, SCREEN_HEIGHT, COLORS, GRAVITY, JUMP_FORCE, GAME_SPEED

class RobotMode:
    def __init__(self, screen, clock):
        self.screen = screen
        self.clock = clock
        self.player = Player(100, SCREEN_HEIGHT - 100)
        
        # Initialisation des obstacles et plateforme
        self.platform = Platform(0, SCREEN_HEIGHT - 50, SCREEN_WIDTH * 3)
        self.obstacles = []
        self.generate_level()
        
        # Variables de jeu
        self.scroll_x = 0
        self.score = 0
        self.game_over = False
        self.font = pygame.font.SysFont('Arial', 24)
        
        # Configuration du robot
        self.robot_active = True
        # Paramètres de chaque obstacle (détection, distance saut, nb sauts)
        self.obstacle_params = {}
        self.init_robot_params()
        
    def init_robot_params(self):
        # Initialiser les paramètres pour chaque obstacle
        for i, obstacle in enumerate(self.obstacles):
            # Par défaut: détecter à 100px, sauter à 50px, 1 saut
            self.obstacle_params[i] = {
                "detection_distance": 100,
                "jump_distance": 50,
                "jump_count": 1
            }
    
    def generate_level(self):
        # Création du niveau avec obstacles
        for i in range(10):
            spike = Spike(500 + i * 300, SCREEN_HEIGHT - 70)
            self.obstacles.append(spike)
    
    def robot_logic(self):
        # Logique du robot pour sauter automatiquement
        for i, obstacle in enumerate(self.obstacles):
            # Distance entre le joueur et l'obstacle
            distance = obstacle.rect.x - self.player.rect.right
            params = self.obstacle_params.get(i, {"detection_distance": 100, "jump_distance": 50, "jump_count": 1})
            
            # Si on est dans la zone de détection
            if 0 < distance < params["detection_distance"]:
                # Si on est à la distance de saut
                if distance < params["jump_distance"] and self.player.on_ground:
                    # Effectuer le nombre de sauts nécessaires
                    for _ in range(params["jump_count"]):
                        self.player.jump()
                        # Petite pause entre les sauts multiples si nécessaire
                        if params["jump_count"] > 1:
                            pygame.time.delay(50)
    
    def update(self):
        # Mise à jour du joueur
        self.player.update(pygame.key.get_pressed())
        
        # Robot automatique
        if self.robot_active:
            self.robot_logic()
        
        # Défilement
        self.scroll_x = self.player.rect.x - 100
        if self.scroll_x < 0:
            self.scroll_x = 0
        
        # Collision avec la plateforme
        if self.player.rect.colliderect(self.platform.rect):
            self.player.rect.bottom = self.platform.rect.top
            self.player.on_ground = True
            self.player.velocity_y = 0
        
        # Collision avec les obstacles
        for obstacle in self.obstacles:
            if self.player.rect.colliderect(obstacle.rect):
                self.game_over = True
        
        # Score basé sur la distance
        self.score = self.player.rect.x // 100
        
    def draw(self):
        self.screen.fill(COLORS["background"])
        
        # Dessin des éléments avec décalage de scrolling
        self.platform.draw(self.screen, self.scroll_x)
        for obstacle in self.obstacles:
            obstacle.draw(self.screen, self.scroll_x)
        
        # Dessin du joueur
        self.player.draw(self.screen, self.scroll_x)
        
        # Affichage du score et statut robot
        score_text = self.font.render(f"Score: {self.score}", True, COLORS["text"])
        robot_text = self.font.render(f"Robot: {'Actif' if self.robot_active else 'Inactif'}", True, COLORS["text"])
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(robot_text, (10, 40))
        
        # Interface des paramètres du robot
        if self.robot_active:
            # Affichage de base des paramètres
            robot_params = self.font.render("Params Robot (F1: éditer)", True, COLORS["text"])
            self.screen.blit(robot_params, (SCREEN_WIDTH - 250, 10))
        
        # Écran de game over
        if self.game_over:
            game_over_text = self.font.render("GAME OVER - Press R to Restart", True, COLORS["text"])
            self.screen.blit(game_over_text, (SCREEN_WIDTH//2 - game_over_text.get_width()//2, SCREEN_HEIGHT//2))
        
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.robot_active:
                        self.player.jump()
                    if event.key == pygame.K_r and self.game_over:
                        # Réinitialisation
                        self.__init__(self.screen, self.clock)
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_F1:
                        # Interface pour éditer les paramètres (pour une version plus avancée)
                        pass
                    if event.key == pygame.K_TAB:
                        # Activer/désactiver le robot
                        self.robot_active = not self.robot_active
            
            if not self.game_over:
                self.update()
            
            self.draw()
            self.clock.tick(60)
