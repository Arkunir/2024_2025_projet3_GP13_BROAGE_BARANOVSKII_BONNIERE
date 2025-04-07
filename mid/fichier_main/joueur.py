import pygame
import sys
from tools import Player, Platform, Spike, SCREEN_WIDTH, SCREEN_HEIGHT, COLORS, GRAVITY, JUMP_FORCE, GAME_SPEED

class JoueurMode:
    def __init__(self, screen, clock):
        self.screen = screen
        self.clock = clock
        self.player = Player(100, SCREEN_HEIGHT - 100)
        
        # Initialisation de la plateforme "infinie" à la même hauteur
        self.platform = Platform(0, SCREEN_HEIGHT - 50)  # Plateforme avec longueur "infinie"
        self.obstacles = []
        self.generate_level()
        
        # Variables de jeu
        self.scroll_x = 0
        self.score = 0
        self.game_over = False
        self.font = pygame.font.SysFont('Arial', 24)
        
    def generate_level(self):
        # Ici, vous pouvez créer une séquence d'obstacles
        # Pour l'exemple, j'ajoute quelques spikes
        for i in range(10):
            spike = Spike(500 + i * 300, SCREEN_HEIGHT - 50)  # Ajusté pour être au niveau de la plateforme
            self.obstacles.append(spike)
    
    def update(self):
        # Mise à jour du joueur
        self.player.update(pygame.key.get_pressed())
        
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
        
        # Affichage du score
        score_text = self.font.render(f"Score: {self.score}", True, COLORS["text"])
        self.screen.blit(score_text, (10, 10))
        
        # Écran de game over
        if self.game_over:
            game_over_text = self.font.render("GAME OVER - Appuyez sur ESPACE pour recommencer", True, COLORS["text"])
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
                    if event.key == pygame.K_SPACE:
                        if self.game_over:
                            # Recommencer avec la barre d'espace si game over
                            self.__init__(self.screen, self.clock)
                        else:
                            # Sauter avec la barre d'espace pendant le jeu
                            self.player.jump()
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            if not self.game_over:
                self.update()
            
            self.draw()
            self.clock.tick(60)
