import pygame
import sys
from joueur import JoueurMode
from robot import RobotMode
from tools import SCREEN_WIDTH, SCREEN_HEIGHT, COLORS

class Menu:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Geometry Dash Clone")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 30)
        
        # Boutons
        self.button_joueur = pygame.Rect(SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 60, 200, 50)
        self.button_robot = pygame.Rect(SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 + 10, 200, 50)
        
    def draw(self):
        self.screen.fill(COLORS["background"])
        
        # Titre
        title = self.font.render("GEOMETRY DASH CLONE", True, COLORS["text"])
        self.screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, SCREEN_HEIGHT//4))
        
        # Boutons
        pygame.draw.rect(self.screen, COLORS["button"], self.button_joueur)
        pygame.draw.rect(self.screen, COLORS["button"], self.button_robot)
        
        # Texte des boutons
        text_joueur = self.font.render("JOUEUR", True, COLORS["text"])
        text_robot = self.font.render("ROBOT", True, COLORS["text"])
        
        self.screen.blit(text_joueur, (self.button_joueur.x + self.button_joueur.width//2 - text_joueur.get_width()//2, 
                                    self.button_joueur.y + self.button_joueur.height//2 - text_joueur.get_height()//2))
        self.screen.blit(text_robot, (self.button_robot.x + self.button_robot.width//2 - text_robot.get_width()//2, 
                                   self.button_robot.y + self.button_robot.height//2 - text_robot.get_height()//2))
        
        pygame.display.flip()
    
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if self.button_joueur.collidepoint(mouse_pos):
                        # Lancer le mode joueur
                        joueur_mode = JoueurMode(self.screen, self.clock)
                        joueur_mode.run()
                    elif self.button_robot.collidepoint(mouse_pos):
                        # Lancer le mode robot
                        robot_mode = RobotMode(self.screen, self.clock)
                        robot_mode.run()
            
            self.draw()
            self.clock.tick(60)

if __name__ == "__main__":
    menu = Menu()
    menu.run()
