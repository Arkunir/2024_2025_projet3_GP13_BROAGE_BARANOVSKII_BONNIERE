import pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import numpy as np
from collections import deque
import threading
import time

class TrainingVisualizer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.running = True
        
        # Données pour les graphiques
        self.episodes = deque(maxlen=200)
        self.scores = deque(maxlen=200)
        self.rewards = deque(maxlen=200)
        self.epsilons = deque(maxlen=200)
        self.avg_scores = deque(maxlen=200)
        self.survival_times = deque(maxlen=200)
        
        # Statistiques actuelles
        self.current_episode = 0
        self.current_score = 0
        self.current_reward = 0
        self.current_epsilon = 1.0
        self.current_avg_score = 0
        self.current_survival_time = 0
        self.best_score = 0
        self.total_states = 0
        
        # Thread pour la fenêtre
        self.thread = threading.Thread(target=self._run_window, daemon=True)
        self.lock = threading.Lock()
        
    def start(self):
        """Démarre la fenêtre de visualisation dans un thread séparé"""
        self.thread.start()
        
    def stop(self):
        """Arrête la fenêtre de visualisation"""
        self.running = False
        
    def update_data(self, episode, score, reward, epsilon, avg_score, survival_time, best_score, total_states):
        """Met à jour les données à afficher"""
        with self.lock:
            self.current_episode = episode
            self.current_score = score
            self.current_reward = reward
            self.current_epsilon = epsilon
            self.current_avg_score = avg_score
            self.current_survival_time = survival_time
            self.best_score = best_score
            self.total_states = total_states
            
            # Ajouter aux historiques
            self.episodes.append(episode)
            self.scores.append(score)
            self.rewards.append(reward)
            self.epsilons.append(epsilon)
            self.avg_scores.append(avg_score)
            self.survival_times.append(survival_time)
    
    def _create_plots(self):
        """Crée les graphiques matplotlib"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor('white')
        
        with self.lock:
            if len(self.episodes) < 2:
                return fig
                
            episodes_list = list(self.episodes)
            scores_list = list(self.scores)
            rewards_list = list(self.rewards)
            epsilons_list = list(self.epsilons)
            avg_scores_list = list(self.avg_scores)
            survival_times_list = list(self.survival_times)
        
        # Graphique 1: Scores
        ax1.clear()
        ax1.plot(episodes_list, scores_list, 'b-', alpha=0.6, linewidth=1, label='Score')
        ax1.plot(episodes_list, avg_scores_list, 'r-', linewidth=2, label='Score moyen')
        ax1.axhline(y=self.best_score, color='g', linestyle='--', label=f'Meilleur: {self.best_score}')
        ax1.set_title('Evolution des Scores', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Récompenses
        ax2.clear()
        ax2.plot(episodes_list, rewards_list, 'purple', alpha=0.7, linewidth=1)
        ax2.set_title('Récompenses par Episode', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Récompense Totale')
        ax2.grid(True, alpha=0.3)
        
        # Graphique 3: Epsilon (exploration)
        ax3.clear()
        ax3.plot(episodes_list, epsilons_list, 'orange', linewidth=2)
        ax3.set_title('Taux d\'Exploration (Epsilon)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.grid(True, alpha=0.3)
        
        # Graphique 4: Temps de survie
        ax4.clear()
        ax4.plot(episodes_list, survival_times_list, 'green', alpha=0.7, linewidth=1)
        ax4.set_title('Temps de Survie', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Temps (secondes)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _run_window(self):
        """Fonction principale de la fenêtre de visualisation"""
        pygame.init()
        # Créer une nouvelle fenêtre pygame séparée
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Visualisation Entraînement IA - Geometry Dash")
        clock = pygame.time.Clock()
        
        font_title = pygame.font.Font(None, 32)
        font_stats = pygame.font.Font(None, 24)
        font_small = pygame.font.Font(None, 20)
        
        # Couleurs
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        BLUE = (0, 100, 200)
        GREEN = (0, 150, 0)
        RED = (200, 0, 0)
        GRAY = (128, 128, 128)
        
        last_plot_update = 0
        plot_surface = None
        
        while self.running:
            current_time = time.time()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
            
            screen.fill(WHITE)
            
            # Titre
            title = font_title.render("Visualisation Entraînement IA", True, BLUE)
            title_rect = title.get_rect(center=(self.width // 2, 30))
            screen.blit(title, title_rect)
            
            # Statistiques en temps réel
            with self.lock:
                stats_y = 70
                
                stats = [
                    f"Episode: {self.current_episode}",
                    f"Score Actuel: {self.current_score}",
                    f"Meilleur Score: {self.best_score}",
                    f"Score Moyen: {self.current_avg_score:.1f}",
                    f"Récompense: {self.current_reward:.1f}",
                    f"Epsilon: {self.current_epsilon:.4f}",
                    f"Temps Survie: {self.current_survival_time:.1f}s",
                    f"États Connus: {self.total_states}"
                ]
                
                for i, stat in enumerate(stats):
                    color = GREEN if "Meilleur" in stat else BLACK
                    text = font_stats.render(stat, True, color)
                    screen.blit(text, (20, stats_y + i * 25))
            
            # Mise à jour des graphiques toutes les 2 secondes
            if current_time - last_plot_update > 2.0 and len(self.episodes) > 1:
                try:
                    fig = self._create_plots()
                    
                    # Convertir matplotlib en surface pygame
                    canvas = agg.FigureCanvasAgg(fig)
                    canvas.draw()
                    renderer = canvas.get_renderer()
                    raw_data = renderer.tostring_rgb()
                    size = canvas.get_width_height()
                    
                    plot_surface = pygame.image.fromstring(raw_data, size, 'RGB')
                    plot_surface = pygame.transform.scale(plot_surface, (self.width - 40, 350))
                    
                    plt.close(fig)  # Libérer la mémoire
                    last_plot_update = current_time
                    
                except Exception as e:
                    print(f"Erreur lors de la création des graphiques: {e}")
            
            # Afficher les graphiques
            if plot_surface:
                screen.blit(plot_surface, (20, 240))
            else:
                # Message d'attente
                waiting_text = font_small.render("En attente de données pour les graphiques...", True, GRAY)
                screen.blit(waiting_text, (20, 300))
            
            # Instructions
            instructions = [
                "Fermez cette fenêtre pour arrêter la visualisation",
                "Les graphiques se mettent à jour automatiquement"
            ]
            
            for i, instruction in enumerate(instructions):
                text = font_small.render(instruction, True, GRAY)
                screen.blit(text, (20, self.height - 40 + i * 20))
            
            pygame.display.flip()
            clock.tick(30)  # 30 FPS pour la visualisation
            
# Au début de la fonction
        visualizer = TrainingVisualizer()
        visualizer.start()

    # Dans la boucle d'entraînement, après chaque épisode
        visualizer.update_data(
            episode=current_episode,
            score=score,
            reward=total_reward,
            epsilon=agent.epsilon,
            avg_score=avg_score,
            survival_time=episode_duration,
            best_score=agent.high_score,
            total_states=len(agent.q_table)
            )

    # A la fin
        visualizer.stop()