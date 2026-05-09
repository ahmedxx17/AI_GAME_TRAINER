import pygame
import torch
import sys
import os
import random

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import FlappyBirdEnv
from agent import DQNAgent
from config import FPS

SCREEN_W = FlappyBirdEnv.SCREEN_W
SCREEN_H = FlappyBirdEnv.SCREEN_H

def run_vs_mode():
    pygame.init()
    
    # Create a wider window for side-by-side play (2x width)
    screen = pygame.display.set_mode((SCREEN_W * 2, SCREEN_H))
    pygame.display.set_caption("Flappy Bird: Human vs AI")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 32, bold=True)
    small_font = pygame.font.SysFont("Arial", 20)

    # Generate a random seed so both environments get IDENTICAL pipes
    seed = random.randint(0, 1000000)
    
    env_human = FlappyBirdEnv(seed=seed)
    env_ai = FlappyBirdEnv(seed=seed)

    # Reset both to start
    env_human.reset()
    state_ai = env_ai.reset()

    # Load the trained AI agent
    agent = DQNAgent()
    if os.path.exists("model.pth"):
        agent.model.load_state_dict(torch.load("model.pth", weights_only=True))
        agent.epsilon = 0.0  # Zero exploration, play perfectly
        print("Loaded fully trained AI from 'model.pth'!")
    else:
        print("Warning: 'model.pth' not found. AI will just guess randomly.")

    # Subsurfaces for rendering left and right screens
    human_surface = screen.subsurface((0, 0, SCREEN_W, SCREEN_H))
    ai_surface = screen.subsurface((SCREEN_W, 0, SCREEN_W, SCREEN_H))

    running = True
    human_done = False
    ai_done = False
    
    # Track states manually here
    human_score = 0
    ai_score = 0
    
    print("\n" + "="*50)
    print("  HUMAN VS AI RACE STARTED!")
    print("  Press Spacebar to flap the left bird.")
    print("  Press 'R' at any time to Restart a new race.")
    print("="*50 + "\n")

    while running:
        human_flap = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not human_done:
                    human_flap = True
                elif event.key == pygame.K_r:
                    # Restart Race
                    seed = random.randint(0, 1000000)
                    env_human = FlappyBirdEnv(seed=seed)
                    env_ai = FlappyBirdEnv(seed=seed)
                    env_human.reset()
                    state_ai = env_ai.reset()
                    human_done = False
                    ai_done = False

        # --- Human Step ---
        if not human_done:
            # 1 = flap, 0 = do nothing
            action_h = 1 if human_flap else 0
            _, _, human_done = env_human.step(action_h)
            human_score = env_human.score
        
        # --- AI Step ---
        if not ai_done:
            # AI uses its neural network to pick the best action
            action_ai = agent.choose_action(state_ai)
            state_ai, _, ai_done = env_ai.step(action_ai)
            ai_score = env_ai.score

        # --- Rendering ---
        screen.fill((0, 0, 0))
        
        # Render the environments into their respective subsurfaces
        env_human.render(human_surface)
        env_ai.render(ai_surface)
        
        # Draw a divider line down the middle
        pygame.draw.line(screen, (200, 200, 200), (SCREEN_W, 0), (SCREEN_W, SCREEN_H), 4)

        # Draw Labels
        human_lbl = font.render("YOU (Spacebar)", True, (255, 255, 255))
        ai_lbl = font.render("AI AGENT", True, (255, 255, 255))
        
        # Shadow for readability
        human_shadow = font.render("YOU (Spacebar)", True, (0, 0, 0))
        ai_shadow = font.render("AI AGENT", True, (0, 0, 0))
        
        screen.blit(human_shadow, (SCREEN_W // 2 - human_lbl.get_width() // 2 + 2, 32))
        screen.blit(human_lbl, (SCREEN_W // 2 - human_lbl.get_width() // 2, 30))
        
        screen.blit(ai_shadow, (SCREEN_W + SCREEN_W // 2 - ai_lbl.get_width() // 2 + 2, 32))
        screen.blit(ai_lbl, (SCREEN_W + SCREEN_W // 2 - ai_lbl.get_width() // 2, 30))

        # Game Over / Winner Text
        if human_done and ai_done:
            msg = "IT'S A TIE!" if human_score == ai_score else ("AI WINS!" if ai_score > human_score else "YOU WIN!")
            color = (255, 255, 50) if human_score == ai_score else ((255, 50, 50) if ai_score > human_score else (50, 255, 50))
            
            end_msg = font.render(msg, True, color)
            screen.blit(end_msg, (SCREEN_W - end_msg.get_width() // 2, SCREEN_H // 2 - 50))
            
            restart_msg = small_font.render("Press 'R' to race again", True, (200, 200, 200))
            screen.blit(restart_msg, (SCREEN_W - restart_msg.get_width() // 2, SCREEN_H // 2 + 20))
            
        elif human_done:
            msg = font.render("CRASHED!", True, (255, 50, 50))
            screen.blit(msg, (SCREEN_W // 2 - msg.get_width() // 2, SCREEN_H // 2))
            
        elif ai_done:
            msg = font.render("CRASHED!", True, (255, 50, 50))
            screen.blit(msg, (SCREEN_W + SCREEN_W // 2 - msg.get_width() // 2, SCREEN_H // 2))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    run_vs_mode()
