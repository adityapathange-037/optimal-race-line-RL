import pygame
import numpy as np
import sys
import math

# --- CONFIGURATION & HYPERPARAMETERS ---
WIDTH, HEIGHT = 1000, 700
TRACK_WIDTH = 10  # Grid cells wide
GRID_SIZE = 15    # Size of each cell in pixels
GAMMA = 0.95      # Discount factor
MAX_SPEED = 6
MIN_SPEED = 1

# Colors
COLOR_WALL = (40, 40, 40)
COLOR_TRACK = (100, 100, 100)
COLOR_LINE = (255, 255, 255)
COLOR_BRAKE = (255, 0, 0)       # Red
COLOR_ACCEL = (0, 255, 0)       # Green
COLOR_MAINTAIN = (0, 150, 255)  # Blue

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Racing Line Finder (Bellman Equation)")
clock = pygame.time.Clock()

# --- TRACK GENERATION (S-Curve) ---
# Create a grid representation: 0 = Wall, 1 = Track, 2 = Finish Line
grid_w = WIDTH // GRID_SIZE
grid_h = HEIGHT // GRID_SIZE
track_grid = np.zeros((grid_w, grid_h), dtype=int)

# Define a center line for a smooth track curve
center_points = []
for x in range(5, grid_w - 5):
    # Create an S-shape using a sine wave
    y = int(grid_h / 2 + math.sin(x * 0.15) * (grid_h / 4))
    center_points.append((x, y))

# Thicken the center line to create a constant 10-cell wide track
for x, y in center_points:
    for ty in range(y - TRACK_WIDTH // 2, y + TRACK_WIDTH // 2):
        if 0 <= ty < grid_h:
            track_grid[x, ty] = 1

# Define Finish Line at the end of the track
finish_x = center_points[-1][0]
finish_y_center = center_points[-1][1]
for ty in range(finish_y_center - TRACK_WIDTH // 2, finish_y_center + TRACK_WIDTH // 2):
    if 0 <= ty < grid_h:
        track_grid[finish_x, ty] = 2

# --- RL AGENT SETUP (Value Iteration) ---
# State: (x, y, speed)
# Actions: -1 (Brake), 0 (Maintain), 1 (Accelerate)
# For simplicity in 2D grid, direction is always moving left-to-right (+1 in X), 
# and the agent can choose to change its Y offset by {-1, 0, 1} to steer.
ACTIONS = [-1, 0, 1] # Accel/Brake choices
STEER_ACTIONS = [-1, 0, 1] # Y-axis changes

# Initialize Value Table: V(x, y, speed)
V = np.zeros((grid_w, grid_h, MAX_SPEED + 1))
Policy = {} # Stores best (speed_change, steer_change) for each state

print("Running Bellman Value Iteration... Please wait a moment.")

# Value Iteration (Offline learning before visualization)
for iteration in range(20): # 20 iterations is enough for this track length
    delta = 0
    for x in range(grid_w - 2, -1, -1): # Iterate backwards from finish line
        for y in range(grid_h):
            if track_grid[x, y] == 0:
                continue # Skip walls
            
            for speed in range(MIN_SPEED, MAX_SPEED + 1):
                v_old = V[x, y, speed]
                max_val = -float('inf')
                best_action = (0, 0)
                
                # Check all combinations of accelerating/braking and steering
                for dv in ACTIONS:
                    for dy in STEER_ACTIONS:
                        next_speed = np.clip(speed + dv, MIN_SPEED, MAX_SPEED)
                        next_x = x + next_speed
                        next_y = y + dy
                        
                        # Boundary checks
                        if next_x >= grid_w:
                            continue
                        if next_y < 0 or next_y >= grid_h:
                            continue
                            
                        # Calculate Reward
                        if track_grid[next_x, next_y] == 2: # Hit finish line
                            reward = 100.0
                        elif track_grid[next_x, next_y] == 0: # Hit wall/Off-track
                            reward = -50.0
                        else:
                            # Standard step cost: penalize taking too long, penalize high speed turns
                            # Friction constraint: if speed is max, steering costs more
                            reward = -1.0 - (abs(dy) * 0.5 if next_speed == MAX_SPEED else 0)
                        
                        # Bellman Equation update step
                        next_val = reward + GAMMA * V[next_x, next_y, next_speed]
                        if next_val > max_val:
                            max_val = next_val
                            best_action = (dv, dy)
                            
                V[x, y, speed] = max_val
                Policy[(x, y, speed)] = best_action

print("Value Iteration Complete! Drawing the optimal line...")

# --- EXTRACT OPTIMAL PATH ---
optimal_path = []
start_x = center_points[0][0]

START_OFFSET = -3

start_y = center_points[0][1] + START_OFFSET

curr_speed = MAX_SPEED-4
curr_state = (start_x, start_y, curr_speed)

while curr_state[0] < finish_x:
    cx, cy, cv = curr_state
    if track_grid[cx, cy] == 0 or (cx, cy, cv) not in Policy:
        break # Path safely broke out or hit a wall
        
    dv, dy = Policy[(cx, cy, cv)]
    
    # Track metrics for rendering
    action_type = "maintain"
    if dv > 0: action_type = "accel"
    elif dv < 0: action_type = "brake"
    
    optimal_path.append(((cx, cy), cv, action_type))
    
    # Move to next state
    next_speed = np.clip(cv + dv, MIN_SPEED, MAX_SPEED)
    curr_state = (cx + next_speed, cy + dy, next_speed)

# --- MAIN PYGAME LOOP ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    screen.fill(COLOR_WALL)
    
    # Draw Track Background
    for x in range(grid_w):
        for y in range(grid_h):
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            if track_grid[x, y] == 1:
                pygame.draw.rect(screen, COLOR_TRACK, rect)
            elif track_grid[x, y] == 2:
                pygame.draw.rect(screen, (200, 200, 0), rect) # Yellow finish line

    # Draw Optimal Racing Line & Braking/Accelerating Points
    font = pygame.font.SysFont(None, 18)
    for i, ((gx, gy), speed, act) in enumerate(optimal_path):
        pos_x = gx * GRID_SIZE + GRID_SIZE // 2
        pos_y = gy * GRID_SIZE + GRID_SIZE // 2
        
        # Color-code based on action taken at that node
        if act == "accel":
            color = COLOR_ACCEL
        elif act == "brake":
            color = COLOR_BRAKE
        else:
            color = COLOR_MAINTAIN
            
        # Draw the node point
        pygame.draw.circle(screen, color, (pos_x, pos_y), 6)
        
        # Connect nodes with lines
        if i < len(optimal_path) - 1:
            next_gx, next_gy = optimal_path[i+1][0]
            next_pos_x = next_gx * GRID_SIZE + GRID_SIZE // 2
            next_pos_y = next_gy * GRID_SIZE + GRID_SIZE // 2
            pygame.draw.line(screen, COLOR_LINE, (pos_x, pos_y), (next_pos_x, next_pos_y), 2)
            
        # Text overlay for speed every few steps so it doesn't clutter
        if i % 2 == 0:
            img = font.render(f"v:{speed}", True, (255, 255, 255))
            screen.blit(img, (pos_x - 10, pos_y - 18))

    # Legend UI
    pygame.draw.rect(screen, (20, 20, 20), pygame.Rect(10, 10, 260, 95))
    ui_font = pygame.font.SysFont(None, 22)
    screen.blit(ui_font.render("GREEN = Accelerating Point", True, COLOR_ACCEL), (20, 20))
    screen.blit(ui_font.render("RED = Braking Point", True, COLOR_BRAKE), (20, 45))
    screen.blit(ui_font.render("BLUE = Maintaining Speed", True, COLOR_MAINTAIN), (20, 70))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()