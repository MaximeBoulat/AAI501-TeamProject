import numpy as np
import time
from typing import List, Tuple, Optional
from world import World
from agent import Agent

class Renderer:
    
    
    def render_world(self, world: World, agent: Agent):
        
        raise NotImplementedError
    
    

class ConsoleRenderer(Renderer):
    
    
    def __init__(self, show_path_history: bool = True):
        
        self.show_path_history = show_path_history
        
    def render_world(self, world: World, agent: Agent, full_path: List[Tuple[int, int]] = None):
        
        # Create display grid
        display = np.full(world.grid.shape, ".", dtype=str)
        
        # Add obstacles
        for y in range(world.size):
            for x in range(world.size):
                if world.grid[y][x] == 1:
                    display[y][x] = "#"
        
        
        # Show agent's path history
        if self.show_path_history:
            for x, y in agent.path_history[:-1]:  # Exclude current position
                if (x, y) != world.start and (x, y) != world.goal:
                    display[y][x] = "*"
        
        # Mark start, goal, and agent positions
        start_x, start_y = world.start
        goal_x, goal_y = world.goal
        agent_x, agent_y = agent.position
        
        display[start_y][start_x] = "S"
        display[goal_y][goal_x] = "G"
        display[agent_y][agent_x] = "A"
        
        # Print the display
        print("\n" + "=" * (world.size * 2 + 1))
        for row in display:
            print(" " + " ".join(row))
        print("=" * (world.size * 2 + 1))
        
        # Print status information
        sensors = agent.get_current_sensors()
        distance = agent.get_distance_to_goal()
        steps = len(agent.path_history) - 1
        
        print(f"Position: {agent.position} | Distance to goal: {distance:.2f} | Steps taken: {steps}")
        print(f"Sensors: {sensors}")
        
        if agent.is_at_goal():
            print("🎉 GOAL REACHED!")
    
    def render_step_info(self, step: int, action: Optional[int], success: bool):
        
        action_names = ["left", "left+down", "down", "right+down", 
                       "right", "right+up", "up", "left+up"]
        
        if action is not None:
            action_name = action_names[action]
            status = "✓" if success else "✗"
            print(f"Step {step}: Action {action} ({action_name}) {status}")
        else:
            print(f"Step {step}: No action available")
    
    


class PygameRenderer(Renderer):
    
    def __init__(self, cell_size: int = 30, show_path_history: bool = True, 
                 window_title: str = "Robot Navigation"):

        try:
            import pygame
            self.pygame = pygame
        except ImportError:
            raise ImportError("pygame is required for PygameRenderer. Install with: pip install pygame")
        
        self.cell_size = cell_size
        self.show_path_history = show_path_history
        self.window_title = window_title
        
        # Colors (RGB)
        self.colors = {
            'background': (255, 255, 255),      # White
            'obstacle': (0, 0, 0),              # Black
            'free_space': (240, 240, 240),      # Light gray
            'start': (0, 255, 0),               # Green
            'goal': (255, 0, 0),                # Red
            'agent': (0, 0, 255),               # Blue
            'path_history': (150, 150, 255),    # Light blue
            'full_path': (200, 200, 200),       # Gray
            'grid_lines': (180, 180, 180)       # Light gray
        }
        
        # Initialize pygame
        self.pygame.init()
        self.screen = None
        self.clock = self.pygame.time.Clock()
        self.font = self.pygame.font.Font(None, 24)
        self.running = True
        
    def _init_display(self, world_size: int):

        if self.screen is None:
            window_size = world_size * self.cell_size
            info_height = 100  # Space for status info
            self.screen = self.pygame.display.set_mode((window_size, window_size + info_height))
            self.pygame.display.set_caption(self.window_title)
            self.world_size = world_size
            self.info_height = info_height
    
    def render_world(self, world: World, agent: Agent, full_path: List[Tuple[int, int]] = None):
        
        if not self.running:
            return
            
        self._init_display(world.size)
        
        # Handle pygame events
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.running = False
                return
        
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw grid cells
        for y in range(world.size):
            for x in range(world.size):
                rect = self.pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                      self.cell_size, self.cell_size)
                
                # Determine cell color
                if world.grid[y][x] == 1:  # Obstacle
                    color = self.colors['obstacle']
                else:  # Free space
                    color = self.colors['free_space']
                
                # Draw cell
                self.pygame.draw.rect(self.screen, color, rect)
                
                # Draw grid lines
                self.pygame.draw.rect(self.screen, self.colors['grid_lines'], rect, 1)
        
        
        
        # Draw agent's path history
        if self.show_path_history:
            for pos in agent.path_history[:-1]:  # Exclude current position
                x, y = pos
                if (x, y) != world.start and (x, y) != world.goal:
                    rect = self.pygame.Rect(x * self.cell_size + 8, y * self.cell_size + 8,
                                          self.cell_size - 16, self.cell_size - 16)
                    self.pygame.draw.rect(self.screen, self.colors['path_history'], rect)
        
        # Draw start position
        start_x, start_y = world.start
        start_rect = self.pygame.Rect(start_x * self.cell_size + 3, start_y * self.cell_size + 3,
                                    self.cell_size - 6, self.cell_size - 6)
        self.pygame.draw.rect(self.screen, self.colors['start'], start_rect)
        
        # Draw goal position
        goal_x, goal_y = world.goal
        goal_rect = self.pygame.Rect(goal_x * self.cell_size + 3, goal_y * self.cell_size + 3,
                                   self.cell_size - 6, self.cell_size - 6)
        self.pygame.draw.rect(self.screen, self.colors['goal'], goal_rect)
        
        # Draw agent position
        agent_x, agent_y = agent.position
        agent_center = (agent_x * self.cell_size + self.cell_size // 2,
                       agent_y * self.cell_size + self.cell_size // 2)
        self.pygame.draw.circle(self.screen, self.colors['agent'], agent_center, 
                              self.cell_size // 3)
        
        # Draw status information
        self._draw_status_info(world, agent)
        
        # Update display
        self.pygame.display.flip()
        
        # Check if goal reached
        if agent.is_at_goal():
            self._show_success_message()
    
    def _draw_status_info(self, world: World, agent: Agent):
        
        info_y = world.size * self.cell_size + 10
        
        # Position info
        pos_text = f"Position: {agent.position}"
        pos_surface = self.font.render(pos_text, True, (0, 0, 0))
        self.screen.blit(pos_surface, (10, info_y))
        
        # Distance to goal
        distance = agent.get_distance_to_goal()
        dist_text = f"Distance to goal: {distance:.2f}"
        dist_surface = self.font.render(dist_text, True, (0, 0, 0))
        self.screen.blit(dist_surface, (10, info_y + 25))
        
        # Steps taken
        steps = len(agent.path_history) - 1
        steps_text = f"Steps taken: {steps}"
        steps_surface = self.font.render(steps_text, True, (0, 0, 0))
        self.screen.blit(steps_surface, (10, info_y + 50))
        
        # Sensor readings (abbreviated)
        sensors = agent.get_current_sensors()
        sensor_text = f"Sensors: [{', '.join(str(s) for s in sensors[:4])}...]"
        sensor_surface = self.font.render(sensor_text, True, (0, 0, 0))
        self.screen.blit(sensor_surface, (250, info_y))
    
    def _show_success_message(self):
        
        success_text = "GOAL REACHED!"
        success_surface = self.font.render(success_text, True, (0, 128, 0))
        text_rect = success_surface.get_rect()
        text_rect.center = (self.world_size * self.cell_size // 2, 
                           self.world_size * self.cell_size + 75)
        self.screen.blit(success_surface, text_rect)
    
    def render_step_info(self, step: int, action: Optional[int], success: bool):
        
        action_names = ["left", "left+down", "down", "right+down", 
                       "right", "right+up", "up", "left+up"]
        
        if action is not None:
            action_name = action_names[action]
            status = "✓" if success else "✗"
            title = f"{self.window_title} - Step {step}: {action_name} {status}"
        else:
            title = f"{self.window_title} - Step {step}: No action"
        
        if self.running:
            self.pygame.display.set_caption(title)
    
    
    def close(self):
        
        if self.screen is not None:
            self.pygame.quit()
            self.running = False
    
    def __del__(self):
        
        try:
            self.close()
        except:
            pass 