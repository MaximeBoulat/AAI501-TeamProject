# sensors.py
import numpy as np

DIRECTIONS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
def read_sensors(grid, pos):
    """
    For each of 8 directions, scan 1–3 cells ahead.
    Return distance-to-obstacle as sensor readings.
    """
    readings = []
    for dx,dy in DIRECTIONS:
        dist = 0
        x,y = pos
        for step in range(1,4):
            nx, ny = x + dx*step, y + dy*step
            if nx < 0 or nx >= grid.shape[0] or ny < 0 or ny >= grid.shape[1] or grid[nx,ny]==1:
                dist = step
                break
        readings.append(dist)
    return np.array(readings)
