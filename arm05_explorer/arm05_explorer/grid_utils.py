import math
from collections import deque
 
import numpy as np

NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]
 

def yaw_from_quaternion(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
 
 
def quaternion_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)
 

def world_to_cell(x, y, origin_x, origin_y, resolution):
    col = int(math.floor((x - origin_x) / resolution))
    row = int(math.floor((y - origin_y) / resolution))
    return row, col
 
 
def cell_to_world(row, col, origin_x, origin_y, resolution):
    x = origin_x + (col + 0.5) * resolution
    y = origin_y + (row + 0.5) * resolution
    return x, y
 
def bresenham(r0, c0, r1, c1):
    cells = []
    dc = abs(c1 - c0)
    dr = -abs(r1 - r0)
    sc = 1 if c1 >= c0 else -1
    sr = 1 if r1 >= r0 else -1
    err = dc + dr
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 >= dr:
            err += dr
            c += sc
        if e2 <= dc:
            err += dc
            r += sr
    return cells
 
def bfs_distances(passable, start):
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    r0, c0 = start
    if not (0 <= r0 < h and 0 <= c0 < w) or not passable[r0, c0]:
        return dist
    dist[r0, c0] = 0
    queue = deque([(r0, c0)])
    while queue:
        r, c = queue.popleft()
        d = dist[r, c] + 1
        for dr, dc in NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = d
                queue.append((nr, nc))
    return dist
 
 
def nearest_true_cell(mask, cell):
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    d2 = (rows - cell[0]) ** 2 + (cols - cell[1]) ** 2
    i = int(np.argmin(d2))
    return int(rows[i]), int(cols[i])