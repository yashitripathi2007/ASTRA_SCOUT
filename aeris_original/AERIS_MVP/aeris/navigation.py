import heapq
import math
import numpy as np

NEIGHBOURS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

def plan_route(z, slope, roughness, hazard, known, start, goal, unknown_penalty=2.5):
    """A* over an 8-connected grid with a simple terrain-risk cost."""
    h, w = z.shape
    pq = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while pq:
        _, current = heapq.heappop(pq)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g_score[goal]
        closed.add(current)

        for dy, dx in NEIGHBOURS:
            ny, nx = current[0]+dy, current[1]+dx
            nxt = (ny, nx)
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            # Critical information boundary: the planner may inspect terrain
            # measurements only in cells that its agent has actually observed.
            # Unknown cells receive a conservative prior, not hidden ground truth.
            if known[ny, nx]:
                cell_hazard = hazard[ny, nx]
                cell_slope = slope[ny, nx]
                cell_roughness = roughness[ny, nx]
                if cell_hazard >= 0.98:
                    continue
                unknown_cost = 0.0
            else:
                cell_hazard = 0.35
                cell_slope = 12.0
                cell_roughness = 0.5
                unknown_cost = unknown_penalty
            distance = math.sqrt(2) if dy and dx else 1.0
            terrain_cost = 1.0 + 2.8*cell_hazard + 0.04*cell_slope + 0.15*cell_roughness
            step = distance * (terrain_cost + unknown_cost)
            tentative = g_score[current] + step
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                heuristic = math.hypot(goal[0]-ny, goal[1]-nx)
                heapq.heappush(pq, (tentative + heuristic, nxt))
    return None, float("inf")

def simulate_rover_only(size, start, sensor_radius=5):
    """Initial rover knowledge: local circular sensor footprint around start."""
    yy, xx = np.mgrid[0:size, 0:size]
    return (yy-start[0])**2 + (xx-start[1])**2 <= sensor_radius**2

def route_energy(path, slope, roughness, hazard):
    """
    Relative simplified Wh-like estimate for demo comparisons.
    Not a calibrated physical rover power model.
    """
    if not path:
        return float("inf")
    energy = 0.0
    for a, b in zip(path[:-1], path[1:]):
        d = math.hypot(b[0]-a[0], b[1]-a[1])
        energy += d * (0.10 + 0.004*slope[b] + 0.025*roughness[b] + 0.12*hazard[b])
    return energy
