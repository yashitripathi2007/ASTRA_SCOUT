import numpy as np

def scout_observation(size, start, goal, radius=14):
    """
    Simulated top-down camera survey: observes a broad corridor ahead.
    This is a visibility/coverage abstraction, not photogrammetric reconstruction.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    # Scout surveys the corridor between start and goal, with a small lateral sweep.
    cx = (start[1] + goal[1]) / 2
    cy = (start[0] + goal[0]) / 2
    along = (xx >= start[1] + 4) & (xx <= goal[1] - 2)
    corridor = np.abs(yy - cy) <= radius
    # Add two offset reconnaissance lobes to make scouting spatially selective.
    lobe1 = (xx - (size * .42))**2 + (yy - (cy - radius*.7))**2 <= (radius*.65)**2
    lobe2 = (xx - (size * .68))**2 + (yy - (cy + radius*.7))**2 <= (radius*.65)**2
    return (along & corridor) | lobe1 | lobe2
