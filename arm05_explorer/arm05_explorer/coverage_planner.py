import math
import os
 
import cv2
import numpy as np
 
try:
    from arm05_explorer.grid_utils import (bfs_distances, bresenham, cell_to_world,
                                           nearest_true_cell, world_to_cell)
except ImportError:
    from grid_utils import (bfs_distances, bresenham, cell_to_world,
                            nearest_true_cell, world_to_cell)
 
 
def downsample(grid, resolution, coarse_resolution):
    f = max(1, int(round(coarse_resolution / resolution)))
    h, w = grid.shape
    hh, ww = int(math.ceil(h / f)), int(math.ceil(w / f))
    padded = np.full((hh * f, ww * f), -1, dtype=np.int16)
    padded[:h, :w] = grid
    blocks = padded.reshape(hh, f, ww, f)
    blocked = ((blocks > 50) | (blocks < 0)).any(axis=(1, 3))
    return ~blocked, resolution * f
 
 
def precompute_rays(range_cells, min_range_cells, n_rays):
    rays = []
    for k in range(n_rays):
        a = 2.0 * math.pi * k / n_rays
        er = int(round(range_cells * math.sin(a)))
        ec = int(round(range_cells * math.cos(a)))
        cells = bresenham(0, 0, er, ec)[1:]
        ray = [(dr, dc, min_range_cells <= math.hypot(dr, dc) <= range_cells)
               for dr, dc in cells]
        rays.append(ray)
    return rays
 
 
def visible_cells(r, c, blocked, rays, h, w):
    seen = set()
    for ray in rays:
        for dr, dc, countable in ray:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w or blocked[rr][cc]:
                break
            if countable:
                seen.add(rr * w + cc)
    return np.fromiter(seen, dtype=np.int64, count=len(seen))
 
 
def plan_viewpoints(grid, resolution, origin_x, origin_y, start_xy,
                    coarse_resolution=0.2, robot_clearance=0.35,
                    target_wall_margin=0.25, sensor_range=2.5, sensor_min_range=0.3,
                    candidate_step=0.4, coverage_target=0.97, min_gain_cells=5,
                    n_rays=180, max_viewpoints=60):
    free, res = downsample(grid, resolution, coarse_resolution)
    h, w = free.shape
    blocked_list = (~free).tolist()
 
    dist_m = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) * res
    safe = free & (dist_m > robot_clearance)
 
    start_rc = world_to_cell(start_xy[0], start_xy[1], origin_x, origin_y, res)
    start_rc = (min(max(start_rc[0], 0), h - 1), min(max(start_rc[1], 0), w - 1))
 
    start_safe = start_rc if safe[start_rc] else nearest_true_cell(safe, start_rc)
    if start_safe is None:
        raise RuntimeError('Brak bezpiecznych kratek na mapie.')
    start_free = start_rc if free[start_rc] else nearest_true_cell(free, start_rc)

    reachable = bfs_distances(safe, start_safe) >= 0
    free_connected = bfs_distances(free, start_free) >= 0

    targets = free_connected & (dist_m > target_wall_margin)
    n_targets = int(np.count_nonzero(targets))
    if n_targets == 0:
        raise RuntimeError('Brak kratek do obejrzenia.')

    step = max(1, int(round(candidate_step / res)))
    grid_mask = np.zeros_like(reachable)
    grid_mask[::step, ::step] = True
    cand_r, cand_c = np.nonzero(reachable & grid_mask)
    cand_r = np.append(cand_r, start_safe[0])
    cand_c = np.append(cand_c, start_safe[1])
 
    rays = precompute_rays(sensor_range / res, sensor_min_range / res, n_rays)
    vis = [visible_cells(int(r), int(c), blocked_list, rays, h, w)
           for r, c in zip(cand_r, cand_c)]

    unseen = targets.ravel().copy()
    chosen = []
    while len(chosen) < max_viewpoints:
        gains = np.array([np.count_nonzero(unseen[v]) for v in vis])
        best = int(np.argmax(gains))
        if gains[best] < min_gain_cells:
            break
        chosen.append(best)
        unseen[vis[best]] = False
        covered = n_targets - int(np.count_nonzero(unseen))
        if covered >= coverage_target * n_targets:
            break
    coverage = (n_targets - int(np.count_nonzero(unseen))) / n_targets

    order = []
    current = start_safe
    remaining = list(chosen)
    while remaining:
        d = bfs_distances(safe, current)
 
        def path_len(i):
            v = d[cand_r[i], cand_c[i]]
            return v if v >= 0 else 10 ** 9
 
        nxt = min(remaining, key=path_len)
        remaining.remove(nxt)
        order.append(nxt)
        current = (int(cand_r[nxt]), int(cand_c[nxt]))
 
    viewpoints = []
    prev = cell_to_world(start_safe[0], start_safe[1], origin_x, origin_y, res)
    for i in order:
        x, y = cell_to_world(int(cand_r[i]), int(cand_c[i]), origin_x, origin_y, res)
        yaw = math.atan2(y - prev[1], x - prev[0])
        viewpoints.append((x, y, yaw))
        prev = (x, y)
 
    seen_mask = targets & ~unseen.reshape(h, w)
    return {
        'viewpoints': viewpoints,
        'coverage': coverage,
        'n_targets': n_targets,
        'n_candidates': len(vis),
        'coarse_resolution': res,
        'free': free, 'safe': safe, 'targets': targets, 'seen': seen_mask,
    }
 
def load_map_yaml(path):
    import yaml
    with open(path) as f:
        meta = yaml.safe_load(f)
    img_path = meta['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(path)), img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    p = img.astype(np.float32) / 255.0
    occ_prob = p if int(meta.get('negate', 0)) else 1.0 - p
    grid = np.full(img.shape, -1, dtype=np.int8)
    grid[occ_prob > float(meta.get('occupied_thresh', 0.65))] = 100
    grid[occ_prob < float(meta.get('free_thresh', 0.25))] = 0
    grid = np.flipud(grid)
    origin = meta['origin']
    return grid, float(meta['resolution']), float(origin[0]), float(origin[1])
 
 
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test planera punktow na pliku mapy')
    parser.add_argument('map_yaml')
    parser.add_argument('--start', nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument('--range', type=float, default=2.5)
    parser.add_argument('--out', default='plan.png')
    args = parser.parse_args()
 
    grid, res, ox, oy = load_map_yaml(args.map_yaml)
    plan = plan_viewpoints(grid, res, ox, oy, args.start, sensor_range=args.range)
    print(f"Punktow: {len(plan['viewpoints'])}, pokrycie: {plan['coverage'] * 100:.1f}%, "
          f"celow: {plan['n_targets']}, kandydatow: {plan['n_candidates']}")
    for i, (x, y, _) in enumerate(plan['viewpoints'], 1):
        print(f'  {i:2d}: x={x:6.2f}  y={y:6.2f}')
 
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = plan['coarse_resolution']
    h, w = plan['free'].shape
    img = np.zeros((h, w, 3))
    img[plan['free']] = (1.0, 1.0, 1.0)
    img[plan['targets'] & ~plan['seen']] = (1.0, 0.6, 0.6)
    img[plan['seen']] = (0.7, 0.9, 0.7)
    extent = [ox, ox + w * r, oy, oy + h * r]
    plt.figure(figsize=(10, 7))
    plt.imshow(img, origin='lower', extent=extent)
    xs = [args.start[0]] + [v[0] for v in plan['viewpoints']]
    ys = [args.start[1]] + [v[1] for v in plan['viewpoints']]
    plt.plot(xs, ys, '-', color='tab:blue', lw=1)
    plt.plot(xs[1:], ys[1:], 'o', color='tab:blue')
    plt.plot(xs[0], ys[0], 's', color='black', label='start')
    for i, (x, y) in enumerate(zip(xs[1:], ys[1:]), 1):
        plt.annotate(str(i), (x, y), textcoords='offset points', xytext=(4, 4))
    plt.title(f"Punkty obserwacyjne ({len(plan['viewpoints'])}), "
              f"pokrycie {plan['coverage'] * 100:.1f}%  [zielone = obejrzane, czerwone = nie]")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f'Zapisano {args.out}')
 
 
if __name__ == '__main__':
    main()