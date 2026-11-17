import argparse
import math
import os
import re
import sys
 
PATTERN = re.compile(r'Aruco\s+(\d+):\s*x:\s*([-+0-9.eE]+)\s*y:\s*([-+0-9.eE]+)')
 
 
def load_positions(path):
    positions = {}
    with open(os.path.expanduser(path)) as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                positions[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return positions
 
 
def main():
    parser = argparse.ArgumentParser(description='Blad lokalizacji kostek')
    parser.add_argument('--gt', default='/root/.ros/arm05_sim/logs/latest.txt')
    parser.add_argument('--result', default='~/.ros/arm05_explorer/last_result.txt')
    args, _ = parser.parse_known_args(sys.argv[1:])
 
    gt = load_positions(args.gt)
    res = load_positions(args.result)
 
    print(f'{"ID":>3} | {"prawda x":>9} {"prawda y":>9} | {"robot x":>9} {"robot y":>9} | '
          f'{"blad [m]":>8}')
    print('-' * 64)
    errors = []
    for cube_id in sorted(gt):
        gx, gy = gt[cube_id]
        if cube_id in res:
            rx, ry = res[cube_id]
            err = math.hypot(rx - gx, ry - gy)
            errors.append(err)
            print(f'{cube_id:>3} | {gx:9.3f} {gy:9.3f} | {rx:9.3f} {ry:9.3f} | {err:8.3f}')
        else:
            print(f'{cube_id:>3} | {gx:9.3f} {gy:9.3f} | {"---":>9} {"---":>9} | {"BRAK":>8}')
    extra = sorted(set(res) - set(gt))
    print('-' * 64)
    print(f'Znaleziono {len(errors)}/{len(gt)} kostek')
    if errors:
        print(f'Sredni blad: {sum(errors) / len(errors):.3f} m, '
              f'maksymalny: {max(errors):.3f} m')
    if extra:
        print(f'Uwaga: ID spoza ground truth: {extra}')
 
 
if __name__ == '__main__':
    main()