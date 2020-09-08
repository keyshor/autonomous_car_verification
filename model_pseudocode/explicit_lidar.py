#!/usr/bin/python

import pprint
import math
import sys
from enum import Enum, auto
from typing import List, Final

HALF_NUM_LIDAR_RAYS: Final[int] = 10
NUM_LIDAR_RAYS: Final[int] = 2 * HALF_NUM_LIDAR_RAYS + 1
LIDAR_FIELD_OF_VIEW: Final[float] = math.radians(115)
WALL_RADIUS: Final[float] = 0.75
MAX_LIDAR_RANGE: Final[float] = 5.0
assert(0.5 * math.pi <= LIDAR_FIELD_OF_VIEW <= math.pi)

threshold: Final[float] = 0.005
r: Final[float] = threshold * 5 + 2.5
pid_p_coeff: Final[float] = math.radians(-5.0)
pid_d_coeff: Final[float] = math.radians(-0.6)
pid_i_coeff: Final[float] = 0.0

def compute_lidar(x: float, y: float, theta: float) -> List[float]:
    assert(-WALL_RADIUS <= x <= WALL_RADIUS and y <= WALL_RADIUS
            or x >= -WALL_RADIUS and -WALL_RADIUS <= y <= WALL_RADIUS)
    assert(-0.5 * math.pi <= theta <= math.pi)
    d_L = x + WALL_RADIUS
    d_R = WALL_RADIUS - x
    d_F = WALL_RADIUS - y
    d_B = y + WALL_RADIUS
    eta = math.atan2(-WALL_RADIUS - y, WALL_RADIUS - x)
    eta_big = eta + 2 * math.pi
    kappa = math.atan2(WALL_RADIUS - y, -WALL_RADIUS - x)
    kappa_small = kappa - 2 * math.pi
    angles = [theta + i * LIDAR_FIELD_OF_VIEW / HALF_NUM_LIDAR_RAYS
            for i in range(-HALF_NUM_LIDAR_RAYS, HALF_NUM_LIDAR_RAYS + 1)]
    assert(all([-1.5 * math.pi <= angle <= 2 * math.pi for angle in angles]))
    return [
            d_F / math.cos(angle + 1.5 * math.pi) if angle <= kappa_small
            else d_L / math.cos(angle + math.pi) if angle <= min(-0.5 * math.pi, eta)
            else d_R / math.cos(angle) if angle <= eta
            else d_B / math.cos(angle + 0.5 * math.pi) if angle <= 0
            else d_F / math.cos(angle - 0.5 * math.pi) if angle <= kappa
            else d_L / math.cos(angle - math.pi) if angle <= min(1.5 * math.pi, eta_big)
            else d_R / math.cos(angle - 2 * math.pi) if angle <= eta_big
            else d_B / math.cos(angle - 1.5 * math.pi)
            for angle in angles]

def normalize_lidar(l: List[float]) -> List[float]:
    return [(x - 2.5) / 5 for x in l]

def compute_err(l: List[float]) -> int:
    free_rays = [i for i,x in enumerate(l) if x > threshold]
    right = len([i for i in free_rays if i < HALF_NUM_LIDAR_RAYS])
    left = len([i for i in free_rays if i > HALF_NUM_LIDAR_RAYS])
    return left - right

def err(x: float, y: float, theta: float) -> int:
    return compute_err(normalize_lidar(compute_lidar(x, y, theta)))

if __name__ == '__main__':
    import sys
    import csv
    with open(sys.argv[1], newline='') as f, \
            open(sys.argv[2], mode='x', newline='') as of:
        writer = csv.writer(of)
        for row in csv.reader(f):
            x = float(row[0])
            y = float(row[1])
            theta = float(row[2])
            writer.writerow([
                x, y, theta,
                compute_err(normalize_lidar(compute_lidar(x, y, theta)))
                ])
