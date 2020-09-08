#!/usr/bin/python

import math
from typing import Tuple, Final

HALF_NUM_LIDAR_RAYS: Final[int] = 10
NUM_LIDAR_RAYS: Final[int] = 2 * HALF_NUM_LIDAR_RAYS + 1
LIDAR_FIELD_OF_VIEW: Final[float] = math.radians(115)
WALL_RADIUS: Final[float] = 0.75
k_P: Final[float] = 50 / HALF_NUM_LIDAR_RAYS
k_D: Final[float] = 6 / HALF_NUM_LIDAR_RAYS
r: Final[float] = 0.005 * 5 + 2.5
# distance in radians between adjacent lidar rays
RAY_DIST: Final[float] = LIDAR_FIELD_OF_VIEW / HALF_NUM_LIDAR_RAYS

def atan1(x: float) -> float:
    if x < -1:
        return math.atan(x)
    elif -1 <= x <= 1:
        return math.atan(x)
    else:
        return math.atan(x)

def atan2(y: float, x: float) -> float:
    if x == 0:
        if y >= 0:
            return 0.5 * math.pi
        else:
            return -0.5 * math.pi

    atanyx = atan1(y / x)

    if x > 0:
        return atanyx
    elif y > 0:
        return math.pi + atanyx
    else:
        return -math.pi + atanyx

def acos(x: float) -> float:
    assert(abs(x) <= 1)
    return atan2(math.sqrt(1 - x ** 2), x)

def min2(a: float, b: float) -> float:
    if a <= b:
        return a
    else:
        return b

def min3(a: float, b: float, c: float) -> float:
    if a <= b and a <= c:
        return a
    elif b <= c:
        return b
    else:
        return c

def max2(a: float, b: float) -> float:
    if a >= b:
        return a
    else:
        return b

def max3(a: float, b: float, c: float) -> float:
    if a >= b and a >= c:
        return a
    elif b >= c:
        return b
    else:
        return c

def rays_in_interval(lb: float, ub: float) -> float:
    if lb >= ub:
        return 0
    else:
        return (ub - lb) / RAY_DIST #+ [-1, 1]

def err(x: float, y: float, theta: float) -> Tuple[float, int]:
    theta_L = theta + RAY_DIST
    theta_R = theta - RAY_DIST
    zeta_L = theta + LIDAR_FIELD_OF_VIEW
    zeta_R = theta - LIDAR_FIELD_OF_VIEW
    d_L = WALL_RADIUS + x
    d_R = WALL_RADIUS - x
    d_F = WALL_RADIUS - y
    d_B = WALL_RADIUS + y
    eta = atan2(-d_B, d_R)
    eta_big = eta + 2 * math.pi
    if d_L ** 2 + d_F ** 2 <= r ** 2:
        # find rays blocked by left or front wall
        alpha_L = acos(d_L / r)
        gamma_LB = math.pi + alpha_L
        gamma_LB_small = -math.pi + alpha_L
        alpha_F = acos(d_F / r)
        gamma_FR = 0.5 * math.pi - alpha_F
        frontL_lb = max2(gamma_FR, theta_L)
        frontL_ub = min2(gamma_LB, zeta_L)
        front_L = rays_in_interval(frontL_lb, frontL_ub)
        frontR_lb = max2(gamma_FR, zeta_R)
        frontR_ub = theta_R
        front_R = rays_in_interval(frontR_lb, frontR_ub)
        left_L = 0.0
        assert(zeta_R >= gamma_LB_small)
    else:
        # rays in corner are above threshold
        # need to examine left and front walls separately
        if d_L < r:
            alpha_L = acos(d_L / r)
            gamma_LB = math.pi + alpha_L
            gamma_LF = math.pi - alpha_L
            gamma_LB_small = -math.pi + alpha_L
            gamma_LF_small = -math.pi - alpha_L
            leftL_lb = max2(gamma_LF, theta_L)
            leftL_ub = min2(gamma_LB, zeta_L)
            left_L = rays_in_interval(leftL_lb, leftL_ub)
            assert(zeta_R >= gamma_LB_small)
            assert(theta_R <= gamma_LF)
        else:
            left_L = 0.0
        if d_F < r:
            alpha_F = acos(d_F / r)
            gamma_FL = 0.5 * math.pi + alpha_F
            gamma_FR = 0.5 * math.pi - alpha_F
            gamma_FL_small = -1.5 * math.pi + alpha_F
            gamma_FR_small = -1.5 * math.pi - alpha_F
            frontL_lb = max3(gamma_FR, eta, theta_L)
            frontL_ub = min2(gamma_FL, zeta_L)
            front_L = rays_in_interval(frontL_lb, frontL_ub)
            frontR_lb = max3(gamma_FR, eta, zeta_R)
            frontR_ub = min2(gamma_FL, theta_R)
            front_R = rays_in_interval(frontR_lb, frontR_ub)
            assert(zeta_R >= gamma_FL_small)
        else:
            front_L = 0.0
            front_R = 0.0
    if 0 <= d_R < r and d_B <= 0:
        # before reaching the corner
        alpha_R = acos(d_R / r)
        gamma_RF = alpha_R
        gamma_RB = -alpha_R
        gamma_RF_big = 2 * math.pi + alpha_R
        gamma_RB_big = 2 * math.pi - alpha_R
        cornerR_lb = max2(gamma_RB, zeta_R)
        cornerR_ub = min2(eta, theta_R)
        corner_R = rays_in_interval(cornerR_lb, cornerR_ub)
        assert(theta_L >= eta or theta_L >= gamma_RF)
        assert(zeta_L <= gamma_RB_big)
    elif d_R >= 0 and d_B >= 0 and d_R ** 2 + d_B ** 2 < r ** 2:
        # in the corner
        alpha_R = acos(d_R / r)
        gamma_RB = -alpha_R
        gamma_RB_big = 2 * math.pi - alpha_R
        alpha_B = acos(d_B / r)
        gamma_BR = -0.5 * math.pi + alpha_B
        cornerR_lb = max2(gamma_RB, zeta_R)
        cornerR_ub = min2(gamma_BR, theta_R)
        corner_R = rays_in_interval(cornerR_lb, cornerR_ub)
        assert(theta_L >= gamma_BR)
        assert(zeta_L <= gamma_RB_big)
    elif 0 <= d_B < r and d_R <= 0:
        # after the corner
        alpha_B = acos(d_B / r)
        gamma_BL = -0.5 * math.pi - alpha_B
        gamma_BR = -0.5 * math.pi + alpha_B
        gamma_BL_big = 1.5 * math.pi - alpha_B
        cornerR_lb = max2(eta, zeta_R)
        cornerR_ub = min2(gamma_BR, theta_R)
        corner_R = rays_in_interval(cornerR_lb, cornerR_ub)
        assert(theta_L >= gamma_BR)
        assert(zeta_L <= eta_big or zeta_L <= gamma_BL_big)
    else:
        corner_R = 0.0
    # need to reverse sign because this function computes rays less than threshold,
    # while controller computes rays greater than threshold

    # can't add intervals to numbers in Python, so we pass interval size separately
    return (front_R + corner_R - (front_L + left_L),
            int(front_R > 0) + int(corner_R > 0) + int(front_L > 0) + int(left_L > 0))

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
            error, uncertainty = err(x, y, theta)
            writer.writerow([
                x, y, theta,
                math.ceil(error - uncertainty),
                math.floor(error + uncertainty)
                ])
