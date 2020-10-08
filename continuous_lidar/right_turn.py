#!/usr/bin/python3

import math
from typing import Iterator, TextIO, List, Dict, Tuple, Final
import itertools

R: Final[float] = 0.005 * 5 + 2.5
EPSILON: Final[float] = 1e-2
SAFE_DISTANCE: Final[float] = 0.3
atan_uncertainty: Final[float] = math.atan(EPSILON / SAFE_DISTANCE)
acos_uncertainty: Final[float] = math.acos((R - EPSILON) / R)

freshModeId: Iterator = itertools.count()
varIndex: Final[Dict[str, int]] = {v:idx for (idx, v) in enumerate([
    'x_x', 'x_y', 'x_V', 'x_theta',
    'delta', 'u', 'prev_err',
    'alpha_L', 'alpha_R', 'alpha_F', 'alpha_B',
    'eta',
    'left_L', 'front_L', 'front_R', 'corner_R',
    't', 'clock',
    'tmp1', 'tmp2', 'tmp3', 'tmp4', 'tmp5', 'tmp6', 'tmpLb', 'tmpUb'
    ])}

defaultDynamics: Final[Dict[str, str]] = {
        v: str(int(v == 't' or v == 'clock'))
        for v in varIndex
        }

def bicycle_dynamics(use_beta: bool) -> Dict[str, str]:
    d = dict(defaultDynamics)
    #if use_beta:
    #    beta = 'atan((CAR_CENTER_OF_MASS * sin(delta)) / (CAR_LENGTH * cos(delta)))'
    #    d.update({
    #        'x_x': f'x_V * cos(x_theta + {beta})',
    #        'x_y': f'x_V * sin(x_theta + {beta})',
    #        'x_theta': f'(x_V * cos({beta}) * sin(delta)) / (CAR_LENGTH * cos(delta))'
    #        })
    #else:
    d.update({
        'x_x': 'x_V * cos(x_theta)',
        'x_y': 'x_V * sin(x_theta)',
        'x_theta': '(x_V * sin(delta)) / (CAR_LENGTH * cos(delta))',
        })
    d['x_V'] = 'CAR_ACCEL_CONST * (CAR_MOTOR_CONST * (u - HYSTERESIS_CONSTANT) - x_V)'
    return d

class Mode:
    name: str
    int_scheme: str
    dynamics: Dict[str, str]
    invariants: List[str]
    def __init__(self, name: str, int_scheme:str = 'nonpoly ode', dynamics:Dict[str,str] = defaultDynamics, invariants:List[str] = ['clock <= 0']) -> None:
        self.name = name
        self.int_scheme = int_scheme
        self.dynamics = dynamics
        self.invariants = invariants
    def output(self, f: TextIO) -> None:
        f.write(f'\t\t{self.name}\n')
        f.write('\t\t{\n')
        f.write(f'\t\t\t{self.int_scheme}\n\t\t\t{{\n')
        for v,e in self.dynamics.items():
            f.write(f"\t\t\t\t{v}' = {e}\n")
        f.write('\t\t\t}\n\t\t\tinv\n\t\t\t{\n')
        for inv in self.invariants:
            f.write(f'\t\t\t\t{inv}\n')
        f.write('\t\t\t}\n')
        f.write('\t\t}\n')

class Transition:
    srcMode: Mode
    destMode: Mode
    guards: List[str]
    resets: Dict[str, str]
    def __init__(self, srcMode: Mode, destMode: Mode, guards:List[str] = ['clock = 0'], resets:Dict[str,str] = {'clock': '0'}) -> None:
        self.srcMode = srcMode
        self.destMode = destMode
        self.guards = guards
        self.resets = resets
    def output(self, f: TextIO) -> None:
        f.write(f'\t\t{self.srcMode.name} -> {self.destMode.name}\n\t\tguard\n\t\t{{\n')
        for g in self.guards:
            f.write(f'\t\t\t{g}\n')
        f.write('\t\t}\n\t\treset\n\t\t{\n')
        for v,e in self.resets.items():
            f.write(f"\t\t\t{v}' := {e}\n")
        f.write('\t\t}\n\t\tinterval aggregation\n')

transitions: List[Transition] = []

errorMode: Final[Mode] = Mode(name=f'error_m{next(freshModeId)}')

bicycle_mode: Final[Mode] = Mode(
        name=f'_cont_bicycle_m{next(freshModeId)}',
        int_scheme='nonpoly ode',
        invariants=['clock <= TIME_STEP'],
        dynamics=bicycle_dynamics(use_beta=False)
        )

#overwrites tmp1, tmp2, tmp3
def atan1(srcExpr: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    src = 'tmp1'
    src_plus_one = 'tmp2'
    src_minus_one = 'tmp3'
    beginMode = Mode(name=f'atan1_begin_m{next(freshModeId)}')
    arcModeMid = Mode(name=f'_arc_{varIndex[destVar]}_{varIndex[src]}_atan1_m{next(freshModeId)}')
    divMode = Mode(name=f'_div_{varIndex[destVar]}_{varIndex[src]}_atan1_m{next(freshModeId)}')
    arcModeOut = Mode(name=f'_arc_{varIndex[destVar]}_{varIndex[destVar]}_atan1_m{next(freshModeId)}')
    beginMode.output(f)
    arcModeMid.output(f)
    divMode.output(f)
    arcModeOut.output(f)
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=beginMode,
        resets={
            src: f'{srcExpr}',
            src_plus_one: f'{srcExpr} + 1',
            src_minus_one: f'{srcExpr} - 1'
            }
        ))
    transitions.append(Transition(
        srcMode=beginMode,
        destMode=arcModeMid,
        guards=['clock = 0', f'{src_plus_one} >= 0', f'{src_minus_one} <= 0']
        ))
    transitions.append(Transition(
        srcMode=beginMode,
        destMode=divMode,
        guards=['clock = 0', f'{src_plus_one} <= 0']
        ))
    transitions.append(Transition(
        srcMode=beginMode,
        destMode=divMode,
        guards=['clock = 0', f'{src_minus_one} >= 0']
        ))
    transitions.append(Transition(
        srcMode=divMode,
        destMode=arcModeOut
        ))
    transitions.append(Transition(
        srcMode=arcModeMid,
        destMode=destMode
        ))
    transitions.append(Transition(
        srcMode=arcModeOut,
        destMode=destMode,
        guards=['clock = 0', f'{destVar} >= 0'],
        resets={
            'clock': '0',
            destVar: f'0.5 * PI - {destVar}'
            }
        ))
    transitions.append(Transition(
        srcMode=arcModeOut,
        destMode=destMode,
        guards=['clock = 0', f'{destVar} <= 0'],
        resets={
            'clock': '0',
            destVar: f'-0.5 * PI - {destVar}'
            }
        ))

# overwrites tmp1, tmp2, tmp3, tmp4, tmp5
def atan2(srcExprY: str, srcExprX: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    y_var = 'tmp4'
    x_inv = 'tmp5'
    x_plus_epsilon = 'tmp1'
    x_minus_epsilon = 'tmp2'
    beginMode = Mode(name=f'atan2_begin_m{next(freshModeId)}')
    divMode = Mode(name=f'_div_{varIndex[x_inv]}_{varIndex[x_inv]}_atan2_m{next(freshModeId)}')
    joinMode = Mode(name=f'atan2_join_m{next(freshModeId)}')
    beginMode.output(f)
    divMode.output(f)
    joinMode.output(f)
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=beginMode,
        resets={
            'clock': '0',
            y_var: srcExprY,
            x_inv: srcExprX,
            x_plus_epsilon: f'{srcExprX} + epsilon',
            x_minus_epsilon: f'{srcExprX} - epsilon'
            }
        ))
    transitions.append(Transition(
        srcMode=beginMode,
        destMode=destMode,
        guards=['clock = 0', f'{x_plus_epsilon} >= 0', f'{x_minus_epsilon} <= 0', f'{y_var} >= 0'],
        resets={
            'clock': '0',
            destVar: f'0.5 * PI + [{-atan_uncertainty}, {atan_uncertainty}]'
            }
        ))
    transitions.append(Transition(
        srcMode=beginMode,
        destMode=destMode,
        guards=['clock = 0', f'{x_plus_epsilon} >= 0', f'{x_minus_epsilon} <= 0', f'{y_var} <= 0'],
        resets={
            'clock': '0',
            destVar: f'-0.5 * PI + [{-atan_uncertainty}, {atan_uncertainty}]'
            }
        ))
    transitions.append(Transition(
        srcMode=beginMode,
        destMode=divMode,
        guards=['clock = 0', f'{x_minus_epsilon} >= 0']
        ))
    transitions.append(Transition(
        srcMode=beginMode,
        destMode=divMode,
        guards=['clock = 0', f'{x_plus_epsilon} <= 0']
        ))

    atanyx = 'tmp1'

    atan1(srcExpr=f'{y_var} * {x_inv}', srcMode=divMode, destVar=atanyx, destMode=joinMode, f=f)

    transitions.append(Transition(
        srcMode=joinMode,
        destMode=destMode,
        guards=['clock = 0', f'{x_inv} >= 0'],
        resets={
            'clock': '0',
            destVar: atanyx
            }
        ))
    transitions.append(Transition(
        srcMode=joinMode,
        destMode=destMode,
        guards=['clock = 0', f'{x_inv} <= 0', f'{y_var} >= 0'],
        resets={
            'clock': '0',
            destVar: f'PI + {atanyx}'
            }
        ))
    transitions.append(Transition(
        srcMode=joinMode,
        destMode=destMode,
        guards=['clock = 0', f'{x_inv} <= 0', f'{y_var} <= 0'],
        resets={
            'clock': '0',
            destVar: f'-PI + {atanyx}'
            }
        ))

#computes acos(srcExpr / r)
#overwrites tmp1, tmp2, tmp3, tmp4, tmp5
def div_r_acos(srcExpr: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    src = 'tmp1'
    opp = 'tmp2'
    src_plus_r = 'tmp3'
    src_minus_r = 'tmp4'
    beginMode = Mode(name=f'acos_begin_m{next(freshModeId)}')
    sqrtMode = Mode(name=f'_sqrt_{varIndex[opp]}_{varIndex[opp]}_acos_m{next(freshModeId)}')
    beginMode.output(f)
    sqrtMode.output(f)
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=beginMode,
        resets={
            'clock': '0',
            src: srcExpr,
            opp: f'r^2 - ({srcExpr})^2',
            src_plus_r: f'{srcExpr} + r - epsilon',
            src_minus_r: f'{srcExpr} - r + epsilon'
            }
        ))
    transitions.append(Transition(
        srcMode=beginMode,
        destMode=destMode,
        guards=['clock = 0', f'{src_plus_r} <= 0'],
        resets={
            'clock': '0',
            destVar: f'PI + [{-acos_uncertainty}, 0]'
            }
        ))
    transitions.append(Transition(
        srcMode=beginMode,
        destMode=destMode,
        guards=['clock = 0', f'{src_minus_r} >= 0'],
        resets={
            'clock': '0',
            destVar: f'0 + [0, {acos_uncertainty}]'
            }
        ))
    transitions.append(Transition(
        srcMode=beginMode,
        destMode=sqrtMode,
        guards=[f'{src_plus_r} >= 0', f'{src_minus_r} <= 0']
        ))

    atan2(srcExprY=opp, srcExprX=src, srcMode=sqrtMode, destVar=destVar, destMode=destMode, f=f)

# overwrites tmp1, tmp2, tmp3
def min2(srcExprA: str, srcExprB: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    a_var = 'tmp1'
    b_var = 'tmp2'
    a_minus_b = 'tmp3'
    midMode = Mode(name=f'min2_m{next(freshModeId)}')
    midMode.output(f)
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=midMode,
        resets={
            'clock': '0',
            a_var: srcExprA,
            b_var: srcExprB,
            a_minus_b: f'({srcExprA}) - ({srcExprB})'
            }
        ))
    transitions.append(Transition(
        srcMode=midMode,
        destMode=destMode,
        guards=[f'{a_minus_b} <= 0'],
        resets={
            'clock': '0',
            destVar: a_var
            }
        ))
    transitions.append(Transition(
        srcMode=midMode,
        destMode=destMode,
        guards=[f'{a_minus_b} >= 0'],
        resets={
            'clock': '0',
            destVar: b_var
            }
        ))

# overwrites tmp1, tmp2, tmp3, tmp4, tmp5, tmp6
def min3(srcExprA: str, srcExprB: str, srcExprC: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    a_var = 'tmp1'
    b_var = 'tmp2'
    c_var = 'tmp3'
    a_minus_b = 'tmp4'
    a_minus_c = 'tmp5'
    b_minus_c = 'tmp6'
    midMode = Mode(name=f'min3_m{next(freshModeId)}')
    midMode.output(f)
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=midMode,
        resets={
            'clock': '0',
            a_var: srcExprA,
            b_var: srcExprB,
            c_var: srcExprC,
            a_minus_b: f'({srcExprA}) - ({srcExprB})',
            a_minus_c: f'({srcExprA}) - ({srcExprC})',
            b_minus_c: f'({srcExprB}) - ({srcExprC})'
            }
        ))
    transitions.append(Transition(
        srcMode=midMode,
        destMode=destMode,
        guards=['clock = 0', f'{a_minus_b} <= 0', f'{a_minus_c} <= 0'],
        resets={
            'clock': '0',
            destVar: a_var
            }
        ))
    transitions.append(Transition(
        srcMode=midMode,
        destMode=destMode,
        guards=['clock = 0', f'{a_minus_b} >= 0', f'{b_minus_c} <= 0'],
        resets={
            'clock': '0',
            destVar: b_var
            }
        ))
    transitions.append(Transition(
        srcMode=midMode,
        destMode=destMode,
        guards=['clock = 0', f'{a_minus_c} >= 0', f'{b_minus_c} >= 0'],
        resets={
            'clock': '0',
            destVar: c_var
            }
        ))

# overwrites tmp1, tmp2, tmp3
def max2(srcExprA: str, srcExprB: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    a_var = 'tmp1'
    b_var = 'tmp2'
    a_minus_b = 'tmp3'
    midMode = Mode(name=f'max2_m{next(freshModeId)}')
    midMode.output(f)
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=midMode,
        resets={
            'clock': '0',
            a_var: srcExprA,
            b_var: srcExprB,
            a_minus_b: f'({srcExprA}) - ({srcExprB})'
            }
        ))
    transitions.append(Transition(
        srcMode=midMode,
        destMode=destMode,
        guards=[f'{a_minus_b} >= 0'],
        resets={
            'clock': '0',
            destVar: a_var
            }
        ))
    transitions.append(Transition(
        srcMode=midMode,
        destMode=destMode,
        guards=[f'{a_minus_b} <= 0'],
        resets={
            'clock': '0',
            destVar: b_var
            }
        ))

# overwrites tmp1, tmp2, tmp3, tmp4, tmp5, tmp6
def max3(srcExprA: str, srcExprB: str, srcExprC: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    a_var = 'tmp1'
    b_var = 'tmp2'
    c_var = 'tmp3'
    a_minus_b = 'tmp4'
    a_minus_c = 'tmp5'
    b_minus_c = 'tmp6'
    midMode = Mode(name=f'max3_m{next(freshModeId)}')
    midMode.output(f)
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=midMode,
        resets={
            'clock': '0',
            a_var: srcExprA,
            b_var: srcExprB,
            c_var: srcExprC,
            a_minus_b: f'({srcExprA}) - ({srcExprB})',
            a_minus_c: f'({srcExprA}) - ({srcExprC})',
            b_minus_c: f'({srcExprB}) - ({srcExprC})'
            }
        ))
    transitions.append(Transition(
        srcMode=midMode,
        destMode=destMode,
        guards=['clock = 0', f'{a_minus_b} >= 0', f'{a_minus_c} >= 0'],
        resets={
            'clock': '0',
            destVar: a_var
            }
        ))
    transitions.append(Transition(
        srcMode=midMode,
        destMode=destMode,
        guards=['clock = 0', f'{a_minus_b} <= 0', f'{b_minus_c} >= 0'],
        resets={
            'clock': '0',
            destVar: b_var
            }
        ))
    transitions.append(Transition(
        srcMode=midMode,
        destMode=destMode,
        guards=['clock = 0', f'{a_minus_c} <= 0', f'{b_minus_c} <= 0'],
        resets={
            'clock': '0',
            destVar: c_var
            }
        ))

# overwrites tmp1
def rays_in_interval(srcExprLb: str, srcExprUb: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    interval_length = 'tmp1'
    midMode = Mode(name=f'rays_m{next(freshModeId)}')
    midMode.output(f)
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=midMode,
        resets={
            'clock': '0',
            interval_length: f'({srcExprUb}) - ({srcExprLb})'
            }
        ))
    transitions.append(Transition(
        srcMode=midMode,
        destMode=destMode,
        guards=[f'{interval_length} <= 0'],
        resets={
            'clock': '0',
            destVar: '0'
            }
        ))
    transitions.append(Transition(
        srcMode=midMode,
        destMode=destMode,
        guards=[f'{interval_length} >= 0'],
        resets={
            'clock': '0',
            destVar: f'RAY_DIST_INV * {interval_length} + [-1, 1]'
            }
        ))

#overwrites tmp1 and tmp2
def steering(srcMode: Mode, destMode: Mode, start_guards: List[str], f: TextIO) -> None:
    theta_L = 'x_theta + RAY_DIST'
    theta_R = 'x_theta - RAY_DIST'
    zeta_L = 'x_theta + LIDAR_FIELD_OF_VIEW'
    zeta_R = 'x_theta - LIDAR_FIELD_OF_VIEW'
    d_L = 'x_x'
    d_R = 'HALL_WIDTH - x_x'
    d_F = '-x_y'
    d_B = 'HALL_WIDTH + x_y'
    pre_mode1 = Mode(name=f'steering_pre1_m{next(freshModeId)}')
    pre_mode2 = Mode(name=f'steering_pre2_m{next(freshModeId)}')
    pre_mode3 = Mode(name=f'steering_pre3_m{next(freshModeId)}')
    pre_mode1.output(f)
    pre_mode2.output(f)
    pre_mode3.output(f)
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=pre_mode1,
        guards=start_guards
        ))
    atan2(srcExprY=f'-({d_B})', srcExprX=d_R, srcMode=pre_mode1, destVar='eta', destMode=pre_mode2, f=f)
    eta_big = 'eta + 2 * PI'
    corner_dist_sq_minus_r_sq = 'tmp1'
    transitions.append(Transition(
        srcMode=pre_mode2,
        destMode=pre_mode3,
        resets={
            'clock': '0',
            corner_dist_sq_minus_r_sq: f'({d_L})^2 + ({d_F})^2 - r^2',
            }
        ))

    corner_start_mode = Mode(name=f'corner_start_m{next(freshModeId)}')
    front_left_mode1 = Mode(name=f'front_left1_m{next(freshModeId)}')
    front_left_mode2 = Mode(name=f'front_left2_m{next(freshModeId)}')
    front_left_mode3 = Mode(name=f'front_left3_m{next(freshModeId)}')
    front_left_mode4 = Mode(name=f'front_left4_m{next(freshModeId)}')
    front_left_mode5 = Mode(name=f'front_left5_m{next(freshModeId)}')
    front_left_mode6 = Mode(name=f'front_left6_m{next(freshModeId)}')
    front_left_mode7 = Mode(name=f'front_left7_m{next(freshModeId)}')
    front_left_mode8 = Mode(name=f'front_left8_m{next(freshModeId)}')
    front_left_mode9 = Mode(name=f'front_left9_m{next(freshModeId)}')
    corner_start_mode.output(f)
    front_left_mode1.output(f)
    front_left_mode2.output(f)
    front_left_mode3.output(f)
    front_left_mode4.output(f)
    front_left_mode5.output(f)
    front_left_mode6.output(f)
    front_left_mode7.output(f)
    front_left_mode8.output(f)
    front_left_mode9.output(f)
    transitions.append(Transition(
        srcMode=pre_mode3,
        destMode=front_left_mode1,
        guards=['clock = 0', f'{corner_dist_sq_minus_r_sq} <= 0']
        ))
    div_r_acos(srcExpr=d_L, srcMode=front_left_mode1, destVar='alpha_L', destMode=front_left_mode2, f=f)
    div_r_acos(srcExpr=d_F, srcMode=front_left_mode2, destVar='alpha_F', destMode=front_left_mode3, f=f)
    gamma_LB = 'PI + alpha_L'
    gamma_FR = '0.5 * PI - alpha_F'
    gamma_LB_small = '-PI + alpha_L'
    max2(srcExprA=gamma_FR, srcExprB=theta_L, srcMode=front_left_mode3, destVar='tmpLb', destMode=front_left_mode4, f=f)
    min2(srcExprA=gamma_LB, srcExprB=zeta_L, srcMode=front_left_mode4, destVar='tmpUb', destMode=front_left_mode5, f=f)
    rays_in_interval(srcExprLb='tmpLb', srcExprUb='tmpUb', srcMode=front_left_mode5, destVar='front_L', destMode=front_left_mode6, f=f)
    max2(srcExprA=gamma_FR, srcExprB=zeta_R, srcMode=front_left_mode6, destVar='tmpLb', destMode=front_left_mode7, f=f)
    rays_in_interval(srcExprLb='tmpLb', srcExprUb=theta_R, srcMode=front_left_mode7, destVar='front_R', destMode=front_left_mode8, f=f)
    zetaR_minus_gammaLBsmall = 'tmp1'
    transitions.append(Transition(
        srcMode=front_left_mode8,
        destMode=front_left_mode9,
        resets={
            'clock': '0',
            zetaR_minus_gammaLBsmall: f'({zeta_R}) - ({gamma_LB_small})'
            }
        ))
    transitions.append(Transition(
        srcMode=front_left_mode9,
        destMode=corner_start_mode,
        guards=['clock = 0', f'{zetaR_minus_gammaLBsmall} >= 0'],
        resets={
            'clock': '0',
            'left_L': '0'
            }
        ))
    transitions.append(Transition(
        srcMode=front_left_mode9,
        destMode=errorMode,
        guards=['clock = 0', f'{zetaR_minus_gammaLBsmall} <= 0']
        ))

    front_left_sep_mode1 = Mode(name=f'front_left_sep1_m{next(freshModeId)}')
    front_left_sep_mode2 = Mode(name=f'front_left_sep2_m{next(freshModeId)}')
    front_left_sep_mode1.output(f)
    front_left_sep_mode2.output(f)
    dL_minus_r = 'tmp1'
    transitions.append(Transition(
        srcMode=pre_mode3,
        destMode=front_left_sep_mode1,
        guards=['clock = 0', f'{corner_dist_sq_minus_r_sq} >= 0'],
        resets={
            'clock': '0',
            dL_minus_r: f'{d_L} - r'
            }
        ))
    left_mode1 = Mode(name=f'left1_m{next(freshModeId)}')
    left_mode2 = Mode(name=f'left2_m{next(freshModeId)}')
    left_mode3 = Mode(name=f'left3_m{next(freshModeId)}')
    left_mode4 = Mode(name=f'left4_m{next(freshModeId)}')
    left_mode5 = Mode(name=f'left5_m{next(freshModeId)}')
    left_mode6 = Mode(name=f'left6_m{next(freshModeId)}')
    left_mode1.output(f)
    left_mode2.output(f)
    left_mode3.output(f)
    left_mode4.output(f)
    left_mode5.output(f)
    left_mode6.output(f)
    transitions.append(Transition(
        srcMode=front_left_sep_mode1,
        destMode=left_mode1,
        guards=['clock = 0', f'{dL_minus_r} <= 0']
        ))
    transitions.append(Transition(
        srcMode=front_left_sep_mode1,
        destMode=front_left_sep_mode2,
        guards=['clock = 0', f'{dL_minus_r} >= 0'],
        resets={
            'clock': '0',
            'left_L': '0'
            }
        ))
    div_r_acos(srcExpr=d_L, srcMode=left_mode1, destVar='alpha_L', destMode=left_mode2, f=f)
    gamma_LB = 'PI + alpha_L'
    gamma_LF = 'PI - alpha_L'
    max2(srcExprA=gamma_LF, srcExprB=theta_L, srcMode=left_mode2, destVar='tmpLb', destMode=left_mode3, f=f)
    min2(srcExprA=gamma_LB, srcExprB=zeta_L, srcMode=left_mode3, destVar='tmpUb', destMode=left_mode4, f=f)
    rays_in_interval(srcExprLb='tmpLb', srcExprUb='tmpUb', srcMode=left_mode4, destVar='left_L', destMode=left_mode5, f=f)
    zetaR_minus_gammaLBsmall = 'tmp1'
    thetaR_minus_gammaLF = 'tmp2'
    transitions.append(Transition(
        srcMode=left_mode5,
        destMode=left_mode6,
        resets={
            zetaR_minus_gammaLBsmall: f'({zeta_R}) - ({gamma_LB_small})',
            thetaR_minus_gammaLF: f'({theta_R}) - ({gamma_LF})'
            }
        ))
    dF_minus_r = 'tmp1'
    transitions.append(Transition(
        srcMode=left_mode6,
        destMode=front_left_sep_mode2,
        guards=['clock = 0', f'{zetaR_minus_gammaLBsmall} >= 0', f'{thetaR_minus_gammaLF} <= 0'],
        resets={
            'clock': '0',
            dF_minus_r: f'{d_F} - r'
            }
        ))
    transitions.append(Transition(
        srcMode=left_mode6,
        destMode=errorMode,
        guards=['clock = 0', f'{zetaR_minus_gammaLBsmall} <= 0']
        ))
    transitions.append(Transition(
        srcMode=left_mode6,
        destMode=errorMode,
        guards=['clock = 0', f'{thetaR_minus_gammaLF} >= 0']
        ))

    front_mode1 = Mode(name=f'front1_m{next(freshModeId)}')
    front_mode2 = Mode(name=f'front2_m{next(freshModeId)}')
    front_mode3 = Mode(name=f'front3_m{next(freshModeId)}')
    front_mode4 = Mode(name=f'front4_m{next(freshModeId)}')
    front_mode5 = Mode(name=f'front5_m{next(freshModeId)}')
    front_mode6 = Mode(name=f'front6_m{next(freshModeId)}')
    front_mode7 = Mode(name=f'front7_m{next(freshModeId)}')
    front_mode8 = Mode(name=f'front8_m{next(freshModeId)}')
    front_mode9 = Mode(name=f'front9_m{next(freshModeId)}')
    front_mode1.output(f)
    front_mode2.output(f)
    front_mode3.output(f)
    front_mode4.output(f)
    front_mode5.output(f)
    front_mode6.output(f)
    front_mode7.output(f)
    front_mode8.output(f)
    front_mode9.output(f)
    transitions.append(Transition(
        srcMode=front_left_sep_mode2,
        destMode=front_mode1,
        guards=['clock = 0', f'{dF_minus_r} <= 0']
        ))
    dR_test = 'tmp1'
    dB_test = 'tmp2'
    dR_minus_r = 'tmp3'
    dB_minus_r = 'tmp4'
    corner_dist_sq_minus_r_sq = 'tmp5'
    transitions.append(Transition(
        srcMode=front_left_sep_mode2,
        destMode=corner_start_mode,
        guards=['clock = 0', f'{dF_minus_r} >= 0'],
        resets={
            'clock': '0',
            'front_L': '0',
            'front_R': '0',
            dR_test: d_R,
            dB_test: d_B,
            dR_minus_r: f'{d_R} - r',
            dB_minus_r: f'{d_B} - r',
            corner_dist_sq_minus_r_sq: f'({d_R})^2 + ({d_B})^2 - r^2'
            }
        ))
    div_r_acos(srcExpr=d_F, srcMode=front_mode1, destVar='alpha_F', destMode=front_mode2, f=f)
    gamma_FL = '0.5 * PI + alpha_F'
    gamma_FR = '0.5 * PI - alpha_F'
    gamma_FL_small = '-1.5 * PI + alpha_F'
    max3(srcExprA=gamma_FR, srcExprB='eta', srcExprC=theta_L, srcMode=front_mode2, destVar='tmpLb', destMode=front_mode3, f=f)
    min2(srcExprA=gamma_FL, srcExprB=zeta_L, srcMode=front_mode3, destVar='tmpUb', destMode=front_mode4, f=f)
    rays_in_interval(srcExprLb='tmpLb', srcExprUb='tmpUb', srcMode=front_mode4, destVar='front_L', destMode=front_mode5, f=f)
    max3(srcExprA=gamma_FR, srcExprB='eta', srcExprC=zeta_R, srcMode=front_mode5, destVar='tmpLb', destMode=front_mode6, f=f)
    min2(srcExprA=gamma_FL, srcExprB=theta_R, srcMode=front_mode6, destVar='tmpUb', destMode=front_mode7, f=f)
    rays_in_interval(srcExprLb='tmpLb', srcExprUb='tmpUb', srcMode=front_mode7, destVar='front_R', destMode=front_mode8, f=f)
    zetaR_minus_gammaFLsmall = 'tmp1'
    transitions.append(Transition(
        srcMode=front_mode8,
        destMode=front_mode9,
        resets={
            'clock': '0',
            zetaR_minus_gammaFLsmall: f'({zeta_R}) - ({gamma_FL_small})'
            }
        ))
    transitions.append(Transition(
        srcMode=front_mode9,
        destMode=corner_start_mode,
        guards=['clock = 0', f'{zetaR_minus_gammaFLsmall} >= 0']
        ))
    transitions.append(Transition(
        srcMode=front_mode9,
        destMode=errorMode,
        guards=['clock = 0', f'{zetaR_minus_gammaFLsmall} <= 0']
        ))

    corner_end_mode = Mode(name=f'corner_end_m{next(freshModeId)}')
    corner_end_mode.output(f)
    transitions.append(Transition(
        srcMode=corner_start_mode,
        destMode=corner_end_mode,
        guards=['clock = 0', f'{dR_minus_r} >= 0'],
        resets={
            'clock': '0',
            'corner_R': '0'
            }
        ))
    transitions.append(Transition(
        srcMode=corner_start_mode,
        destMode=corner_end_mode,
        guards=['clock = 0', f'{dB_minus_r} >= 0'],
        resets={
            'clock': '0',
            'corner_R': '0'
            }
        ))
    transitions.append(Transition(
        srcMode=corner_start_mode,
        destMode=corner_end_mode,
        guards=['clock = 0', f'{dR_test} >= 0', f'{dB_test} >= 0', f'{corner_dist_sq_minus_r_sq} >= 0'],
        resets={
            'clock': '0',
            'corner_R': '0'
            }
        ))

    # before reaching bend
    corner_pre_mode1 = Mode(name=f'corner_pre1_m{next(freshModeId)}')
    corner_pre_mode2 = Mode(name=f'corner_pre2_m{next(freshModeId)}')
    corner_pre_mode3 = Mode(name=f'corner_pre3_m{next(freshModeId)}')
    corner_pre_mode4 = Mode(name=f'corner_pre4_m{next(freshModeId)}')
    corner_pre_mode5 = Mode(name=f'corner_pre5_m{next(freshModeId)}')
    corner_pre_mode6 = Mode(name=f'corner_pre6_m{next(freshModeId)}')
    corner_pre_mode1.output(f)
    corner_pre_mode2.output(f)
    corner_pre_mode3.output(f)
    corner_pre_mode4.output(f)
    corner_pre_mode5.output(f)
    corner_pre_mode6.output(f)
    transitions.append(Transition(
        srcMode=corner_start_mode,
        destMode=corner_pre_mode1,
        guards=['clock = 0', f'{dR_test} >= 0', f'{dB_test} <= 0', f'{dR_minus_r} <= 0']
        ))
    div_r_acos(srcExpr=d_R, srcMode=corner_pre_mode1, destVar='alpha_R', destMode=corner_pre_mode2, f=f)
    gamma_RF = 'alpha_R'
    gamma_RB = '-alpha_R'
    gamma_RB_big = '2 * PI - alpha_R'
    max2(srcExprA=gamma_RB, srcExprB=zeta_R, srcMode=corner_pre_mode2, destVar='tmpLb', destMode=corner_pre_mode3, f=f)
    min3(srcExprA=gamma_RF, srcExprB='eta', srcExprC=theta_R, srcMode=corner_pre_mode3, destVar='tmpUb', destMode=corner_pre_mode4, f=f)
    rays_in_interval(srcExprLb='tmpLb', srcExprUb='tmpUb', srcMode=corner_pre_mode4, destVar='corner_R', destMode=corner_pre_mode5, f=f)
    thetaL_minus_eta = 'tmp1'
    thetaL_minus_gammaRF = 'tmp2'
    zetaL_minus_gammaRBbig = 'tmp3'
    transitions.append(Transition(
        srcMode=corner_pre_mode5,
        destMode=corner_pre_mode6,
        resets={
            'clock': '0',
            thetaL_minus_eta: f'{theta_L} - eta',
            thetaL_minus_gammaRF: f'({theta_L}) - ({gamma_RF})',
            zetaL_minus_gammaRBbig: f'({zeta_L}) - ({gamma_RB_big})'
            }
        ))
    transitions.append(Transition(
        srcMode=corner_pre_mode6,
        destMode=corner_end_mode,
        guards=['clock = 0', f'{thetaL_minus_eta} >= 0', f'{zetaL_minus_gammaRBbig} <= 0']
        ))
    transitions.append(Transition(
        srcMode=corner_pre_mode6,
        destMode=corner_end_mode,
        guards=['clock = 0', f'{thetaL_minus_gammaRF} >= 0', f'{zetaL_minus_gammaRBbig} <= 0']
        ))
    transitions.append(Transition(
        srcMode=corner_pre_mode6,
        destMode=errorMode,
        guards=['clock = 0', f'{thetaL_minus_eta} <= 0', f'{thetaL_minus_gammaRF} <= 0']
        ))
    transitions.append(Transition(
        srcMode=corner_pre_mode6,
        destMode=errorMode,
        guards=['clock = 0', f'{zetaL_minus_gammaRBbig} >= 0']
        ))

    # during bend
    corner_bend_mode1 = Mode(name=f'corner_bend1_m{next(freshModeId)}')
    corner_bend_mode2 = Mode(name=f'corner_bend2_m{next(freshModeId)}')
    corner_bend_mode3 = Mode(name=f'corner_bend3_m{next(freshModeId)}')
    corner_bend_mode4 = Mode(name=f'corner_bend4_m{next(freshModeId)}')
    corner_bend_mode5 = Mode(name=f'corner_bend5_m{next(freshModeId)}')
    corner_bend_mode6 = Mode(name=f'corner_bend6_m{next(freshModeId)}')
    corner_bend_mode7 = Mode(name=f'corner_bend7_m{next(freshModeId)}')
    corner_bend_mode1.output(f)
    corner_bend_mode2.output(f)
    corner_bend_mode3.output(f)
    corner_bend_mode4.output(f)
    corner_bend_mode5.output(f)
    corner_bend_mode6.output(f)
    corner_bend_mode7.output(f)
    transitions.append(Transition(
        srcMode=corner_start_mode,
        destMode=corner_bend_mode1,
        guards=['clock = 0', f'{dR_test} >= 0', f'{dB_test} >= 0', f'{corner_dist_sq_minus_r_sq} <= 0']
        ))
    div_r_acos(srcExpr=d_R, srcMode=corner_bend_mode1, destVar='alpha_R', destMode=corner_bend_mode2, f=f)
    div_r_acos(srcExpr=d_B, srcMode=corner_bend_mode2, destVar='alpha_B', destMode=corner_bend_mode3, f=f)
    gamma_RB = '-alpha_R'
    gamma_BR = '-0.5 * PI + alpha_B'
    max2(srcExprA=gamma_RB, srcExprB=zeta_R, srcMode=corner_bend_mode3, destVar='tmpLb', destMode=corner_bend_mode4, f=f)
    min2(srcExprA=gamma_BR, srcExprB=theta_R, srcMode=corner_bend_mode4, destVar='tmpUb', destMode=corner_bend_mode5, f=f)
    rays_in_interval(srcExprLb='tmpLb', srcExprUb='tmpUb', srcMode=corner_bend_mode5, destVar='corner_R', destMode=corner_bend_mode6, f=f)
    thetaL_minus_gammaBR = 'tmp1'
    zetaL_minus_gammaRBbig = 'tmp2'
    transitions.append(Transition(
        srcMode=corner_bend_mode6,
        destMode=corner_bend_mode7,
        resets={
            thetaL_minus_gammaBR: f'({theta_L}) - ({gamma_BR})',
            zetaL_minus_gammaRBbig: f'({zeta_L}) - ({gamma_RB_big})'
            }
        ))
    transitions.append(Transition(
        srcMode=corner_bend_mode7,
        destMode=corner_end_mode,
        guards=['clock = 0', f'{thetaL_minus_gammaBR} >= 0', f'{zetaL_minus_gammaRBbig} <= 0']
        ))
    transitions.append(Transition(
        srcMode=corner_bend_mode7,
        destMode=errorMode,
        guards=['clock = 0', f'{thetaL_minus_gammaBR} <= 0']
        ))
    transitions.append(Transition(
        srcMode=corner_bend_mode7,
        destMode=errorMode,
        guards=['clock = 0', f'{zetaL_minus_gammaRBbig} >= 0']
        ))

    # after bend
    corner_post_mode1 = Mode(name=f'corner_post1_m{next(freshModeId)}')
    corner_post_mode2 = Mode(name=f'corner_post2_m{next(freshModeId)}')
    corner_post_mode3 = Mode(name=f'corner_post3_m{next(freshModeId)}')
    corner_post_mode4 = Mode(name=f'corner_post4_m{next(freshModeId)}')
    corner_post_mode5 = Mode(name=f'corner_post5_m{next(freshModeId)}')
    corner_post_mode6 = Mode(name=f'corner_post6_m{next(freshModeId)}')
    corner_post_mode1.output(f)
    corner_post_mode2.output(f)
    corner_post_mode3.output(f)
    corner_post_mode4.output(f)
    corner_post_mode5.output(f)
    corner_post_mode6.output(f)
    transitions.append(Transition(
        srcMode=corner_start_mode,
        destMode=corner_post_mode1,
        guards=['clock = 0', f'{dR_test} <= 0', f'{dB_test} >= 0', f'{dB_minus_r} <= 0']
        ))
    div_r_acos(srcExpr=d_B, srcMode=corner_post_mode1, destVar='alpha_B', destMode=corner_post_mode2, f=f)
    gamma_BL = '-0.5 * PI - alpha_B'
    gamma_BR = '-0.5 * PI + alpha_B'
    gamma_BL_big = '1.5 * PI - alpha_B'
    max3(srcExprA=gamma_BL, srcExprB='eta', srcExprC=zeta_R, srcMode=corner_post_mode2, destVar='tmpLb', destMode=corner_post_mode3, f=f)
    min2(srcExprA=gamma_BR, srcExprB=theta_R, srcMode=corner_post_mode3, destVar='tmpUb', destMode=corner_post_mode4, f=f)
    rays_in_interval(srcExprLb='tmpLb', srcExprUb='tmpUb', srcMode=corner_post_mode4, destVar='corner_R', destMode=corner_post_mode5, f=f)
    thetaL_minus_gammaBR = 'tmp1'
    zetaL_minus_gammaBLbig = 'tmp2'
    zetaL_minus_etabig = 'tmp3'
    transitions.append(Transition(
        srcMode=corner_post_mode5,
        destMode=corner_post_mode6,
        resets={
            'clock': '0',
            thetaL_minus_gammaBR: f'({theta_L}) - ({gamma_BR})',
            zetaL_minus_gammaBLbig: f'({zeta_L}) - ({gamma_BL_big})',
            zetaL_minus_etabig: f'({zeta_L}) - ({eta_big})'
            }
        ))
    transitions.append(Transition(
        srcMode=corner_post_mode6,
        destMode=corner_end_mode,
        guards=['clock = 0', f'{thetaL_minus_gammaBR} >= 0', f'{zetaL_minus_etabig} <= 0']
        ))
    transitions.append(Transition(
        srcMode=corner_post_mode6,
        destMode=corner_end_mode,
        guards=['clock = 0', f'{thetaL_minus_gammaBR} >= 0', f'{zetaL_minus_gammaBLbig} <= 0']
        ))
    transitions.append(Transition(
        srcMode=corner_post_mode6,
        destMode=errorMode,
        guards=['clock = 0', f'{thetaL_minus_gammaBR} <= 0']
        ))
    transitions.append(Transition(
        srcMode=corner_post_mode6,
        destMode=errorMode,
        guards=['clock = 0', f'{zetaL_minus_etabig} >= 0']
        ))

    steering_end_mode = Mode(name=f'steering_end_m{next(freshModeId)}')
    steering_end_mode.output(f)
    err_sig = 'front_R + corner_R - (front_L + left_L)'
    delta_middle = f'k_P * ({err_sig}) + k_D * ({err_sig} - prev_err)'
    delta_plus_max = 'tmp1'
    delta_minus_max = 'tmp2'
    transitions.append(Transition(
        srcMode=corner_end_mode,
        destMode=steering_end_mode,
        resets={
            delta_plus_max: f'{delta_middle} + MAX_TURNING_INPUT',
            delta_minus_max: f'{delta_middle} - MAX_TURNING_INPUT',
            }
        ))
    transitions.append(Transition(
        srcMode=steering_end_mode,
        destMode=destMode,
        guards=['clock = 0', f'{delta_plus_max} >= 0', f'{delta_minus_max} <= 0'],
        resets={
            'clock': '0',
            'prev_err': err_sig,
            'delta': delta_middle
            }
        ))
    transitions.append(Transition(
        srcMode=corner_end_mode,
        destMode=destMode,
        guards=['clock = 0', f'{delta_plus_max} <= 0'],
        resets={
            'clock': '0',
            'prev_err': err_sig,
            'delta': '-MAX_TURNING_INPUT'
            }
        ))
    transitions.append(Transition(
        srcMode=corner_end_mode,
        destMode=destMode,
        guards=['clock = 0', f'{delta_minus_max} >= 0'],
        resets={
            'clock': '0',
            'prev_err': err_sig,
            'delta': 'MAX_TURNING_INPUT'
            }
        ))

def throttle(srcMode: Mode, destMode: Mode, f: TextIO) -> None:
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        resets={
            'clock': '0',
            'u': '16'
            }
        ))

def write_modes(f: TextIO) -> None:
    steering_computed = Mode(name=f'steering_computed_m{next(freshModeId)}')
    errorMode.output(f)
    bicycle_mode.output(f)
    steering_computed.output(f)

    steering(srcMode=bicycle_mode, destMode=steering_computed, start_guards=['clock = TIME_STEP'], f=f)
    throttle(srcMode=steering_computed, destMode=bicycle_mode, f=f)

def write_model(num_lidar_rays: int, initial_set: Dict[str, Tuple[float, float]], f: TextIO) -> None:
    half_num_lidar_rays = num_lidar_rays // 2
    lidar_field_of_view = math.radians(115)
    parameters = {
            'NUM_LIDAR_RAYS': num_lidar_rays,
            'HALF_NUM_LIDAR_RAYS': half_num_lidar_rays,
            'LIDAR_FIELD_OF_VIEW': lidar_field_of_view,
            'HALL_WIDTH': 1.5,
            'k_P': 50 / half_num_lidar_rays,
            'k_D': 6 / half_num_lidar_rays,
            'r': R,
            # distance in radians between adjacent lidar rays
            'RAY_DIST': lidar_field_of_view / half_num_lidar_rays,
            'RAY_DIST_INV': half_num_lidar_rays / lidar_field_of_view,
            'TIME_STEP': 0.1, # in seconds

            'PI': math.pi,
            'CAR_LENGTH': .45, # in m
            'CAR_CENTER_OF_MASS': .225, # from rear of car (m)
            'CAR_DECEL_CONST': .4,
            'CAR_ACCEL_CONST': 1.633, # estimated from data
            'CAR_MOTOR_CONST': 0.2, # estimated from data
            'HYSTERESIS_CONSTANT': 4,
            'MAX_TURNING_INPUT': math.radians(15), # in radians
            'SAFE_DISTANCE': SAFE_DISTANCE,
            'epsilon': EPSILON
            }

    f.write('hybrid reachability\n')
    f.write('{\n')
    f.write(f'\tstate var {", ".join(varIndex)}\n')

    f.write('\tpar\n')
    f.write('\t{\n')
    for p,c in parameters.items():
        f.write(f'\t\t{p} = {c}\n')
    f.write('\t}\n')

    f.write('\tsetting\n')
    f.write('\t{\n')
    for s in [
            'adaptive steps {min 1e-6, max 0.005}',
            'time 10.0',
            'remainder estimation 1e-1',
            'identity precondition',
            'gnuplot octagon x_x, x_y',
            'fixed orders 4',
            'cutoff 1e-12',
            'precision 100',
            'output testModel_1',
            'max jumps 13900',
            'print on'
            ]:
        f.write(f'\t\t{s}\n')
    f.write('\t}\n')

    f.write('\tmodes\n')
    f.write('\t{\n')
    write_modes(f)
    f.write('\t}\n')

    f.write('\tjumps\n')
    f.write('\t{\n')
    for t in transitions:
        t.output(f)
    f.write('\t}\n')

    f.write('\tinit\n')
    f.write('\t{\n')
    f.write(f'\t\t{bicycle_mode.name}\n')
    f.write('\t\t{\n')
    for v,(lb,ub) in initial_set.items():
        f.write(f'\t\t\t{v} in [{lb}, {ub}]\n')
    f.write('\t\t}\n')
    f.write('\t}\n')
    f.write('}\n')

    f.write('unsafe\n')
    f.write('{\n')
    f.write(f'\t{errorMode.name}\n')
    f.write('\t{\n')
    f.write('\t}\n')
    f.write('}\n')

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-lidar-rays', default=21, type=int)
    parser.add_argument('-o', '--output-file', default=sys.stdout, type=argparse.FileType('x'))
    args = parser.parse_args()
    initial_set = {v:(0.0,0.0) for v in varIndex}
    initial_set.update({
        'x_x': (0.745, 0.755),
        'x_y': (-5.905, -5.895),
        'x_V': (2.4, 2.4),
        'x_theta': (0.5 * math.pi - 0.000, 0.5 * math.pi + 0.000),
        'u': (16, 16)
        })
    write_model(args.num_lidar_rays, initial_set, args.output_file)
    args.output_file.close()
