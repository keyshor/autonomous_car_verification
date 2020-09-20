#!/usr/bin/python3

import math
from typing import Iterator, TextIO, List, Dict, Tuple, Final
import itertools


freshModeId: Iterator = itertools.count()
varIndex: Final[Dict[str, int]] = {v:idx for (idx, v) in enumerate([
    'x', 'y', 'V', 'theta',
    'delta', 'u',
    'err', 'prev_err',
    'theta_L', 'theta_R', 'zeta_L', 'zeta_R',
    'd_L', 'd_R', 'd_F', 'd_B',
    'gamma_LF', 'gamma_LB',
    'gamma_RF', 'gamma_RB',
    'gamma_FL', 'gamma_FR',
    'gamma_BL', 'gamma_BR',
    'eta',
    'left_L', 'front_L', 'front_R', 'corner_R',
    't', 'clock',
    'tmp1', 'tmp2'
    ])}

defaultDynamics: Final[Dict[str, str]] = {
        v: str(int(v == 't' or v == 'clock'))
        for v in varIndex
        }

def bicycle_dynamics(use_beta: bool) -> Dict[str, str]:
    d = dict(defaultDynamics)
    if use_beta:
        beta = 'atan((CAR_CENTER_OF_MASS * sin(delta)) / (CAR_LENGTH * cos(delta)))'
        d.update({
            'x': f'V * cos(theta + {beta})',
            'y': f'V * sin(theta + {beta})',
            'theta': f'(V * cos({beta}) * sin(delta)) / (CAR_LENGTH * cos(delta))'
            })
    else:
        d.update({
            'x': 'V * cos(theta)',
            'y': 'V * sin(theta)',
            'theta': '(V * sin(delta)) / (CAR_LENGTH * cos(delta))',
            })
    d['V'] = 'CAR_ACCEL_CONST * (CAR_MOTOR_CONST * (u - HYSTERESIS_CONSTANT) - V)'
    return d

class Mode:
    name: str
    int_scheme: str
    dynamics: Dict[str, str]
    invariants: List[str]
    def __init__(self, name: str, int_scheme:str = 'linear', dynamics:Dict[str,str] = defaultDynamics, invariants:List[str] = ['clock <= 0']) -> None:
        self.name = name
        self.int_scheme = int_scheme
        self.dynamics = dynamics
        self.invariants = invariants
    def output(self, f: TextIO) -> None:
        f.write(f'\t\t{self.name}\n\t\t{{\n\t\t\t{self.int_scheme} ode\n\t\t\t{{\n')
        for v,e in self.dynamics.items():
            f.write(f"\t\t\t\t{v}' = {e}\n")
        f.write('\t\t\t}\n\t\t\tinv\n\t\t\t{\n')
        for inv in self.invariants:
            f.write(f'\t\t\t\t{inv}\n')
        f.write('\t\t\t}\n')

class Transition:
    srcMode: Mode
    destMode: Mode
    guards: List[str]
    resets: Dict[str, str]
    def __init__(self, srcMode: Mode, destMode: Mode, guards:List[str] = ['clock = 0'], resets:Dict[str,str] = {}) -> None:
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
        name=f'bicycle_m{next(freshModeId)}',
        int_scheme='nonpoly',
        invariants=['clock <= TIME_STEP'],
        dynamics=bicycle_dynamics(use_beta=False)
        )

def atan1(srcVar: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    arcModeMid = Mode(name=f'arc_{varIndex[destVar]}_{varIndex[srcVar]}_atan1_m{next(freshModeId)}')
    divMode = Mode(name=f'div_{varIndex[destVar]}_{varIndex[srcVar]}_atan1_m{next(freshModeId)}')
    arcModeOut = Mode(name=f'arc_{varIndex[destVar]}_{varIndex[destVar]}_atan1_m{next(freshModeId)}')
    arcModeMid.output(f)
    divMode.output(f)
    arcModeOut.output(f)
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=arcModeMid,
        guards=['clock = 0', f'{srcVar} >= -1', f'{srcVar} <= 1']
        ))
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=divMode,
        guards=['clock = 0', f'{srcVar} <= -1']
        ))
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=divMode,
        guards=['clock = 0', f'{srcVar} >= 1']
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
        resets={destVar: f'0.5 * PI - {destVar}'}
        ))
    transitions.append(Transition(
        srcMode=arcModeOut,
        destMode=destMode,
        guards=['clock = 0', f'{destVar} <= 0'],
        resets={destVar: f'-0.5 * PI - {destVar}'}
        ))

# overwrites tmp1
def atan2(srcVarY: str, srcVarX: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    assert(srcVarY != 'tmp1' and srcVarX != 'tmp1')
    divMode = Mode(name=f'div_{varIndex["tmp1"]}_{varIndex[srcVarX]}_atan2_m{next(freshModeId)}')
    joinMode = Mode(name=f'atan2_join_m{next(freshModeId)}')
    divMode.output(f)
    joinMode.output(f)
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        guards=['clock = 0', f'{srcVarX} = 0', f'{srcVarY} >= 0'],
        resets={destVar: '0.5 * PI'}
        ))
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        guards=['clock = 0', f'{srcVarX} = 0', f'{srcVarY} <= 0'],
        resets={destVar: '-0.5 * PI'}
        ))
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=divMode,
        guards=['clock = 0', f'{srcVarX} > 0']
        ))
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=divMode,
        guards=['clock = 0', f'{srcVarX} < 0']
        ))

    atan1(srcVar='tmp1', srcMode=divMode, destVar='tmp1', destMode=joinMode, f=f)

    transitions.append(Transition(
        srcMode=joinMode,
        destMode=destMode,
        guards=['clock = 0', f'{srcVarX} >= 0'],
        resets={destVar: 'tmp1'}
        ))
    transitions.append(Transition(
        srcMode=joinMode,
        destMode=destMode,
        guards=['clock = 0', f'{srcVarX} <= 0', f'{srcVarY} >= 0'],
        resets={destVar: f'PI + tmp1'}
        ))
    transitions.append(Transition(
        srcMode=joinMode,
        destMode=destMode,
        guards=['clock = 0', f'{srcVarX} <= 0', f'{srcVarY} <= 0'],
        resets={destVar: f'-PI + tmp1'}
        ))

#computes acos(srcVar / r)
#overwrites tmp1 and tmp2
def div_r_acos(srcVar: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    assert(srcVar != 'tmp1' and srcVar != 'tmp2')
    sqrtMode = Mode(name=f'sqrt_{varIndex["tmp2"]}_{varIndex["tmp2"]}_acos_m{next(freshModeId)}')
    sqrtMode.output(f)
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=errorMode,
        guards=[f'{srcVar} < -r']
        ))
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=errorMode,
        guards=[f'{srcVar} > r']
        ))
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=sqrtMode,
        guards=[f'{srcVar} >= -r', f'{srcVar} <= r'],
        resets={'tmp2': f'r^2 - {srcVar}^2'}
        ))

    atan2(srcVarY='tmp2', srcVarX=srcVar, srcMode=sqrtMode, destVar=destVar, destMode=destMode, f=f)

def min2(srcVarA: str, srcVarB: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        guards=[f'{srcVarA} <= {srcVarB}'],
        resets={destVar: srcVarA}
        ))
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        guards=[f'{srcVarB} <= {srcVarA}'],
        resets={destVar: srcVarB}
        ))

def min3(srcVarA: str, srcVarB: str, srcVarC: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        guards=[f'{srcVarA} <= {srcVarB}', f'{srcVarA} <= {srcVarC}'],
        resets={destVar: srcVarA}
        ))
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        guards=[f'{srcVarB} <= {srcVarA}', f'{srcVarB} <= {srcVarC}'],
        resets={destVar: srcVarB}
        ))
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        guards=[f'{srcVarC} <= {srcVarA}', f'{srcVarC} <= {srcVarB}'],
        resets={destVar: srcVarC}
        ))

def max2(srcVarA: str, srcVarB: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        guards=[f'{srcVarA} >= {srcVarB}'],
        resets={destVar: srcVarA}
        ))
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        guards=[f'{srcVarB} >= {srcVarA}'],
        resets={destVar: srcVarB}
        ))

def max3(srcVarA: str, srcVarB: str, srcVarC: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        guards=[f'{srcVarA} >= {srcVarB}', f'{srcVarA} >= {srcVarC}'],
        resets={destVar: srcVarA}
        ))
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        guards=[f'{srcVarB} >= {srcVarA}', f'{srcVarB} >= {srcVarC}'],
        resets={destVar: srcVarB}
        ))
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        guards=[f'{srcVarC} >= {srcVarA}', f'{srcVarC} >= {srcVarB}'],
        resets={destVar: srcVarC}
        ))

def rays_in_interval(srcVarLb: str, srcVarUb: str, srcMode: Mode, destVar: str, destMode: Mode, f: TextIO) -> None:
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        guards=[f'{srcVarLb} >= {srcVarUb}'],
        resets={destVar: '0'}
        ))
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        guards=[f'{srcVarLb} < {srcVarUb}'],
        resets={destVar: f'RAY_DIST_INV * ({srcVarUb} - {srcVarLb}) + [-1, 1]'}
        ))

#overwrites tmp1 and tmp2
def steering(srcMode: Mode, destMode: Mode, start_guards: List[str], f: TextIO) -> None:
    pre_mode1 = Mode(name=f'err_pre1_m{next(freshModeId)}')
    pre_mode2 = Mode(name=f'err_pre2_m{next(freshModeId)}')
    pre_mode1.output(f)
    pre_mode2.output(f)
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=pre_mode1,
        guards=start_guards,
        resets={
            'theta_L': 'theta + RAY_DIST',
            'theta_R': 'theta - RAY_DIST',
            'zeta_L': 'theta + LIDAR_FIELD_OF_VIEW',
            'zeta_R': 'theta - LIDAR_FIELD_OF_VIEW',
            'd_L': 'x',
            'd_R': 'HALL_WIDTH - x',
            'd_F': '-y',
            'd_B': 'HALL_WIDTH + y',
            'tmp2': '-HALL_WIDTH - y'
            }
        ))
    atan2(srcVarY='tmp2', srcVarX='d_R', srcMode=pre_mode1, destVar='eta', destMode=pre_mode2, f=f)

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
        srcMode=pre_mode2,
        destMode=front_left_mode1,
        guards=['clock = 0', 'd_L^2 + d_F^2 <= r^2']
        ))
    div_r_acos(srcVar='d_L', srcMode=front_left_mode1, destVar='alpha_L', destMode=front_left_mode2, f=f)
    div_r_acos(srcVar='d_F', srcMode=front_left_mode2, destVar='alpha_F', destMode=front_left_mode3, f=f)
    transitions.append(Transition(
        srcMode=front_left_mode3,
        destMode=front_left_mode4,
        resets={
            'gamma_LB': 'PI + alpha_L',
            'gamma_FR': '0.5 * PI - alpha_F'
            }
        ))
    max2(srcVarA='gamma_FR', srcVarB='theta_L', srcMode=front_left_mode4, destVar='tmp1', destMode=front_left_mode5, f=f)
    min2(srcVarA='gamma_LB', srcVarB='zeta_L', srcMode=front_left_mode5, destVar='tmp2', destMode=front_left_mode6, f=f)
    rays_in_interval(srcVarLb='tmp1', srcVarUb='tmp2', srcMode=front_left_mode6, destVar='front_L', destMode=front_left_mode7, f=f)
    max2(srcVarA='gamma_FR', srcVarB='zeta_R', srcMode=front_left_mode7, destVar='tmp1', destMode=front_left_mode8, f=f)
    rays_in_interval(srcVarLb='tmp1', srcVarUb='theta_R', srcMode=front_left_mode8, destVar='front_L', destMode=front_left_mode9, f=f)
    transitions.append(Transition(
        srcMode=front_left_mode9,
        destMode=corner_start_mode,
        guards=['clock = 0', 'zeta_R >= gamma_LB - 2 * PI'],
        resets={'left_L': '0'}
        ))
    transitions.append(Transition(
        srcMode=front_left_mode9,
        destMode=errorMode,
        guards=['clock = 0', 'zeta_R < gamma_LB - 2 * PI']
        ))

    front_left_sep_mode1 = Mode(name=f'front_left_sep1_m{next(freshModeId)}')
    front_left_sep_mode2 = Mode(name=f'front_left_sep2_m{next(freshModeId)}')
    front_left_sep_mode1.output(f)
    front_left_sep_mode2.output(f)
    transitions.append(Transition(
        srcMode=pre_mode2,
        destMode=front_left_sep_mode1,
        guards=['clock = 0', 'd_L^2 + d_F^2 > r^2']
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
        guards=['clock = 0', 'd_L < r']
        ))
    transitions.append(Transition(
        srcMode=front_left_sep_mode1,
        destMode=front_left_sep_mode2,
        guards=['clock = 0', 'd_L >= r'],
        resets={'left_L': '0'}
        ))
    div_r_acos(srcVar='d_L', srcMode=left_mode1, destVar='alpha_L', destMode=left_mode2, f=f)
    transitions.append(Transition(
        srcMode=left_mode2,
        destMode=left_mode3,
        resets={
            'gamma_LB': 'PI + alpha_L',
            'gamma_LF': 'PI - alpha_L',
            }
        ))
    max2(srcVarA='gamma_LF', srcVarB='theta_L', srcMode=left_mode3, destVar='tmp1', destMode=left_mode4, f=f)
    min2(srcVarA='gamma_LB', srcVarB='zeta_L', srcMode=left_mode4, destVar='tmp1', destMode=left_mode5, f=f)
    rays_in_interval(srcVarLb='tmp1', srcVarUb='tmp2', srcMode=left_mode5, destVar='left_L', destMode=left_mode6, f=f)
    transitions.append(Transition(
        srcMode=left_mode6,
        destMode=front_left_sep_mode2,
        guards=['clock = 0', 'zeta_R >= gamma_LB - 2 * PI', 'theta_R <= gamma_LF']
        ))
    transitions.append(Transition(
        srcMode=left_mode6,
        destMode=errorMode,
        guards=['clock = 0', 'zeta_R < gamma_LB - 2 * PI']
        ))
    transitions.append(Transition(
        srcMode=left_mode6,
        destMode=errorMode,
        guards=['clock = 0', 'theta_R > gamma_LF']
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
        guards=['clock = 0', 'd_F < r']
        ))
    transitions.append(Transition(
        srcMode=front_left_sep_mode2,
        destMode=corner_start_mode,
        guards=['clock = 0', 'd_F >= r'],
        resets={
            'front_L': '0',
            'front_R': '0'
            }
        ))
    div_r_acos(srcVar='d_F', srcMode=front_mode1, destVar='alpha_F', destMode=front_mode2, f=f)
    transitions.append(Transition(
        srcMode=front_mode2,
        destMode=front_mode3,
        resets={
            'gamma_FL': '0.5 * PI + alpha_F',
            'gamma_FR': '0.5 * PI - alpha_F'
            }
        ))
    max3(srcVarA='gamma_FR', srcVarB='eta', srcVarC='theta_L', srcMode=front_mode3, destVar='tmp1', destMode=front_mode4, f=f)
    min2(srcVarA='gamma_FL', srcVarB='zeta_L', srcMode=front_mode4, destVar='tmp2', destMode=front_mode5, f=f)
    rays_in_interval(srcVarLb='tmp1', srcVarUb='tmp2', srcMode=front_mode5, destVar='front_L', destMode=front_mode6, f=f)
    max3(srcVarA='gamma_FR', srcVarB='eta', srcVarC='zeta_R', srcMode=front_mode6, destVar='tmp1', destMode=front_mode7, f=f)
    min2(srcVarA='gamma_FL', srcVarB='theta_R', srcMode=front_mode7, destVar='tmp2', destMode=front_mode8, f=f)
    rays_in_interval(srcVarLb='tmp1', srcVarUb='tmp2', srcMode=front_mode8, destVar='front_R', destMode=front_mode9, f=f)
    transitions.append(Transition(
        srcMode=front_mode9,
        destMode=corner_start_mode,
        guards=['clock = 0', 'zeta_R >= gamma_FL - 2 * PI']
        ))
    transitions.append(Transition(
        srcMode=front_mode9,
        destMode=errorMode,
        guards=['clock = 0', 'zeta_R < gamma_FL - 2 * PI']
        ))

    corner_end_mode = Mode(name=f'corner_end_m{next(freshModeId)}')
    corner_end_mode.output(f)
    transitions.append(Transition(
        srcMode=corner_start_mode,
        destMode=corner_end_mode,
        guards=['clock = 0', 'd_R >= r'],
        resets={'corner_R': '0'}
        ))
    transitions.append(Transition(
        srcMode=corner_start_mode,
        destMode=corner_end_mode,
        guards=['clock = 0', 'd_B >= r'],
        resets={'corner_R': '0'}
        ))
    transitions.append(Transition(
        srcMode=corner_start_mode,
        destMode=corner_end_mode,
        guards=['clock = 0', 'd_R >= 0', 'd_B >= 0', 'd_R^2 + d_B^2 >= r^2'],
        resets={'corner_R': '0'}
        ))

    # before reaching bend
    corner_pre_mode1 = Mode(name=f'corner_pre1_m{next(freshModeId)}')
    corner_pre_mode2 = Mode(name=f'corner_pre2_m{next(freshModeId)}')
    corner_pre_mode3 = Mode(name=f'corner_pre3_m{next(freshModeId)}')
    corner_pre_mode4 = Mode(name=f'corner_pre4_m{next(freshModeId)}')
    corner_pre_mode5 = Mode(name=f'corner_pre5_m{next(freshModeId)}')
    corner_pre_mode6 = Mode(name=f'corner_pre6_m{next(freshModeId)}')
    transitions.append(Transition(
        srcMode=corner_start_mode,
        destMode=corner_pre_mode1,
        guards=['clock = 0', 'd_R >= 0', 'd_B <= 0', 'd_R < r']
        ))
    div_r_acos(srcVar='d_R', srcMode=corner_pre_mode1, destVar='alpha_R', destMode=corner_pre_mode2, f=f)
    transitions.append(Transition(
        srcMode=corner_pre_mode2,
        destMode=corner_pre_mode3,
        resets={
            'gamma_RF': 'alpha_R',
            'gamma_RB': '-alpha_R'
            }
        ))
    max2(srcVarA='gamma_RB', srcVarB='zeta_R', srcMode=corner_pre_mode3, destVar='tmp1', destMode=corner_pre_mode4, f=f)
    min3(srcVarA='gamma_RF', srcVarB='eta', srcVarC='theta_R', srcMode=corner_pre_mode4, destVar='tmp2', destMode=corner_pre_mode5, f=f)
    rays_in_interval(srcVarLb='tmp1', srcVarUb='tmp2', srcMode=corner_pre_mode5, destVar='corner_R', destMode=corner_pre_mode6, f=f)
    transitions.append(Transition(
        srcMode=corner_pre_mode6,
        destMode=corner_end_mode,
        guards=['clock = 0', 'theta_L >= eta', 'zeta_L <= gamma_RB + 2 * PI']
        ))
    transitions.append(Transition(
        srcMode=corner_pre_mode6,
        destMode=corner_end_mode,
        guards=['clock = 0', 'theta_L >= gamma_RF', 'zeta_L <= gamma_RB + 2 * PI']
        ))
    transitions.append(Transition(
        srcMode=corner_pre_mode6,
        destMode=errorMode,
        guards=['clock = 0', 'theta_L < eta', 'theta_L < gamma_RF']
        ))
    transitions.append(Transition(
        srcMode=corner_pre_mode6,
        destMode=errorMode,
        guards=['clock = 0', 'zeta_L > gamma_RB + 2 * PI']
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
        guards=['clock = 0', 'd_R >= 0', 'd_B >= 0', 'd_R^2 + d_B^2 < r^2']
        ))
    div_r_acos(srcVar='d_R', srcMode=corner_bend_mode1, destVar='alpha_R', destMode=corner_bend_mode2, f=f)
    div_r_acos(srcVar='d_B', srcMode=corner_bend_mode2, destVar='alpha_B', destMode=corner_bend_mode3, f=f)
    transitions.append(Transition(
        srcMode=corner_bend_mode3,
        destMode=corner_bend_mode4,
        resets={
            'gamma_RB': '-alpha_R',
            'gamma_BR': '-0.5 * PI + alpha_B'
            }
        ))
    max2(srcVarA='gamma_RB', srcVarB='zeta_R', srcMode=corner_bend_mode4, destVar='tmp1', destMode=corner_bend_mode5, f=f)
    min2(srcVarA='gamma_BR', srcVarB='theta_R', srcMode=corner_bend_mode5, destVar='tmp2', destMode=corner_bend_mode6, f=f)
    rays_in_interval(srcVarLb='tmp1', srcVarUb='tmp2', srcMode=corner_bend_mode6, destVar='corner_R', destMode=corner_bend_mode7, f=f)
    transitions.append(Transition(
        srcMode=corner_bend_mode7,
        destMode=corner_end_mode,
        guards=['clock = 0', 'theta_L >= gamma_BR', 'zeta_L <= gamma_RB + 2 * PI']
        ))
    transitions.append(Transition(
        srcMode=corner_bend_mode7,
        destMode=errorMode,
        guards=['clock = 0', 'theta_L < gamma_BR']
        ))
    transitions.append(Transition(
        srcMode=corner_bend_mode7,
        destMode=errorMode,
        guards=['clock = 0', 'zeta_L > gamma_RB + 2 * PI']
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
        guards=['clock = 0', 'd_R <= 0', 'd_B >= 0', 'd_B < r']
        ))
    div_r_acos(srcVar='d_B', srcMode=corner_post_mode1, destVar='alpha_B', destMode=corner_post_mode2, f=f)
    transitions.append(Transition(
        srcMode=corner_post_mode2,
        destMode=corner_post_mode3,
        resets={
            'gamma_BL': '-0.5 * PI - alpha_B',
            'gamma_BR': '-0.5 * PI + alpha_B'
            }
        ))
    max3(srcVarA='gamma_BL', srcVarB='eta', srcVarC='zeta_R', srcMode=corner_post_mode3, destVar='tmp1', destMode=corner_post_mode4, f=f)
    min2(srcVarA='gamma_BR', srcVarB='theta_R', srcMode=corner_post_mode4, destVar='tmp2', destMode=corner_post_mode5, f=f)
    rays_in_interval(srcVarLb='tmp1', srcVarUb='tmp2', srcMode=corner_post_mode5, destVar='corner_R', destMode=corner_post_mode6, f=f)
    transitions.append(Transition(
        srcMode=corner_post_mode6,
        destMode=corner_end_mode,
        guards=['clock = 0', 'theta_L >= gamma_BR', 'zeta_L <= eta + 2 * PI']
        ))
    transitions.append(Transition(
        srcMode=corner_post_mode6,
        destMode=corner_end_mode,
        guards=['clock = 0', 'theta_L >= gamma_BR', 'zeta_L <= gamma_BL + 2 * PI']
        ))
    transitions.append(Transition(
        srcMode=corner_post_mode6,
        destMode=errorMode,
        guards=['clock = 0', 'theta_L < gamma_BR']
        ))
    transitions.append(Transition(
        srcMode=corner_post_mode6,
        destMode=errorMode,
        guards=['clock = 0', 'zeta_L > eta + 2 * PI', 'zeta_L > gamma_BL + 2 * PI']
        ))

    err_exp = 'front_R + corner_R - (front_L + left_L)'
    transitions.append(Transition(
        srcMode=corner_end_mode,
        destMode=destMode,
        guards=[
            'clock = 0',
            f'k_P * ({err_exp}) + k_D * ({err_exp} - err) > -MAX_TURNING_INPUT',
            f'k_P * ({err_exp}) + k_D * ({err_exp} - err) < MAX_TURNING_INPUT'
            ],
        resets={
            'err': 'front_R + corner_R - (front_L + left_L)',
            'prev_err': 'err',
            f'delta': 'k_P * ({err_exp}) + k_D * ({err_exp} - err)'
            }
        ))
    transitions.append(Transition(
        srcMode=corner_end_mode,
        destMode=destMode,
        guards=[
            'clock = 0',
            f'k_P * ({err_exp}) + k_D * ({err_exp} - err) <= -MAX_TURNING_INPUT'
            ],
        resets={
            'err': 'front_R + corner_R - (front_L + left_L)',
            'prev_err': 'err',
            'delta': '-MAX_TURNING_INPUT'
            }
        ))
    transitions.append(Transition(
        srcMode=corner_end_mode,
        destMode=destMode,
        guards=[
            'clock = 0',
            f'k_P * ({err_exp}) + k_D * ({err_exp} - err) >= MAX_TURNING_INPUT'
            ],
        resets={
            'err': 'front_R + corner_R - (front_L + left_L)',
            'prev_err': 'err',
            'delta': 'MAX_TURNING_INPUT'
            }
        ))

def throttle(srcMode: Mode, destMode: Mode, f: TextIO) -> None:
    transitions.append(Transition(
        srcMode=srcMode,
        destMode=destMode,
        resets={'u': '16'}
        ))

def write_modes(f: TextIO) -> None:
    steering_computed = Mode(name=f'steering_computed_m{next(freshModeId)}')
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
            'r': 0.005 * 5 + 2.5,
            # distance in radians between adjacent lidar rays
            'RAY_DIST': lidar_field_of_view / half_num_lidar_rays,

            'PI': math.pi,
            'CAR_LENGTH': .45, # in m
            'CAR_CENTER_OF_MASS': .225, # from rear of car (m)
            'CAR_DECEL_CONST': .4,
            'CAR_ACCEL_CONST': 1.633, # estimated from data
            'CAR_MOTOR_CONST': 0.2, # estimated from data
            'HYSTERESIS_CONSTANT': 4,
            'MAX_TURNING_INPUT': math.radians(15), # in radians
            'SAFE_DISTANCE': 0.3
            }

    f.write('hybrid reachablity\n')
    f.write('{\n')
    f.write('\tstate var')
    for v in varIndex:
        f.write(f' {v}')

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
            'gnuplot octagon y1, y2',
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
        'x': (0.65, 0.85),
        'y': (-6, -5.8),
        'V': (2.3, 2.5),
        'theta': (0.5 * math.pi - 0.2, 0.5 * math.pi + 0.2),
        'u': (16, 16)
        })
    write_model(args.num_lidar_rays, initial_set, args.output_file)
    args.output_file.close()
