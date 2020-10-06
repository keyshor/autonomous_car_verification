'''
Copyright (C) 2019 Radoslav Ivanov, Taylor J Carpenter, James Weimer,
                   Rajeev Alur, George J. Pappa, Insup Lee

This file is part of Verisig.

Verisig is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

Verisig is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.  You should have received a copy of the GNU General
Public License along with Verisig.  If not, see
<https://www.gnu.org/licenses/>.

This is a python prototype of the tool Verisig, specifically written
to handle the F1/10 case study, which does not fit the SpaceEx format
that the released tool works with.

Modified for sketch based verification by Kishor Jothimurugan.

Example usage:
    python simple_model.py 21

'''

import os
import subprocess
from subprocess import PIPE
import sys
import numpy as np


MAX_TURNING_INPUT = 15  # in degrees
CONST_THROTTLE = 16  # constant throttle input for this case study
SPEED_EPSILON = 1e-8

CAR_LENGTH = .45  # in m
CAR_LENGTH_INV = 1 / CAR_LENGTH  # in m
CAR_CENTER_OF_MASS = 0.225  # from rear of car (m)
CAR_ACCEL_CONST = 1.633
CAR_MOTOR_CONST = 0.2  # 45 MPH top speed (20 m/s) at 100 throttle

LIDAR_MAX_DISTANCE = 10  # in m

HALLWAY_WIDTH = 1.5
HALLWAY_LENGTH = 10

TIME_STEP = 0.1  # in s

HYSTERESIS_CONSTANT = 4
PIBY180 = np.pi / 180.0
PIBY2 = np.pi / 2

WALL_LIMIT = 0.15
WALL_MIN = str(WALL_LIMIT)
WALL_MAX = str(HALLWAY_WIDTH - WALL_LIMIT)

DYNAMICS = {}
DYNAMICS['y1'] = 'y1\' = -y3 * sin(y4)\n'
DYNAMICS['y2'] = 'y2\' = -y3 * cos(y4)\n'
DYNAMICS['y3'] = 'y3\' = ' + str(CAR_ACCEL_CONST) +\
    ' * ' + str(CAR_MOTOR_CONST) + ' * (' + str(CONST_THROTTLE) +\
    ' - ' + str(HYSTERESIS_CONSTANT) + ') - ' + str(CAR_ACCEL_CONST) + ' * y3\n'
DYNAMICS['y4'] = 'y4\' = ' + str(CAR_LENGTH_INV) + ' * y3 * sin(u) / cos(u)\n'
DYNAMICS['k'] = 'k\' = 0\n'
DYNAMICS['u'] = 'u\' = 0\n'
DYNAMICS['clock'] = 'clock\' = 1\n'


def writeControllerModes(stream, dynamics=DYNAMICS):

    # first mode
    writeOneMode(stream, 0, dynamics)

    # Ouput mode
    writeOneMode(stream, 1, dynamics)


def writeOneMode(stream, modeIndex, dynamics=DYNAMICS, name=''):
    stream.write('\t\t' + 'm' + name + str(modeIndex) + '\n')
    stream.write('\t\t{\n')
    stream.write('\t\t\tnonpoly ode\n')
    stream.write('\t\t\t{\n')

    for sysState in dynamics:

        stream.write('\t\t\t\t' + sysState + '\' = 0\n')

    stream.write('\t\t\t\tt\' = 1\n')
    stream.write('\t\t\t}\n')

    stream.write('\t\t\tinv\n')
    stream.write('\t\t\t{\n')

    stream.write('\t\t\t\tclock <= 0\n')

    stream.write('\t\t\t}\n')
    stream.write('\t\t}\n')


def writePlantModes(stream, name='', dynamics=DYNAMICS):

    stream.write('\t\t' + 'm' + name + str(2) + '\n')
    stream.write('\t\t{\n')
    stream.write('\t\t\tnonpoly ode\n')
    stream.write('\t\t\t{\n')

    for sysState in dynamics:

        stream.write('\t\t\t\t' + dynamics[sysState])

    stream.write('\t\t\t\tt\' = 1\n')
    stream.write('\t\t\t}\n')

    stream.write('\t\t\tinv\n')
    stream.write('\t\t\t{\n')

    stream.write('\t\t\t\tclock <= ' + str(TIME_STEP) + '\n')

    stream.write('\t\t\t}\n')
    stream.write('\t\t}\n')


def writeEndMode(stream, name, dynamics=DYNAMICS):
    stream.write('\t\t' + name + '\n')
    stream.write('\t\t{\n')
    stream.write('\t\t\tnonpoly ode\n')
    stream.write('\t\t\t{\n')

    for sysState in dynamics:

        stream.write('\t\t\t\t' + sysState + '\' = 0\n')

    stream.write('\t\t\t\tt\' = 1\n')
    stream.write('\t\t\t}\n')

    stream.write('\t\t\tinv\n')
    stream.write('\t\t\t{\n')

    stream.write('\t\t\t\tclock >= 0\n')

    stream.write('\t\t\t}\n')
    stream.write('\t\t}\n')


def writeControllerJumps(stream):

    stream.write('\t\tm0 -> m1\n')
    stream.write('\t\tguard { clock = 0 }\n')
    stream.write('\t\treset { ')
    stream.write('u\' := ' + str(1) + ' ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')


def writePlantJumps(stream):
    pass


def writeController2PlantJumps(stream):

    stream.write('\t\tm1 -> m2\n')
    stream.write('\t\tguard { clock = 0 u <= ' + str(MAX_TURNING_INPUT) +
                 ' u >= ' + str(-MAX_TURNING_INPUT) + ' }\n')
    stream.write('\t\treset { ')
    stream.write('u\' := ' + str(PIBY180) + ' * u ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\tm1 -> m2\n')
    stream.write('\t\tguard { clock = 0 u >= ' + str(MAX_TURNING_INPUT) + ' }\n')
    stream.write('\t\treset { ')
    stream.write('u\' := ' + str(PIBY180 * MAX_TURNING_INPUT) + ' ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\tm1 -> m2\n')
    stream.write('\t\tguard { clock = 0 u <= ' + str(-MAX_TURNING_INPUT) + ' }\n')
    stream.write('\t\treset { ')
    stream.write('u\' := ' + str(-PIBY180 * MAX_TURNING_INPUT) + ' ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')


def writePlant2ControllerJumps(stream):

    stream.write('\t\tm2 -> m0\n')
    stream.write('\t\tguard { clock = ' + str(TIME_STEP) + ' y1 <= 10.0' + ' }\n')
    stream.write('\t\treset { ')
    stream.write('k\' := k + 1 ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')


def writeEndJump(stream):

    stream.write('\t\tm2 ->  m_end_pl\n')
    stream.write('\t\tguard { y1 = 10.0 y2 <= 0.65}\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\tm2 ->  m_end_pr\n')
    stream.write('\t\tguard { y1 = 10.0 y2 >= 0.85}\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\tm2 ->  m_end_hr\n')
    stream.write('\t\tguard { y1 = 10.0 y4 <= ' + str(-PIBY2 - 0.02) + '}\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\tm2 ->  m_end_hl\n')
    stream.write('\t\tguard { y1 = 10.0 y4 >= ' + str(-PIBY2 + 0.02) + '}\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\tm2 ->  m_end_sr\n')
    stream.write('\t\tguard { y1 = 10.0 y3 >= ' + str(2.4 + SPEED_EPSILON) + ' }\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\tm2 ->  m_end_sl\n')
    stream.write('\t\tguard { y1 = 10.0 y3 <= ' + str(2.4 - SPEED_EPSILON) + ' }\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\tm2 ->  m_top_wall\n')
    stream.write('\t\tguard { y2 <= ' + WALL_MIN + ' }\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\tm2 ->  m_left_wall\n')
    stream.write('\t\tguard { y1 <= ' + WALL_MIN + ' }\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\tm2 ->  m_right_bottom_wall\n')
    stream.write('\t\tguard { y1 >= ' + WALL_MAX + ' y2 >= ' + WALL_MAX + ' }\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')


def writeInitCond(stream, initProps, initState='m0'):

    stream.write('\tinit\n')
    stream.write('\t{\n')
    stream.write('\t\t' + initState + '\n')
    stream.write('\t\t{\n')

    for prop in initProps:
        stream.write('\t\t\t' + prop + '\n')

    stream.write('\t\t\tt in [0, 0]\n')
    stream.write('\t\t\tclock in [0, 0]\n')
    stream.write('\t\t}\n')
    stream.write('\t}\n')


def writeComposedSystem(filename, initProps, safetyProps, numSteps):
    '''
    Write Flow* model
    '''
    with open(filename, 'w') as stream:

        stream.write('hybrid reachability\n')
        stream.write('{\n')

        # encode variable names--------------------------------------------------
        stream.write('\t' + 'state var ')

        for state in DYNAMICS:
            stream.write(state + ', ')
        stream.write('t\n\n')

        # settings---------------------------------------------------------------------------------
        stream.write('\tsetting\n')
        stream.write('\t{\n')
        # F1/10 case study (HSCC)
        stream.write('\t\tadaptive steps {min 1e-6, max 0.005}\n')
        stream.write('\t\ttime ' + str(numSteps * (0.1)) + '\n')  # F1/10 case study (HSCC)
        stream.write('\t\tremainder estimation 1e-1\n')
        stream.write('\t\tidentity precondition\n')
        stream.write('\t\tgnuplot octagon y1, y2\n')
        stream.write('\t\tfixed orders 4\n')
        stream.write('\t\tcutoff 1e-12\n')
        stream.write('\t\tprecision 100\n')
        stream.write('\t\toutput {}\n'.format(os.path.basename(filename[:-6])))
        stream.write('\t\tmax jumps ' + str(4 * numSteps + 2) + '\n')  # F1/10 case study
        stream.write('\t\tprint on\n')
        stream.write('\t}\n\n')

        # encode modes-----------------------------------------------------------------------------
        stream.write('\tmodes\n')
        stream.write('\t{\n')

        writeControllerModes(stream)
        writePlantModes(stream)
        writeEndMode(stream, 'm_end_pr')
        writeEndMode(stream, 'm_end_pl')
        writeEndMode(stream, 'm_end_hr')
        writeEndMode(stream, 'm_end_hl')
        writeEndMode(stream, 'm_end_sr')
        writeEndMode(stream, 'm_end_sl')
        writeEndMode(stream, 'm_left_wall')
        writeEndMode(stream, 'm_top_wall')
        writeEndMode(stream, 'm_right_bottom_wall')

        # close modes brace
        stream.write('\t}\n')

        # encode jumps-----------------------------------------------------------------------------
        stream.write('\tjumps\n')
        stream.write('\t{\n')

        writeControllerJumps(stream)
        writeController2PlantJumps(stream)
        writePlantJumps(stream)
        writePlant2ControllerJumps(stream)
        writeEndJump(stream)

        # close jumps brace
        stream.write('\t}\n')

        # encode initial condition-----------------------------------------------------------------
        writeInitCond(stream, initProps, 'm0')  # F1/10 (HSCC)

        # close top level brace
        stream.write('}\n')

        # encode unsafe set------------------------------------------------------------------------
        stream.write(safetyProps)


def main():

    numSteps = 100

    # F1/10 Safety + Reachability
    safetyProps = 'unsafe\n{\tm_left_wall\n\t{\n\t\ty1 <= ' + WALL_MIN + '\n\n\t}\n' \
        + ('\tm_right_bottom_wall\n\t{'
           + '\n\t\ty1 >= ' + WALL_MAX + '\n\t\ty2 >= ' + WALL_MAX + '\n\n\t}\n') \
        + '\tm_top_wall\n\t{\n\t\ty2 <= ' + WALL_MIN + '\n\n\t}\n' \
        + '\tm2\n\t{\n\t\tk >= ' + str(numSteps-1) + '\n\n\t}\n' \
        + '\tm_end_pl\n\t{\n\t\ty2 <= 0.65\n\n\t}\n' \
        + '\tm_end_pr\n\t{\n\t\ty2 >= 0.85\n\n\t}\n' \
        + '\tm_end_hl\n\t{\n\t\ty4 >= ' + str(-PIBY2 + 0.02) + '\n\n\t}\n' \
        + '\tm_end_hr\n\t{\n\t\ty4 <= ' + str(-PIBY2 - 0.02) + '\n\n\t}\n' \
        + '\tm_end_sr\n\t{\n\t\ty3 >= ' + str(2.4 + SPEED_EPSILON) + '\n\n\t}\n' \
        + '\tm_end_sl\n\t{\n\t\ty3 <= ' + str(2.4 - SPEED_EPSILON) + '\n\n\t}\n}'

    modelFolder = '../flowstar_models'
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)

    modelFile = modelFolder + '/testModel'

    curLBPos = 0.65
    posOffset = 0.005

    count = 1

    initProps = ['y1 in [' + str(curLBPos) + ', ' + str(curLBPos + posOffset) + ']',
                 'y2 in [6.5, 6.5]',
                 'y3 in [' + str(2.4 - SPEED_EPSILON) + ', ' + str(2.4 + SPEED_EPSILON) + ']',
                 'y4 in [-0.005, 0.005]', 'k in [0, 0]', 'u in [0, 0]']  # F1/10

    curModelFile = modelFile + '_' + str(count) + '.model'

    writeComposedSystem(curModelFile, initProps, safetyProps, numSteps)

    args = '../flowstar_verisig/flowstar' + ' < ' + curModelFile
    _ = subprocess.Popen(args, shell=True, stdin=PIPE)


if __name__ == '__main__':
    main()
