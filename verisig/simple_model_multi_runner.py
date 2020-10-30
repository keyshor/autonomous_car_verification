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

import os, sys
import subprocess
from subprocess import PIPE
import numpy as np
import yaml

MAX_TURNING_INPUT = 20  # in degrees
CONST_THROTTLE = 16  # constant throttle input for this case study
SPEED_EPSILON = 1e-8

CAR_LENGTH = .45  # in m
CAR_LENGTH_INV = 1 / CAR_LENGTH  # in m
CAR_CENTER_OF_MASS = 0.225  # from rear of car (m)
CAR_ACCEL_CONST = 1.633
CAR_MOTOR_CONST = 0.2  # 45 MPH top speed (20 m/s) at 100 throttle

EXIT_DISTANCE = 10

TIME_STEP = 0.1  # in s

HYSTERESIS_CONSTANT = 4
PIBY180 = np.pi / 180.0
PIBY2 = np.pi / 2

HALLWAY_WIDTH = 1.5
HALLWAY_LENGTH = 20

POS_LB = 0.3
POS_UB = 1.2
HEADING_LB = -0.02
HEADING_UB = 0.02

WALL_LIMIT = 0.15
WALL_MIN = str(WALL_LIMIT)
WALL_MAX = str(HALLWAY_WIDTH - WALL_LIMIT)

TURN_ANGLE = -np.pi/2
#TURN_ANGLE = -2 * np.pi / 3

CORNER_ANGLE = np.pi - np.abs(TURN_ANGLE)
SIN_CORNER = np.sin(CORNER_ANGLE)
COS_CORNER = np.cos(CORNER_ANGLE)

# just a check to avoid numerical error
if TURN_ANGLE == -np.pi/2:
    SIN_CORNER = 1
    COS_CORNER = 0

NORMAL_TO_TOP_WALL = [SIN_CORNER, -COS_CORNER]

DYNAMICS = {}
DYNAMICS['y1'] = 'y1\' = -y3 * sin(y4)\n'
DYNAMICS['y2'] = 'y2\' = -y3 * cos(y4)\n'
DYNAMICS['y3'] = 'y3\' = ' + str(CAR_ACCEL_CONST) +\
    ' * ' + str(CAR_MOTOR_CONST) + ' * (' + str(CONST_THROTTLE) +\
    ' - ' + str(HYSTERESIS_CONSTANT) + ') - ' + str(CAR_ACCEL_CONST) + ' * y3\n'
DYNAMICS['y4'] = 'y4\' = ' + str(CAR_LENGTH_INV) + ' * y3 * sin(u) / cos(u)\n'
DYNAMICS['k'] = 'k\' = 0\n'
DYNAMICS['u'] = 'u\' = 0\n'
DYNAMICS['ax'] = 'ax\' = 0\n'
DYNAMICS['clock'] = 'clock\' = 1\n'


def getCornerDist(next_heading=np.pi/2 + TURN_ANGLE, reverse_cur_heading=-np.pi/2,\
                  hallLength=HALLWAY_LENGTH, hallWidth=HALLWAY_WIDTH, turnAngle=TURN_ANGLE):

    outer_x = -hallWidth/2.0
    outer_y = hallLength/2.0
    
    out_wall_proj_length = np.abs(hallWidth / np.sin(turnAngle))
    proj_point_x = outer_x + np.cos(next_heading) * out_wall_proj_length
    proj_point_y = outer_y + np.sin(next_heading) * out_wall_proj_length
    
    in_wall_proj_length = np.abs(hallWidth / np.sin(turnAngle))
    inner_x = proj_point_x + np.cos(reverse_cur_heading) * in_wall_proj_length
    inner_y = proj_point_y + np.sin(reverse_cur_heading) * in_wall_proj_length

    corner_dist = np.sqrt((outer_x - inner_x) ** 2 + (outer_y - inner_y) ** 2)
    wall_dist = np.sqrt(corner_dist ** 2 - hallWidth ** 2)

    return wall_dist


def writeControllerModes(stream, numDnnInputs, dynamics=DYNAMICS):

    # first mode
    writeOneMode(stream, 0, numDnnInputs, dynamics=dynamics)

    # Ouput mode
    writeOneMode(stream, 1, numDnnInputs, dynamics=dynamics, name='DNN1')


def writeOneMode(stream, modeIndex, numDnnInputs, dynamics=DYNAMICS, name=''):
    stream.write('\t\t' + name + 'm' + str(modeIndex) + '\n')
    stream.write('\t\t{\n')
    stream.write('\t\t\tnonpoly ode\n')
    stream.write('\t\t\t{\n')

    for sysState in dynamics:

        if sysState != 'clock':
            stream.write('\t\t\t\t' + sysState + '\' = 0\n')


    for neurState in range(numDnnInputs):
        stream.write('\t\t\t\t_f' + str(neurState+1) + '\' = 0\n')

    stream.write('\t\t\t\tt\' = 1\n')
    stream.write('\t\t\t\tclock\' = 1\n')
    stream.write('\t\t\t}\n')

    stream.write('\t\t\tinv\n')
    stream.write('\t\t\t{\n')

    stream.write('\t\t\t\tclock <= 0\n')

    stream.write('\t\t\t}\n')
    stream.write('\t\t}\n')


def writePlantModes(stream, numDnnInputs, name='', dynamics=DYNAMICS):

    # init mode
    writeOneMode(stream, 1, numDnnInputs, dynamics=dynamics, name='init_mode')  

    # dynamics mode
    stream.write('\t\t' + name + 'm2\n')
    stream.write('\t\t{\n')
    stream.write('\t\t\tnonpoly ode\n')
    stream.write('\t\t\t{\n')

    for sysState in dynamics:
        if sysState != 'clock':
            stream.write('\t\t\t\t' + dynamics[sysState])

    for neurState in range(numDnnInputs):
        stream.write('\t\t\t\t_f' + str(neurState+1) + '\' = 0\n')

    stream.write('\t\t\t\tt\' = 1\n')
    stream.write('\t\t\t\tclock\' = 1\n')
    stream.write('\t\t\t}\n')

    stream.write('\t\t\tinv\n')
    stream.write('\t\t\t{\n')

    stream.write('\t\t\t\tclock <= ' + str(TIME_STEP) + '\n')

    stream.write('\t\t\t}\n')
    stream.write('\t\t}\n')

    # hallway switch mode
    writeOneMode(stream, 3, numDnnInputs, dynamics=dynamics, name='switch')  


def writeEndMode(stream, name, numDnnInputs, dynamics=DYNAMICS):
    stream.write('\t\t' + name + '\n')
    stream.write('\t\t{\n')
    stream.write('\t\t\tnonpoly ode\n')
    stream.write('\t\t\t{\n')

    for sysState in dynamics:
        if sysState != 'clock':
            stream.write('\t\t\t\t' + sysState + '\' = 0\n')

    for neurState in range(numDnnInputs):
        stream.write('\t\t\t\t_f' + str(neurState+1) + '\' = 0\n')            

    stream.write('\t\t\t\tt\' = 1\n')
    stream.write('\t\t\t\tclock\' = 1\n')
    stream.write('\t\t\t}\n')

    stream.write('\t\t\tinv\n')
    stream.write('\t\t\t{\n')

    stream.write('\t\t\t\tclock >= 0\n')

    stream.write('\t\t\t}\n')
    stream.write('\t\t}\n')


def writeControllerJumps(stream):

    stream.write('\t\tm0 -> DNN1m1\n')
    stream.write('\t\tguard { clock = 0 }\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')


def writePlantJumps(stream):

    # check for new hallway
    stream.write('\t\t_cont_m2 -> switchm3\n')
    stream.write('\t\tguard { clock = ' + str(TIME_STEP) + ' y1 <= ' + str(HALLWAY_WIDTH) + ' ax = 0 }\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\t_cont_m2 -> switchm3\n')
    stream.write('\t\tguard { clock = ' + str(TIME_STEP) + ' y1 <= ' + str(HALLWAY_WIDTH) +\
                 ' ax = 1 y2 >= ' + str(EXIT_DISTANCE) + ' }\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')    

    stream.write('\t\t_cont_m2 -> switchm3\n')
    stream.write('\t\tguard { clock = ' + str(TIME_STEP) + ' y1 >= ' + str(HALLWAY_WIDTH) + ' ax = 0 }\n')
    stream.write('\t\treset { ')
    stream.write('y1\' := ' + str(SIN_CORNER) + ' * y2 - ' + str(COS_CORNER) + ' * y1 ')
    stream.write('y2\' := ' + str(HALLWAY_LENGTH) + ' - ' + str(COS_CORNER) + ' * y2 - ' + str(SIN_CORNER) + ' * y1 ')
    stream.write('y4\' := y4 + ' + str(-TURN_ANGLE) + ' ')
    stream.write('ax\' := 1 ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

def writeController2PlantJumps(stream):

    stream.write('\t\tDNN1m1 -> _cont_m2\n')
    stream.write('\t\tguard { clock = 0 }\n')
    stream.write('\t\treset { ')
    stream.write('u\' := ' + str(PIBY180 * MAX_TURNING_INPUT) + ' * _f1 ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

def writePlant2ControllerJumps(stream):

    # initial transition
    stream.write('\t\tinit_modem1 -> m0\n')
    stream.write('\t\tguard { clock = 0 }\n')
    stream.write('\t\treset { ')
    stream.write('_f1\' := y1 ')
    stream.write('_f2\' := ' + str(HALLWAY_WIDTH) + ' - y1 ')
    stream.write('_f3\' := 5 ')
    stream.write('_f4\' := 5 ')
    stream.write('_f5\' := y4 ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    # standard transition
    wall_dist = getCornerDist()
    
    # different cases depending on value of y2
    stream.write('\t\tswitchm3 -> m0\n')
    stream.write('\t\tguard { clock = 0 y2 <= 5 }\n')
    stream.write('\t\treset { ')
    stream.write('k\' := k + 1 ')
    stream.write('_f1\' := y1 ')
    stream.write('_f2\' := ' + str(HALLWAY_WIDTH) + ' - y1 ')
    stream.write('_f3\' := y2 ')
    stream.write('_f4\' := y2 - ' + str(wall_dist))
    stream.write('_f5\' := y4 ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\tswitchm3 -> m0\n')
    stream.write('\t\tguard { clock = 0 y2 >= 5 y2 <= ' + str(5 + wall_dist) + ' }\n')
    stream.write('\t\treset { ')
    stream.write('k\' := k + 1 ')
    stream.write('_f1\' := y1 ')
    stream.write('_f2\' := ' + str(HALLWAY_WIDTH) + ' - y1 ')
    stream.write('_f3\' := 5 ')
    stream.write('_f4\' := y2 - ' + str(wall_dist))
    stream.write('_f5\' := y4 ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\tswitchm3 -> m0\n')
    stream.write('\t\tguard { clock = 0 y2 >= ' + str(5 + wall_dist) + ' }\n')
    stream.write('\t\treset { ')
    stream.write('k\' := k + 1 ')
    stream.write('_f1\' := y1 ')
    stream.write('_f2\' := ' + str(HALLWAY_WIDTH) + ' - y1 ')
    stream.write('_f3\' := 5 ')
    stream.write('_f4\' := 5 ')
    stream.write('_f5\' := y4 ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')        


def writeEndJump(stream):

    stream.write('\t\t_cont_m2 ->  m_end_pl\n')
    stream.write('\t\tguard { y2 = ' + str(EXIT_DISTANCE) + ' y1 <= ' + str(POS_LB) + ' ax = 1}\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\t_cont_m2 ->  m_end_pr\n')
    stream.write('\t\tguard { y2 = ' + str(EXIT_DISTANCE) + ' y1 >= ' + str(POS_UB) + ' ax = 1}\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\t_cont_m2 ->  m_end_hr\n')
    stream.write('\t\tguard { y2 = ' + str(EXIT_DISTANCE) + ' y4 <= ' + str(HEADING_LB) + ' ax = 1 }\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\t_cont_m2 ->  m_end_hl\n')
    stream.write('\t\tguard { y2 = ' + str(EXIT_DISTANCE) + ' y4 >= ' + str(HEADING_UB) + ' ax = 1 }\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\t_cont_m2 ->  m_end_sr\n')
    stream.write('\t\tguard { y2 = ' + str(EXIT_DISTANCE) + ' y3 >= ' + str(2.4 + SPEED_EPSILON) + ' ax = 1 }\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\t_cont_m2 ->  m_end_sl\n')
    stream.write('\t\tguard { y2 = ' + str(EXIT_DISTANCE) + ' y3 <= ' + str(2.4 - SPEED_EPSILON) + ' ax = 1 }\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\t_cont_m2 ->  m_top_wall\n')
    stream.write('\t\tguard { ' + str(NORMAL_TO_TOP_WALL[0]) + ' * y2 + ' + str(NORMAL_TO_TOP_WALL[1]) + ' * y1  <= ' + WALL_MIN + ' }\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    stream.write('\t\t_cont_m2 ->  m_left_wall\n')
    stream.write('\t\tguard { y1 <= ' + WALL_MIN + ' }\n')
    stream.write('\t\treset { ')
    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')

    wall_dist = getCornerDist()
    stream.write('\t\t_cont_m2 ->  m_right_bottom_wall\n')
    stream.write('\t\tguard { y1 >= ' + WALL_MAX + ' ' + str(NORMAL_TO_TOP_WALL[0])\
                 + ' * y2 + ' + str(NORMAL_TO_TOP_WALL[1]) + ' * y1  >= ' + WALL_MAX + ' }\n')
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


def writeComposedSystem(filename, initProps, dnn, safetyProps, numSteps):
    '''
    Write Flow* model
    '''
    with open(filename, 'w') as stream:

        numDnnInputs = len(dnn['weights'][1][0])

        stream.write('hybrid reachability\n')
        stream.write('{\n')

        # encode variable names--------------------------------------------------
        stream.write('\t' + 'state var ')

        for state in DYNAMICS:
            if 'clock' in state:
                continue
            
            stream.write(state + ', ')

        for state in range(numDnnInputs):
            stream.write('_f' + str(state+1) + ', ')
            
        stream.write('t, clock\n\n')

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
        stream.write('\t\tmax jumps ' + str(5 * numSteps + 2) + '\n')  # F1/10 case study
        stream.write('\t\tprint on\n')
        stream.write('\t}\n\n')

        # encode modes-----------------------------------------------------------------------------
        stream.write('\tmodes\n')
        stream.write('\t{\n')

        writeControllerModes(stream, numDnnInputs)
        writePlantModes(stream, numDnnInputs, name='_cont_')
        writeEndMode(stream, 'm_end_pr', numDnnInputs)
        writeEndMode(stream, 'm_end_pl', numDnnInputs)
        writeEndMode(stream, 'm_end_hr', numDnnInputs)
        writeEndMode(stream, 'm_end_hl', numDnnInputs)
        writeEndMode(stream, 'm_end_sr', numDnnInputs)
        writeEndMode(stream, 'm_end_sl', numDnnInputs)
        writeEndMode(stream, 'm_left_wall', numDnnInputs)
        writeEndMode(stream, 'm_top_wall', numDnnInputs)
        writeEndMode(stream, 'm_right_bottom_wall', numDnnInputs)

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
        writeInitCond(stream, initProps, 'init_modem1')  # F1/10 (HSCC)

        # close top level brace
        stream.write('}\n')

        # encode unsafe set------------------------------------------------------------------------
        stream.write(safetyProps)


def main(argv):

    dnnYaml = argv[0]

    with open(dnnYaml, 'rb') as f:

        dnn = yaml.load(f)
    
    numSteps = 65

    wall_dist = getCornerDist()

    # F1/10 Safety + Reachability
    safetyProps = 'unsafe\n{\tm_left_wall\n\t{\n\t\ty1 <= ' + WALL_MIN + '\n\n\t}\n' \
        + '\tm_right_bottom_wall\n\t{'\
        + '\n\t\ty1 >= ' + WALL_MAX + '\n\t\ty2 >= ' + str(wall_dist - WALL_LIMIT) + '\n\n\t}\n' \
        + '\tm_top_wall\n\t{\n\t\t ' + str(NORMAL_TO_TOP_WALL[0]) + ' * y2 + ' + str(NORMAL_TO_TOP_WALL[1]) + ' * y1 <= ' + WALL_MIN + '\n\n\t}\n' \
        + '\t_cont_m2\n\t{\n\t\tk >= ' + str(numSteps-1) + '\n\n\t}\n' \
        + '\tm_end_pl\n\t{\n\t\ty1 <= ' + str(POS_LB) + '\n\n\t}\n' \
        + '\tm_end_pr\n\t{\n\t\ty1 >= ' + str(POS_UB) + '\n\n\t}\n' \
        + '\tm_end_hl\n\t{\n\t\ty4 >= ' + str(HEADING_UB) + '\n\n\t}\n' \
        + '\tm_end_hr\n\t{\n\t\ty4 <= ' + str(HEADING_LB) + '\n\n\t}\n' \
        + '\tm_end_sr\n\t{\n\t\ty3 >= ' + str(2.4 + SPEED_EPSILON) + '\n\n\t}\n' \
        + '\tm_end_sl\n\t{\n\t\ty3 <= ' + str(2.4 - SPEED_EPSILON) + '\n\n\t}\n}'

    modelFolder = '../flowstar_models'
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)

    modelFile = modelFolder + '/testModel'

    curLBPos = POS_LB
    posOffset = 0.05

    init_y2 = 8
    if TURN_ANGLE == -np.pi/2:
        init_y2 = 7

    count = 1

    while curLBPos < POS_UB:

        initProps = ['y1 in [' + str(curLBPos) + ', ' + str(curLBPos + posOffset) + ']',
                     'y2 in [' + str(init_y2) + ', ' + str(init_y2) + ']',
                     'y3 in [' + str(2.4 - SPEED_EPSILON) + ', ' + str(2.4 + SPEED_EPSILON) + ']',
                     'y4 in [' + str(HEADING_LB) + ', ' + str(HEADING_UB) + ']', 'k in [0, 0]', 'u in [0, 0]']  # F1/10

        curModelFile = modelFile + '_' + str(count) + '.model'

        writeComposedSystem(curModelFile, initProps, dnn, safetyProps, numSteps)

        args = '../flowstar_verisig/flowstar ' + dnnYaml + ' < ' + curModelFile
        _ = subprocess.Popen(args, shell=True, stdin=PIPE)

        curLBPos += posOffset
        count += 1


if __name__ == '__main__':
    main(sys.argv[1:])
