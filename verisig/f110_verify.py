'''
Copyright (C) 2019 Radoslav Ivanov, Taylor J Carpenter, James Weimer, Rajeev Alur, George J. Pappa, Insup Lee

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

Example usage:

python verisig_multi_runner.py ../dnns/TD3_L21_64x64_C1.yml ../plant_models/dynamics_21.pickle ../plant_models/glue_21.pickle

'''

from six.moves import cPickle as pickle
import os
import subprocess
from subprocess import PIPE
import sys


def writeControllerModes(stream, numRays, dynamics):

    # first mode
    writeOneMode(stream, 0, numRays, dynamics)

    # Ouput mode
    writeOneMode(stream, 1, numRays, dynamics)


def writeOneMode(stream, modeIndex, numStates, dynamics, name=''):
    stream.write('\t\t' + name + 'm' + str(modeIndex) + '\n')
    stream.write('\t\t{\n')
    stream.write('\t\t\tnonpoly ode\n')
    stream.write('\t\t\t{\n')

    lidarStates = []

    for lidarState in range(numStates):

        fName = 'f' + str(lidarState + 1)
        lidarStates.append(fName)

        stream.write('\t\t\t\t' + fName + '\' = 0\n')

    for sysState in dynamics:

        if sysState not in lidarStates:
            if sysState != 'clock':
                stream.write('\t\t\t\t' + sysState + '\' = 0\n')

    stream.write('\t\t\t\tclock\' = 1\n')
    stream.write('\t\t\t}\n')

    stream.write('\t\t\tinv\n')
    stream.write('\t\t\t{\n')

    stream.write('\t\t\t\tclock <= 0\n')

    stream.write('\t\t\t}\n')
    stream.write('\t\t}\n')


def writePlantModes(stream, plant, numRays, numNeurLayers):

    for modeId in plant:

        modeName = ''
        if 'name' in plant[modeId] and len(plant[modeId]['name']) > 0:
            modeName = plant[modeId]['name']

        stream.write('\t\t' + modeName + 'm' +
                     str(numNeurLayers + modeId) + '\n')
        stream.write('\t\t{\n')
        stream.write('\t\t\tnonpoly ode\n')
        stream.write('\t\t\t{\n')

        for sysState in plant[modeId]['dynamics']:
            if sysState != 'clock':
                stream.write('\t\t\t\t' + plant[modeId]['dynamics'][sysState])

        for i in range(numRays):

            fName = 'f' + str(i + 1)

            # these if-cases check if f/c variables are also used in the plant model
            # (sometimes the case in order to minimize number of states)
            if fName not in plant[modeId]['dynamics']:
                stream.write('\t\t\t\t' + fName + '\' = 0\n')

        stream.write('\t\t\t\tclock\' = 1\n')
        stream.write('\t\t\t}\n')
        stream.write('\t\t\tinv\n')
        stream.write('\t\t\t{\n')

        usedClock = False

        for inv in plant[modeId]['invariants']:
            stream.write('\t\t\t\t' + inv + '\n')

            if 'clock' in inv:
                usedClock = True

        if not usedClock:
            stream.write('\t\t\t\tclock <= 0')

        stream.write('\n')
        stream.write('\t\t\t}\n')
        stream.write('\t\t}\n')


def writeControllerJumps(stream, numRays, dynamics):

    # jump from m0 to DNN-----------------------------------------------------
    writeConstantControllerJump(stream, 'm0', 'm1', numRays, dynamics)


def writeConstantControllerJump(stream, curModeName, nextModeName, numRays, dynamics):

    stream.write('\t\t' + curModeName + ' -> ' + nextModeName + '\n')

    stream.write('\t\tguard { clock = 0 }\n')

    stream.write('\t\treset { ')
    stream.write('f1\' := ' + str(0.5) + ' ')

    # not resetting plant states in dnn jumps anymore since they might overlap
    # for sysState in dynamics:
    #     stream.write(str(sysState) +'\' := ' + str(sysState) + ' ')

    stream.write('clock\' := 0')
    stream.write('}\n')
    stream.write('\t\tinterval aggregation\n')


def writePlantJumps(stream, plant, numRays, numNeurLayers):

    for modeId in plant:
        for trans in plant[modeId]['transitions']:

            for i in range(1, int(round(len(plant[modeId]['transitions'][trans])/2)) + 1):

                curModeName = ''
                nextModeName = ''

                if 'name' in plant[modeId] and len(plant[modeId]['name']) > 0:
                    curModeName = plant[modeId]['name']

                if 'name' in plant[trans[1]] and len(plant[trans[1]]['name']) > 0:
                    nextModeName = plant[trans[1]]['name']

                stream.write('\t\t' + curModeName + 'm' + str(trans[0] + numNeurLayers) +
                             ' -> ' + nextModeName + 'm' + str(trans[1] + numNeurLayers) + '\n')
                stream.write('\t\tguard { ')

                for guard in plant[modeId]['transitions'][trans]['guards' + str(i)]:
                    stream.write(guard + ' ')

                stream.write('}\n')

                stream.write('\t\treset { ')

                usedClock = False

                for reset in plant[modeId]['transitions'][trans]['reset' + str(i)]:
                    stream.write(reset + ' ')
                    if 'clock' in reset:
                        usedClock = True

                if not usedClock:
                    stream.write('clock\' := 0')

                stream.write('}\n')
                stream.write('\t\tinterval aggregation\n')


def writeController2PlantJumps(stream, trans, numRays, numNeurLayers, plant):

    for modeId in trans:

        for i in range(1, int(round(len(trans[modeId])/2)) + 1):

            stream.write('\t\tm1 -> ')

            if 'name' in plant[modeId]:
                stream.write(plant[modeId]['name'])

            stream.write('m' + str(numNeurLayers + modeId) + '\n')
            stream.write('\t\tguard { ')

            for guard in trans[modeId]['guards' + str(i)]:
                stream.write(guard + ' ')

            stream.write('}\n')

            stream.write('\t\treset { ')

            for reset in trans[modeId]['reset' + str(i)]:
                stream.write(reset + ' ')

            # for state in range(numNeurStates):
            #     stream.write('f' + str(state + 1) + '\' := f' + str(state + 1) + ' ')

            stream.write('clock\' := 0')
            stream.write('}\n')
            stream.write('\t\tinterval aggregation\n')


def writePlant2ControllerJumps(stream, trans, dynamics, numRays, numNeurLayers):

    for nextTrans in trans:

        for i in range(1, int(round(len(trans[nextTrans])/2)) + 1):

            stream.write('\t\tm' + str(nextTrans + numNeurLayers) + ' -> m0\n')
            stream.write('\t\tguard { ')

            for guard in trans[nextTrans]['guards' + str(i)]:
                stream.write(guard + ' ')

            stream.write('}\n')

            stream.write('\t\treset { ')

            for reset in trans[nextTrans]['reset' + str(i)]:
                stream.write(reset + ' ')

            stream.write('clock\' := 0')
            stream.write('}\n')
            stream.write('\t\tinterval aggregation\n')


def writeInitCond(stream, initProps, numInputs, numRays, initState='m0'):

    stream.write('\tinit\n')
    stream.write('\t{\n')
    stream.write('\t\t' + initState + '\n')
    stream.write('\t\t{\n')

    for prop in initProps:
        stream.write('\t\t\t' + prop + '\n')

    for i in range(numRays):
        stream.write('\t\t\tf' + str(i + 1) + ' in [0, 0]\n')

    stream.write('\t\t\tclock in [0, 0]\n')
    stream.write('\t\t}\n')
    stream.write('\t}\n')


def getInputLBUB(state, bounds, weights, offsets):
    lbSum = 0
    ubSum = 0

    varIndex = 0
    for inVar in bounds:
        weight = weights[1][state][varIndex]
        if weight >= 0:
            lbSum += weight * bounds[inVar][0]
            ubSum += weight * bounds[inVar][1]
        else:
            lbSum += weight * bounds[inVar][1]
            ubSum += weight * bounds[inVar][0]

        varIndex += 1

    lb = lbSum + offsets[1][state]
    ub = ubSum + offsets[1][state]

    numLayers = len(offsets)
    if numLayers > 1:
        for layer in range(1, numLayers):
            lbSum = 0
            ubSum = 0

            for weight in weights[layer + 1][state]:
                if weight >= 0:
                    ubSum += weight
                else:
                    lbSum += weight

            if ubSum + offsets[layer + 1][state] > ub:
                ub = ubSum + offsets[layer + 1][state]

            if lbSum + offsets[layer + 1][state] < lb:
                lb = lbSum + offsets[layer + 1][state]

    return (lb, ub)


'''
1. initProps is a list of properties written in strings that can be parsed by Flow*
  -- assumes the states are given as 'xi'
2. dnn is a dictionary such that:
  -- key 'weights' is a dictionary mapping layer index
     to a MxN-dimensional list of weights
  -- key 'offsets'  is a dictionary mapping layer index
     to a list of offsets per neuron in that layer
  -- key 'activations' is a dictionary mapping layer index
     to the layer activation function type
3. plant is a dictionary such that:
  -- Each dictionary key is a mode id that maps to a dictionary such that:
    -- key 'dynamics' maps to a dictionary of the dynamics of each var in that mode such that:
      -- each key is of the form 'xi' and maps to a dynamics string that can be parsed by Flow*
      -- assume inputs in dynamics are coded as 'ci' to make composition work
    -- key 'invariants' maps to a list of invariants that can be parsed by Flow*
    -- key 'transitions' maps to a dictionary such that:
      -- each key is a tuple of the form '(mode id, mode id)' that maps to a dictionary such that:
        -- key 'guards' maps to a list of guards that can be parsed by Flow*
        -- key 'reset' maps to a list of resets that can be parsed by Flow*
    -- key 'odetype' maps to a string describing the Flow* dynamics ode type
4. glueTrans is a dictionary such that:
  -- key 'dnn2plant' maps to a dictionary such that:
    -- each key is an int specifying plant mode id that maps to a dictionary such that:
       -- key 'guards' maps to a list of guards that can be parsed by Flow*
       -- key 'reset' maps to a list of resets that can be parsed by Flow*
  -- key 'plant2dnn' maps to a dictionary such that:
    -- each key is an int specifying plant mode id that maps to a dictionary such that:
       -- key 'guards' maps to a list of guards that can be parsed by Flow*
       -- key 'reset' maps to a list of resets that can be parsed by Flow*
5. safetyProps is assumed to be a string containing a
   logic formula that can be parsed by Flow*'''


def writeComposedSystem(filename, initProps, numRays, plant, glueTrans, safetyProps, numSteps):

    with open(filename, 'w') as stream:

        stream.write('hybrid reachability\n')
        stream.write('{\n')

        # encode variable names--------------------------------------------------
        stream.write('\t' + 'state var ')

        lidarStates = []
        for i in range(numRays):
            fName = 'f' + str(i + 1)
            lidarStates.append(fName)

        if 'states' in plant[1]:
            for index in range(len(plant[1]['states'])):
                if not plant[1]['states'][index] in lidarStates:
                    stream.write(plant[1]['states'][index] + ', ')
        else:
            for state in plant[1]['dynamics']:
                if 'clock' in state:
                    continue
                stream.write(state + ', ')

        for i in range(numRays):
            stream.write('f' + str(i + 1) + ', ')

        stream.write('clock\n\n')

        # settings---------------------------------------------------------------------------------
        stream.write('\tsetting\n')
        stream.write('\t{\n')
        # F1/10 case study (HSCC)
        stream.write('\t\tadaptive steps {min 1e-6, max 0.005}\n')
        stream.write('\t\ttime ' + str(numSteps * (0.1)) +
                     '\n')  # F1/10 case study (HSCC)
        stream.write('\t\tremainder estimation 1e-1\n')
        stream.write('\t\tidentity precondition\n')
        stream.write('\t\tgnuplot octagon y1, y2\n')
        stream.write('\t\tfixed orders 4\n')
        stream.write('\t\tcutoff 1e-12\n')
        stream.write('\t\tprecision 100\n')
        stream.write('\t\toutput autosig\n')
        stream.write('\t\tmax jumps ' + str((1 + 2 + 10 +
                                             5 * numRays) * numSteps) + '\n')  # F1/10 case study
        stream.write('\t\tprint on\n')
        stream.write('\t}\n\n')

        # encode modes-----------------------------------------------------------------------------
        stream.write('\tmodes\n')
        stream.write('\t{\n')

        writeControllerModes(stream, numRays, plant[1]['dynamics'])
        writePlantModes(stream, plant, numRays, 1)

        # close modes brace
        stream.write('\t}\n')

        # encode jumps-----------------------------------------------------------------------------
        stream.write('\tjumps\n')
        stream.write('\t{\n')

        writeControllerJumps(stream, numRays, plant[1]['dynamics'])
        writeController2PlantJumps(
            stream, glueTrans['dnn2plant'], numRays, 1, plant)
        writePlantJumps(stream, plant, numRays, 1)
        writePlant2ControllerJumps(
            stream, glueTrans['plant2dnn'], plant[1]['dynamics'], numRays, 1)

        # close jumps brace
        stream.write('\t}\n')

        # encode initial condition-----------------------------------------------------------------
        writeInitCond(stream, initProps, numRays,
                      numRays, 'm3')  # F1/10 (HSCC)

        # close top level brace
        stream.write('}\n')

        # encode unsafe set------------------------------------------------------------------------
        stream.write(safetyProps)


def main(argv):

    # dnnYaml = argv[0]
    numRays = int(argv[0])
    plantPickle = '../plant_models/dynamics_{}.pickle'.format(numRays)
    gluePickle = '../plant_models/glue_{}.pickle'.format(numRays)

    with open(plantPickle, 'rb') as f:

        plant = pickle.load(f)

    with open(gluePickle, 'rb') as f:

        glue = pickle.load(f)

    numSteps = 70

    # F1/10 (HSCC)
    safetyProps = 'unsafe\n{\tcont_m2\n\t{\n\t\ty1 <= 0.3\n\n\t}\n\tcont_m2\n\t{\n\t\ty1 >= 1.2\n\t\ty2 >= 1.5\n\n\t}\n\tcont_m2\n\t{\n\t\ty1 >= 1.5\n\t\ty2 >= 1.2\n\n\t}\n\tcont_m2\n\t{\n\t\ty2 <= 0.3\n\n\t}\n}'

    modelFolder = '../flowstar_models'
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)

    modelFile = modelFolder + '/testModel'

    curLBPos = 0.65
    posOffset = 0.005

    count = 1

    while curLBPos < 0.66:

        initProps = ['y1 in [' + str(curLBPos) + ', ' + str(curLBPos + posOffset) + ']',
                     'y2 in [9.9, 9.9]', 'y3 in [0, 0]', 'y4 in [0, 0]', 'k in [0, 0]',
                     'u in [0, 0]', 'angle in [0, 0]', 'temp1 in [0, 0]', 'temp2 in [0, 0]',
                     'theta_l in [0, 0]', 'theta_r in [0, 0]']  # F1/10

        curModelFile = modelFile + '_' + str(count) + '.model'

        writeComposedSystem(curModelFile, initProps, numRays,
                            plant, glue, safetyProps, numSteps)

        args = '../../flowstar/flowstar' + ' < ' + curModelFile
        _ = subprocess.Popen(args, shell=True, stdin=PIPE)

        curLBPos += posOffset
        count += 1


if __name__ == '__main__':
    main(sys.argv[1:])
