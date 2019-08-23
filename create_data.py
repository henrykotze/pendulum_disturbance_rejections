#!/usr/bin/env python3


from second_order import second_order
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import random as rand
import argparse
from single_pendulum_v2 import pendulum, noisy_pendulum
import os
import pickle
import shelve
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings
from tqdm import trange
parser = argparse.ArgumentParser(\
        prog='create data 2nd order',\
        description='Creates .npz files of step responses with given damping ratio\
                    frequency'
        )


parser.add_argument('-zeta', default = 1, help='the damping ratio the response, default: 1')
parser.add_argument('-wn', default= 2, help='the natural frequency of the response, default: 2')
parser.add_argument('-loc', default='./train_data/', help='location to store responses, default: ./train_data')
parser.add_argument('-filename', default="response-0.npz", help='filename, default: response-0.npz')
parser.add_argument('-t', default=10, help='time lenght of responses, default: 10s')
parser.add_argument('-numSim', default=2, help='number of responses to generate, default: 1')
parser.add_argument('-inputMag', default=1, help='magnitude of input given to system, default: +-1')
parser.add_argument('-system', default='pendulum', help='type of system to generate data, default: pendulum')
parser.add_argument('-init', default=0, help='Initial Condition, default: 0')
parser.add_argument('-rand', default=0, help='pick from normal distribution the input, default: 0')
parser.add_argument('-inputTime', default=50, help='time at which inputs starts, default: 50ms')
parser.add_argument('-startNumSim', default=0, help='to start the response-* different then 0, default: 0')
parser.add_argument('-dt', default=0.01, help='timestep increments of responses, default: 0.01')
parser.add_argument('-maxInput', default=0.5, help='maximum input given to system')
parser.add_argument('-minInput', default=-0.5, help='minimum input given to system')
parser.add_argument('-noise', default=0, help='use a noise pendulum system')
parser.add_argument('-randomInput', default=0, help='use a noise pendulum system')
parser.add_argument('-biases', default=0, help='add biases to the inputs')

parser.add_argument('-bias_freq', default=10, help='max frequency of bias')

parser.add_argument('-min_freq', default=10, help='min frequency of bias')
parser.add_argument('-max_freq', default=10, help='max frequency of bias')




args = parser.parse_args()

zeta=float(vars(args)['zeta'])
wn=float(vars(args)['wn'])
t=int(vars(args)['t'])
startSimNum = int(vars(args)['startNumSim'])
numberSims = int(vars(args)['numSim']) + startSimNum
dir = vars(args)['loc']
filename = vars(args)['loc']+'/'+vars(args)['filename']
system = str(vars(args)['system'])
initial = float(vars(args)['init'])
randomMag = int(vars(args)['rand'])
inputTime = int(vars(args)['inputTime'])
dt = float(vars(args)['dt'])
inputMag = float(vars(args)['inputMag'])
maxInput = float(vars(args)['maxInput'])
minInput = float(vars(args)['minInput'])

bias_freq = int(vars(args)['bias_freq'])

min_freq = int(vars(args)['min_freq'])
max_freq = int(vars(args)['max_freq'])


system_info = system

# Add a Readme file in directory to show selected variables that describe the
# responses

filename = filename.replace(str(0),str(startSimNum))


with shelve.open( str(dir+'/readme') ) as db:
    for arg in vars(args):
        db[arg] = getattr(args,arg)
db.close()



def determine_system(system,wn,zeta,initial_condition):
    response = pendulum(wn,zeta,y=initial_condition*np.pi/180,time_step=dt)
    return response

def generateDisturbance(responseDuration,startInput,minInput,maxInput):

    input = np.zeros( (responseDuration,1) )
    labels = np.zeros([responseDuration,max_freq-min_freq+1])
    # zeroInputDur = int(responseDuration/10*(np.random.random())    ) # Duration of zero input
    startInput = int(np.random.randint(0,responseDuration))
    timestep = startInput


    freq =  np.random.randint(min_freq,max_freq+1)
    inputDur = int(responseDuration-startInput)
    freq_content = np.zeros([1,max_freq-min_freq+1])
    # print(freq_content)
    # print(freq,freq-min_freq)
    freq_content[0,freq-min_freq] = 1

    magInput = (2)*np.random.random() # Magnitude Size of Input

    labels[0:startInput][:] = np.zeros([1,max_freq-min_freq+1])

    t = np.arange(timestep,timestep+inputDur)
    input[timestep:timestep+inputDur] = np.transpose(np.array([magInput*np.sin(2*np.pi*freq*t/inputDur)]))
    labels[timestep:timestep+inputDur][:] = freq_content

    return input,labels



def straightline_func(x, a, b):
    return a*x+b

def exponential_func(x, a, b):
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            y= a*np.exp(b*x)
            return y
        except:
            return 0*x

def square_func(x,a,b,c):
    return a*np.power(x,2) + b*x + c

def generateStepInput(responseDuration,startInput,minInput,maxInput):

    input = np.zeros( (responseDuration,1) )
    timestep = startInput

    while timestep < responseDuration:

        magInput = (maxInput-minInput)*np.random.random()+minInput # Magnitude Size of Input
        inputDur = int(responseDuration/10*(np.random.random() ) ) # Duration of input
        zeroInputDur = int(responseDuration/10*(np.random.random()) ) # Duration of zero input


        input[timestep:timestep+inputDur] = magInput
        timestep += inputDur
        input[timestep:timestep+zeroInputDur] = 0
        timestep += zeroInputDur

    return input

def generateRampInput(responseDuration,startInput,minInput,maxInput):

    input = np.zeros( (responseDuration,1) )
    timestep = startInput

    while timestep < responseDuration:
        magInput = (maxInput-minInput)*np.random.random()+minInput # peak point in ramp
        firstDur = int(responseDuration/10*(np.random.random()))+1 # Duration of first half ramp
        secondDur = int(responseDuration/10*(np.random.random()))+1 # Duration of second half ramp

        if(timestep + firstDur+secondDur < responseDuration):

            grad1 = magInput/firstDur   # gradient of first part
            grad2 = -magInput/secondDur  # Gradientr of second part

            firstLine = np.arange(firstDur)*grad1

            secondLine = -1*np.arange(secondDur,0,-1)*grad2
            input[timestep:timestep+firstDur] = np.transpose(np.array([firstLine]))
            timestep += firstDur
            input[timestep:timestep+secondDur] = np.transpose(np.array([secondLine]))
            timestep += secondDur
        else:
            break

    # input = addNoise(input,250)
    return input


def generateSquareInput(responseDuration,startInput,minInput,maxInput):

    input = np.zeros( (responseDuration,1) )
    timestep = startInput

    while timestep < responseDuration:

        y1 = ((maxInput-minInput)*np.random.random()+minInput)/3
        y2 = ((maxInput-minInput)*np.random.random()+minInput)/3
        y3 = ((maxInput-minInput)*np.random.random()+minInput)/3
        y4 = ((maxInput-minInput)*np.random.random()+minInput)/3

        x1 = int(responseDuration/10*(np.random.random()))+1
        x2 = int(responseDuration/10*(np.random.random()))+1
        x3 = int(responseDuration/10*(np.random.random()))+1

        x  = np.array([timestep+1,timestep+x1,timestep+x1+x2,timestep+x1+x2+x3])
        y = np.array([y1,y1+y2,y1+y2+y3,y1+y2+y3+y4])

        if(timestep + x1 + x2 +x3 < responseDuration):

            popt, pcov = curve_fit(square_func, x, y)
            c = popt[2]
            b = popt[1]
            a = popt[0]
            curve = np.arange(timestep,timestep+x1+x2+x3)
            curve = square_func(curve, a, b, c)
            input[timestep:timestep+x1+x2+x3] = np.transpose(np.array([curve]))
            timestep = timestep + x1 + x2 + x3

        else:
            break

    # input = addNoise(input,250)
    return input

def generateExpoInput(responseDuration,startInput,minInput,maxInput):

    input = np.zeros( (responseDuration,1) )
    timestep = startInput
    error_check = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while timestep < responseDuration:


            y11 = ((maxInput-minInput)*np.random.random()+minInput)/2 #
            y12 = ((maxInput-minInput)*np.random.random()+minInput)/2 #

            x11 = int(responseDuration/10*(np.random.random())) #
            x12 = int(responseDuration/10*(np.random.random()))+20 #

            y21 = ((maxInput-minInput)*np.random.random()+minInput)/2 #

            x21 = int(responseDuration/10*(np.random.random())) #
            x22 = int(responseDuration/10*(np.random.random()))+20 #

            if(timestep + x11+ x12 + x21 + x22 + 1 < responseDuration):

                y = np.log(np.array([0.00001, y11, y11+y12]))
                x = np.array([timestep+1,timestep+x11,timestep+x11+x12])
                popt, pcov = curve_fit(straightline_func, x, y)
                b = popt[0]
                a = np.exp(popt[1])
                curve = np.arange(timestep,timestep+x11+x12)
                curve = exponential_func(curve, a, b)
                input[timestep:timestep+x11+x12] = np.transpose(np.array([curve]))

                y = np.log(np.array([y11+y12,y21, 0.0001]))
                x = np.array([timestep+x11+x12,timestep+x11+x12+x21, timestep+x11+x12+x21+x22])
                popt, pcov = curve_fit(straightline_func, x, y)
                b = popt[0]
                a = np.exp(popt[1])
                curve = np.arange(timestep+x11+x12,timestep+x11+x12+x21+x22)
                curve = exponential_func(curve, a, b)
                input[timestep+x11+x12:timestep+x11+x12+x21+x22] = np.transpose(np.array([curve]))
                timestep = timestep + x11 + x12 + x21 + x22

            else:
                break
    return input

def generateNoiseInput(responseDuration,startInput,minInput,maxInput):

    input = np.zeros( (responseDuration,1) )
    input += (maxInput-minInput)*np.random.random((np.size(input),1))+minInput
    return input

def addNoise(response,level):
    sizeOfArray = np.size(response)
    response += np.random.random((sizeOfArray,1))/level
    return response

def generateCombinationInput(responseDuration,startInput,minInput,maxInput):
    input1 = generateStepInput(responseDuration,startInput,minInput/4,maxInput/4)
    input2 = generateRampInput(responseDuration,startInput,minInput/4,maxInput/4)
    input3 = generateExpoInput(responseDuration,startInput,minInput/4,maxInput/4)
    input4 = generateSquareInput(responseDuration,startInput,minInput/4,maxInput/4)
    input = addNoise(input1+input2+input3+input4,500)
    return input



if __name__ == '__main__':
    print('Creating the response of ', str(system_info))
    print('Writing responses to:', filename )
    # print(startSimNum, numberSims)

    timeSteps= int(t/dt) # time in number of step
    numSim = trange(numberSims, desc='# of response', leave=True)

    i = 0
    while i < numberSims:

        numSim.set_description("# of response (%s)" %filename)
        numSim.refresh() # to show immediately the update
        # for numSim in range(startSimNum,numberSims):
        # print('Number of responses: ', numSim)
        # response = determine_system(system,wn,zeta,initial)
        response = pendulum(wn,zeta,y=initial*np.pi/180,time_step=dt)
        biased_response = pendulum(wn,zeta,y=initial*np.pi/180,time_step=dt)

        [bias,bias_labels] = generateDisturbance(timeSteps,inputTime,minInput/5,maxInput/5)

        input = generateCombinationInput(timeSteps,inputTime,minInput,maxInput)

        y = np.zeros( (timeSteps,1) )
        ydot = np.zeros( (timeSteps,1) )
        ydotdot = np.zeros( (timeSteps,1) )
        # Biased response
        biased_y = np.zeros( (timeSteps,1) )
        biased_ydot = np.zeros( (timeSteps,1) )
        biased_ydotdot = np.zeros( (timeSteps,1) )

        for t in range(0,timeSteps):

            # time at which input starts
            biased_response.update_input( input[t]+bias[t] )
            response.update_input( input[t])


            # temporary variables
            t1,t2,t3,t4 = response.getAllStates()
            input[t] = t1
            y[t] = t4
            ydot[t] = t3
            ydotdot[t] = t2

            t1,t2,t3,t4 = biased_response.getAllStates()
            biased_y[t] = t4
            biased_ydot[t] = t3
            biased_ydotdot[t] = t2



            # next time step
            response.step()
            biased_response.step()


            # y = addNoise(y,500)

        # Saves response in *.npz file
        # print(system)
        np.savez(filename,input=input,y_=y,y_dot=ydot,y_dotdot=ydotdot,zeta=zeta,wn=wn,system=str(system_info),bias=bias,
        biased_y=biased_y,biased_y_dot=biased_ydot,biased_y_dotdot=biased_ydotdot,bias_labels=bias_labels)

        # Change number on filename to correspond to simulation number
        filename = filename.replace(str(i),str(i+1))
        numSim.update()
        i += 1
