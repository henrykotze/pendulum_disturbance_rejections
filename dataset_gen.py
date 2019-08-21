#!/usr/bin/env python3


import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
from second_order import second_order
import numpy as np
# import scipy.integrate as spi
# import matplotlib.pyplot as plt
import random as rand
import argparse
from single_pendulum import pendulum
import os
import pickle
import shelve
import h5py


parser = argparse.ArgumentParser(\
        prog='create data 2nd order',\
        description='Creates .npz files of step responses with given damping ratio\
                    frequency'
        )


parser.add_argument('-loc', default='./biased_train_data/', help='location to stored responses, default: ./train_data')
parser.add_argument('-Nt', default=5, help='number of previous output timesteps used, default: 5')
parser.add_argument('-dataset_name', default='biased_dataset0', help='name of your dataset, default: dataset*')
parser.add_argument('-dataset_loc', default='./datasets/', help='location to store dataset: ./datasets/')


args = parser.parse_args()

dir = vars(args)['loc']

# Getting information from readme file of training data

print('----------------------------------------------------------------')
print('Fetching training info from: ', str(dir+'/readme'))
print('----------------------------------------------------------------')
with shelve.open( str(dir+'/readme')) as db:
    system = db['system']
    t = int(db['t'])
    dt = float(db['dt'])
    numberSims = int(db['numSim'])
    filename = db['filename']
    bias_activated = int(db['biases'])
    maxInput = float(db['maxInput'])
    bias_freq = int(db['bias_freq'])
    bias_offset = float(db['bias_offset'])
    bias_mag = float(db['bias_mag'])
db.close()

if(not bias_activated):
    raise Exception('The data being loaded does not contain an bias')



N_t = int(vars(args)['Nt'])
timeSteps = int(t/dt)
nameOfDataset = str(vars(args)['dataset_name'])
dataset_loc = str(vars(args)['dataset_loc'])



with shelve.open( str(dataset_loc + '/'+nameOfDataset+'_readme') ) as db:
    for arg in vars(args):
        db[arg] = getattr(args,arg)

    with shelve.open(str(dir+'/readme')) as data_readme:
        for key in data_readme:
            db[key] = data_readme[key]

    data_readme.close()
db.close()

if __name__ == '__main__':

    # Pre-creating correct sizes of arrays
    features = np.zeros( (timeSteps*numberSims,5*N_t) )   # +1 is for the input
    labels = np.zeros( (timeSteps*numberSims,bias_freq) )
    max_input = 0
    max_bias_ydotdot = 0
    max_ydotdot = 0

    for numFile in range(numberSims):
        with np.load(str(dir+'/'+filename)) as data:
            print('Loading Data from: ', filename)

            biased_response_y = data['biased_y'] # inputs from given file
            response_y = data['y_'] # inputs from given file

            biased_response_y_dotdot = data['biased_y_dotdot'] # inputs from given file
            response_y_dotdot = data['y_dotdot'] # inputs from given file

            input = data['input']
            bias = data['bias']
            bias_labels  = data['bias_labels']

            if(np.amax(input) > max_input):
                 max_input = np.amax(input)

            if(np.amax(biased_response_y_dotdot) > max_bias_ydotdot):
                 max_bias_ydotdot = np.amax(biased_response_y_dotdot)

            if(np.amax(response_y_dotdot) > max_ydotdot):
                 max_ydotdot = np.amax(response_y_dotdot)

            for step in range( N_t, timeSteps- N_t ):
                # labels[step+timeSteps*numFile,0] = bias_labels[step][0]/bias_mag
                # labels[step+timeSteps*numFile,1] = bias_labels[step][1]/bias_freq
                # labels[step+timeSteps*numFile,2] = bias_labels[step][2]/bias_offset
                labels[step+timeSteps*numFile,:] = bias_labels[step]
                # print(bias_labels[step])

                for n in range(0,N_t):
                    features[step+timeSteps*numFile,n] = input[step-n]/maxInput
                for n in range(0,N_t):
                    features[step+timeSteps*numFile,N_t+n] = np.sin(response_y[step-n])
                for n in range(0,N_t):
                    features[step+timeSteps*numFile,2*N_t+n] = np.sin(biased_response_y[step-n])
                for n in range(0,N_t):
                    features[step+timeSteps*numFile,3*N_t+n] = response_y_dotdot[step-n]
                for n in range(0,N_t):
                    features[step+timeSteps*numFile,4*N_t+n] = biased_response_y_dotdot[step-n]
            db.close()

            # fetch next name of *.npz file to be loaded
            filename = filename.replace(str(numFile),str(numFile+1))


    features[:,3*N_t:3*N_t+N_t] = features[:,3*N_t:3*N_t+N_t]/max_ydotdot
    features[:,4*N_t:4*N_t+N_t] = features[:,4*N_t:4*N_t+N_t]/max_bias_ydotdot

    print()
    print('----------------------------------------------------------------')
    print('Information from all Responses')
    print('----------------------------------------------------------------')
    print('max_bias_ydotdot: ', max_bias_ydotdot)
    print('max_ydotdot: ', max_ydotdot)
    print('----------------------------------------------------------------')



    with shelve.open( str(dataset_loc + '/'+nameOfDataset+'_readme')) as db:
        db['max_ydotdot'] = max_ydotdot
        db['max_bias_ydotdot'] = max_bias_ydotdot
    db.close()

    h5f = h5py.File(str(dataset_loc + '/'+nameOfDataset),'w')
    h5f.create_dataset('features', data=features)
    h5f.create_dataset('labels', data=labels)
    h5f.close()

    # with open(dataset_loc + '/'+nameOfDataset,'wb+') as filen:
    #
    #     print('\n--------------------------------------------------------------')
    #     print('Saving features and labels to:', nameOfDataset)
    #     print('\n--------------------------------------------------------------')
    #
    #     pickle.dump([features,labels],filen)


    # filen.close()
