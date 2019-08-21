#!/usr/bin/env python3
import scipy.integrate as spi
import matplotlib.pyplot as plt
import numpy as np




class second_order:
    def __init__(self,wn=1.5,zeta=0.5,time_step=0.01,input=0, ydotdot = 0, ydot = 0, y = 0, y_wp=0):
        self.wn = wn
        self.zeta = zeta
        self.input = input
        self.ydotdot = ydotdot
        self.ydot = ydot
        self.y = y
        self.y_wp = y_wp
        self.t = 0
        self.dt = time_step
        self.numStates=3


    def update_ydotdot(self):
        self.ydotdot = self.input - 2*self.zeta*self.wn*self.ydot - np.power((self.wn),2)*(self.y-self.y_wp)


    def update_input(self,input):
        self.input = input

    def set_zeta(self,zeta):
        self.zeta = zeta

    def set_wn(self,wn):
        self.wn = wn


    def integration(self,x):
        from scipy.integrate import quad
        f = lambda t: x
        return quad(f,self.t, self.t+self.dt)


    def step(self):
        self.update_ydotdot()


        self.ydot += self.integration(self.ydotdot)[0]
        self.y += self.integration(self.ydot)[0]

        self.t += self.dt



    def getAllStates(self):
        return np.array([self.input, self.ydotdot,self.ydot,self.y])

    def getEstimatedStates(self):
        return np.r_[self.ydotdot,self.ydot,self.y]
