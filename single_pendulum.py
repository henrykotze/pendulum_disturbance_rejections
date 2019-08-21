import scipy.integrate as spi
import matplotlib.pyplot as plt
import numpy as np




class pendulum:
    def __init__(self,wn,zeta,time_step=0.001,input=0, ydotdot = 0, ydot = 0, y = 0):
        self.wn = wn
        self.zeta = zeta
        self.input = input
        self.ydotdot = ydotdot
        self.ydot = ydot
        self.y = y
        self.t = 0
        self.dt = time_step
        self.numStates = 3


    def update_ydotdot(self):
        self.ydotdot = self.input - 2*self.zeta*self.wn*self.ydot - np.power((self.wn),2)*np.sin(self.y)


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
        return np.r_[self.input, self.ydotdot,self.ydot,self.y]

    def getEstimatedStates(self):
        return np.r_[self.ydotdot,self.ydot,self.y]
