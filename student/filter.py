# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        d = params.dim_state
        n = int(d/2)
        dt = params.dt

        F = np.matrix(np.identity(d)) # in order to have it as an np.matrix instead of np.array; this is so as to ensure '*' works as matrix multiplication
        F[0:n, n:2*n] = dt*np.matrix(np.identity(n)) # Set the third off-diagonal entries to dt

        return F
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        d = params.dim_state
        n = int(d/2)
        dt = params.dt
        q = params.q

        Q = np.zeros((d, d))
        Q[0:n, 0:n] = q/3*dt**3*np.matrix(np.identity(n))
        Q[0:n, n:2*n] = q/2*dt**2*np.matrix(np.identity(n))
        Q[n:2*n, 0:n] = q/2*dt**2*np.matrix(np.identity(n))
        Q[n:2*n, n:2*n] = q*dt*2*np.matrix(np.identity(n))
        
        return Q
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        F, Q = self.F(), self.Q()
        x, P = track.x, track.P

        x = F*x
        P = F*P*F.transpose() + Q

        track.set_x(x)
        track.set_P(P)
        # print("P_max = "+str(np.max([P[0, 0], P[1, 1]])))
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        d = params.dim_state
        x, P = track.x, track.P
        H = meas.sensor.get_H(x)

        S = self.S(track, meas, H)
        # Check if matrix S is invertible, i.e. if determinant is larger than system precision
        # Proposed in https://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
        if np.linalg.cond(S) < 1/sys.float_info.epsilon:
            K = P*H.transpose()*np.linalg.inv(S)
            gamma = self.gamma(track, meas)

            x = x + K*gamma
            P = (np.matrix(np.identity(d)) - K*H)*P 

            track.set_x(x)
            track.set_P(P)
        else:
            raise ZeroDivisionError()
            
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############        
        x = track.x
        hx = meas.sensor.get_hx(x)
        z = meas.z

        return z - hx
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        # Why is H an argument of the function? H is already given in meas.
        P = track.P
        R = meas.R

        return H*P*H.transpose() + R 
        
        ############
        # END student code
        ############ 