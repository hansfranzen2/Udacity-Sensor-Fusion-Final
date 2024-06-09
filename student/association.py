# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        
        # # the following only works for at most one track and one measurement
        # self.association_matrix = np.matrix([]) # reset matrix
        # self.unassigned_tracks = [] # reset lists
        # self.unassigned_meas = []
        
        # if len(meas_list) > 0:
        #     self.unassigned_meas = [0]
        # if len(track_list) > 0:
        #     self.unassigned_tracks = [0]
        # if len(meas_list) > 0 and len(track_list) > 0: 
        #     self.association_matrix = np.matrix([[0]])

        # My code:
        N, M = len(track_list), len(meas_list)
        self.unassigned_tracks = list(range(N))
        self.unassigned_meas = list(range(M))
        
        self.association_matrix = np.inf*np.ones((N, M)) # Initialize as N x M matrix with entries infinity

        for i in self.unassigned_tracks:
            track = track_list[i]
            for j in self.unassigned_meas:
                meas = meas_list[j]
                MHD = self.MHD(track, meas, KF)
                if self.gating(MHD, meas.sensor):
                    self.association_matrix[i, j] = MHD

        
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # # the following only works for at most one track and one measurement
        # update_track = 0
        # update_meas = 0
        
        # # remove from list
        # self.unassigned_tracks.remove(update_track) 
        # self.unassigned_meas.remove(update_meas)
        # self.association_matrix = np.matrix([])

        # My code
        A = self.association_matrix

        # print("A = "+str(A))

        # If every entry is infinity then no associations are made
        if np.min(A) == np.inf:
            return np.nan, np.nan
        
        i, j = np.unravel_index(np.argmin(A, axis=None), A.shape)
        
        update_track = self.unassigned_tracks[i]
        update_meas = self.unassigned_meas[j]

        # Delete i-th row and j-th column 
        A = np.delete(A, i, axis=0)
        A = np.delete(A, j, axis=1)
        self.association_matrix = A

        # Remove from unassigned lists
        self.unassigned_tracks.remove(update_track)
        self.unassigned_meas.remove(update_meas)
            
        ############
        # END student code
        ############ 
        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        df = sensor.dim_meas # df should be 2 or 3 depending on if we're dealing with camera or lidar measurements; nice hint with the function argument :)
        limit = chi2.ppf(params.gating_threshold, df=df)

        return (MHD < limit)
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        x = track.x

        gamma = KF.gamma(track, meas)
        H = meas.sensor.get_H(x)
        S = KF.S(track, meas, H)

        if np.linalg.cond(S) < 1/sys.float_info.epsilon: # Check again for invertibility, as in update() in filter.py
            MHD = gamma.transpose()*np.linalg.inv(S)*gamma
            return MHD
        else:
            raise ZeroDivisionError()
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)