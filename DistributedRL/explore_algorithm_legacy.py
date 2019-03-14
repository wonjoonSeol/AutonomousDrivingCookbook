# -*- coding: utf-8 -*-

from Share.scripts_downpour.app.airsim_client import *
import numpy as np
import time
import sys
import json
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import PIL
import PIL.ImageFilter

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, clone_model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam, Adagrad, Adadelta
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
import keras.backend as K
from keras.preprocessing import image
from keras.initializers import random_normal


def compute_reward(car_state, collision_info, road_points):
    #Define some constant parameters for the reward function
    THRESH_DIST = 3.5                # The maximum distance from the center of the road to compute the reward function
    DISTANCE_DECAY_RATE = 1.2        # The rate at which the reward decays for the distance function
    CENTER_SPEED_MULTIPLIER = 2.0    # The ratio at which we prefer the distance reward to the speed reward
    
    # If the car is stopped, the reward is always zero
    speed = car_state.speed
    if (speed < 2):
        return 0
    
    #Get the car position
    position_key = bytes('position', encoding='utf8')
    x_val_key = bytes('x_val', encoding='utf8')
    y_val_key = bytes('y_val', encoding='utf8')

    car_point = np.array([car_state.kinematics_true[position_key][x_val_key], car_state.kinematics_true[position_key][y_val_key], 0])
    
    # Distance component is exponential distance to nearest line
    distance = 999
    
    #Compute the distance to the nearest center line
    for line in road_points:
        local_distance = 0
        length_squared = ((line[0][0]-line[1][0])**2) + ((line[0][1]-line[1][1])**2)
        if (length_squared != 0):
            t = max(0, min(1, np.dot(car_point-line[0], line[1]-line[0]) / length_squared))
            proj = line[0] + (t * (line[1]-line[0]))
            local_distance = np.linalg.norm(proj - car_point)
        
        distance = min(distance, local_distance)
        
    distance_reward = math.exp(-(distance * DISTANCE_DECAY_RATE))
    
    return distance_reward

plt.figure()
   
# Reads in the reward function lines
def init_reward_points():
    road_points = []
    with open('Share\\data\\reward_points.txt', 'r') as f:
        for line in f:
            point_values = line.split('\t')
            first_point = np.array([float(point_values[0]), float(point_values[1]), 0])
            second_point = np.array([float(point_values[2]), float(point_values[3]), 0])
            road_points.append(tuple((first_point, second_point)))

    return road_points

#Draws the car location plot
def draw_rl_debug(car_state, road_points):

    print('')
    for point in road_points:
        plt.plot([point[0][0], point[1][0]], [point[0][1], point[1][1]], 'k-', lw=2)
    
    position_key = bytes('position', encoding='utf8')
    x_val_key = bytes('x_val', encoding='utf8')
    y_val_key = bytes('y_val', encoding='utf8')
    
    car_point = np.array([car_state.kinematics_true[position_key][x_val_key], car_state.kinematics_true[position_key][y_val_key], 0])
    plt.plot([car_point[0]], [car_point[1]], 'bo')
    
    plt.draw()
    plt.pause(0.0001)
    plt.clf()
    
reward_points = init_reward_points()
    
car_client = CarClient()
car_client.confirmConnection()
car_client.enableApiControl(False)
fig = plt.figure(figsize=(15,15))
    
try:
    while(True):
        clear_output(wait=True)
        car_state = car_client.getCarState()
        collision_info = car_client.getCollisionInfo()
        reward = compute_reward(car_state, collision_info, reward_points)
        print('Current reward: {0:.2f}'.format(reward))
        draw_rl_debug(car_state, reward_points)
        time.sleep(1)

#Handle interrupt gracefully
except:
    pass

