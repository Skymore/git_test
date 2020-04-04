#!/usr/bin/env python

from collections import deque
from itertools import product
import datetime, random, math, os, tensorflow as tf, numpy as np, matplotlib.pyplot as plt

SESSION_NAME = "mySession - " + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
hyperParameterList = {  # set hyperparameter combinations
    "Episodes": [1000],
    "Episode Length": [350],
    "Crash Penalty": [-2000],
    "Goal Reward": [200, 200],
    "Reward Direction": [True],
    # if true, reward is given in proportion to angle towards goal, if false, reward is given if bot 'stepped' closer
    "Epsilon Initial": [1],
    "Epsilon Decay": [.992],
    "Epsilon Min": [.05],
    "Reset Target": [2000],  # memories examined count to sync target net to model net
    "Gamma": [.99],
    "Scan Ratio": [12],  # how wany of the 360 scans are read, larger number = less scan count
    "Max Scan Range": [1],  # how far each scan ray sees, max 3.5
    "Scan Reward Scaler": [1],
    "Learning Rate": [0.0002],
    "Optimizer": [tf.compat.v1.keras.optimizers.RMSprop],
    "Loss": [tf.compat.v1.losses.huber_loss],
    "Batch Size": [100],  # memories examined per step
    "Memory Length": [1000000],
    "Direction Scalar": [1],  # changes value of reward for non-terminal steps
    "First Activation": [tf.compat.v1.keras.activations.relu],
    "Hidden Activations": [tf.compat.v1.keras.activations.relu],
    "Last Activation": [tf.compat.v1.keras.activations.linear],
    "Initializer": [tf.variance_scaling_initializer(scale=2)],
    "Load Model": [False]
}
FINAL_SLICE = 50
FINAL_SLICE = (hyperParameterList["Episodes"] // 2 if FINAL_SLICE >= hyperParameterList[
    "Episodes"] else FINAL_SLICE)  # set and save session vars
hyperParameterList["State Space"] = [360 // r + 3 for r in hyperParameterList["Scan Ratio"]]  # 360/12 + 3 = 33

keys, values = zip(*hyperParameterList.items())
maxAvg, maxFinalAvg = -1000000, -1000000
bestParams, bestFinalParams = {}, {}

for v in product(*values):
    experiment = dict(zip(keys, v))  # generate hyperparameter combination
    print("Using hyperparams: " + str(experiment))
