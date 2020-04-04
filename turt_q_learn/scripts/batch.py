# memory里存储的数据:
# -*- coding:utf8 -*-
from collections import deque
import numpy as np
import random

# self.memory.append([lastState, action, state, reward, done])
# lastState和state是(1,actionSpace)维度的'numpy.ndarray'类型
# action reward是int
# done是bool


class Agent()
    #...
    self.memory = deque(maxlen=int(1e6))
    self.gamma = 0.995
    self.stateSpace = 36
    self.actionSpace = 5
    self.batchSize = 64
    #...
    def get_batch(self):
        mini_batch = random.sample(self.memory, self.batchSize)
        X_batch = np.empty((0, self.stateSpace), dtype=np.float64)
        Y_batch = np.empty((0, self.actionSpace), dtype=np.float64)
        state_batch = mini_batch[0]

        for mem in mini_batch:  # get original predictions, get q value of next state, and update original predictions, orig state = x, updated preds = y
            q = self.model.predict(mem[0])  # get prediction from state

            if mem[4]:
                qn = mem[3]
            else:
                qNextMain = self.model.predict(mem[2])
                qNextTarget = self.targetModel.predict(mem[2]) #用target网络预测
                maxAction = np.argmax(qNextMain[0])
                maxQNext = qNextTarget[0][maxAction]
                qn = mem[3] + self.gamma * maxQNext
            q[0][mem[1]] = qn  # replace predicted q values with calculated value for action taken

            X_batch = np.append(X_batch, mem[0])  # append state to X values
            Y_batch = np.append(Y_batch, q)  # append updated predictions to Y values

            return X_batch, Y_batch