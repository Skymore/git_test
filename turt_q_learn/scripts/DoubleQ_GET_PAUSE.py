#!/usr/bin/env python
# -*- coding:utf8 -*-
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState, SetModelStateRequest, GetWorldProperties
from collections import deque
from itertools import product
import rospy, datetime, random, math, os, tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from std_srvs.srv import Empty
import time

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

crashDistances = {  # wall collision distance per model
    "turtlebot3_burger": .15,
    "turtlebot3_burger_front": .2,
    "turtlebot3_waffle": .22,
    "turtlebot3_waffle_front": .22,
}
IF_PRINT_TIME = False

ACTION_SPACE = [1.2, 0.6, 0, -0.6, -1.2]  # angular velocities for bot
TURTLEBOT_NAME = "turtlebot3_burger"
SCAN_MIN_DISTANCE = crashDistances[TURTLEBOT_NAME]  # distance to detect wall collision
SCAN_MIN_DISTANCE_FRONT = crashDistances[TURTLEBOT_NAME + "_front"]
GOAL_MIN_DISTANCE = .35  # distance to detect goal arrival
MODEL_SPEED = .12  # linear speed of model
FINAL_SLICE = 50  # number of scores considered for final average
ROLLING_AVERAGE_SAMPLES = 50  # number of samples used for rolling average graph

class modelClass():

    def __init__(
            self,
            hyperParams,
            env,
            saveName="myModel - "
    ):  # build models, set required parameters
        self.env = env
        self.turt_q_learn_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        if hyperParams["Load Model"]:
            self.model = tf.keras.models.load_model(self.turt_q_learn_path + "/load_model/model")
        else:
            self.model = genModel(hyperParams["Optimizer"], hyperParams["Loss"], hyperParams["Learning Rate"],
                                  hyperParams["First Activation"], hyperParams["Hidden Activations"],
                                  hyperParams["Last Activation"], hyperParams["State Dim"], hyperParams["Action Dim"],
                                  hyperParams["Initializer"])
        self.targetModel = tf.keras.models.clone_model(self.model)
        self.targetModel.set_weights(self.model.get_weights())
        self.doubleQNetwork = hyperParams["Double Q Network"]
        self.episodes = hyperParams["Episodes"]
        self.episodeLength = hyperParams["Episode Length"]
        self.epsilon = hyperParams["Epsilon Initial"]
        self.epsilonMin = hyperParams["Epsilon Min"]
        self.epsilonDecay = hyperParams["Epsilon Decay"]
        self.gamma = hyperParams["Gamma"]
        self.memory = deque(maxlen=hyperParams["Memory Length"])
        self.batchSize = hyperParams["Batch Size"]
        self.stateDim = hyperParams["State Dim"]
        self.actionDim  = hyperParams["Action Dim"]
        self.resetTargetMemories = hyperParams["Reset Target"]
        self.turt_q_learn_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.saveName = saveName + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.paramsString = dictToStr(hyperParams)
        self.memoryCount = 0

    def getAction(self,
                  state):  # generate random value, if greater than epsilon, choose action according to model, else, choose action randomly
        return (np.argmax(self.model.predict(np.expand_dims(state,axis=0))) if random.random() > self.epsilon else random.randint(0, 4))

    def playEpisodes(self):  # complete set amount of episodes, return numpy array of scores from each
        scores = []
        success = []
        totalStep = 0
        for epNum in range(self.episodes):
            print("Episode {0}:".format(epNum))
            score = 0
            doneWithGoal = False
            lastAction = -1
            lastState = -1
            state = self.env.resetState(doneWithGoal)
            print("Reset")
            actionRate = rospy.Rate(1)
            #while not rospy.is_shutdown():
            for stepNum in range(
                    self.episodeLength):  # generate action, get new state, create memory, perform dqn if memory is large enough
                action = self.getAction(state)
                totalStep += 1

                actionRate.sleep()
                #if (stepNum > 0):
                #    print("Act time: {}".format(actEndTime-actBgnTime))
                state, reward, done ,doneWithGoal = self.env.step(action)

                # newState实际为执行当前action时的State
                # 所以存储经验应该为: lastState lastAction state reward done
                if (stepNum > 0): # 第一次执行动作时不存储经验
                    self.memory.append([lastState, lastAction, state, reward, done])
                lastState = state
                lastAction = action
                score += reward
                if len(self.memory) > 2 * self.batchSize:
                    self.train()
                if (rospy.is_shutdown()):
                    print("ERROR: rospy is shutdown ")
                if done:
                    break

            if self.epsilon > self.epsilonMin:
                self.epsilon *= self.epsilonDecay

            scores.append(score)
            success.append(int(doneWithGoal))
            print("	Total Step: {}".format(totalStep))
            print("	Score: {}".format(score))

        folderPath = "/dqnmodels/{}".format(SESSION_NAME) + "/"
        if not os.path.exists(self.turt_q_learn_path + folderPath + self.saveName):
            os.makedirs(self.turt_q_learn_path + folderPath + self.saveName)
        self.model.save(self.turt_q_learn_path + folderPath + self.saveName + "/model")

        plt.figure(2)
        averages = movingAverage(scores)
        success_rate = movingAverage(success) * 100

        plt.subplot(211)
        plt.plot(scores, label="Scores")
        plt.plot(np.arange(ROLLING_AVERAGE_SAMPLES - 1, self.episodes), averages, label="Average")
        plt.ylabel("Scores")

        plt.subplot(212)
        plt.plot(np.arange(ROLLING_AVERAGE_SAMPLES - 1, self.episodes), success_rate)#, label="Success Rate")
        plt.ylabel("Success Rate")
        plt.xlabel("Episodes")
        plt.legend()
        plt.savefig(self.turt_q_learn_path + folderPath + self.saveName + '/plot.png', bbox_inches='tight')
        plt.close()

        with open(self.turt_q_learn_path + folderPath + self.saveName + '/params.txt', "w") as f:
            f.write(self.paramsString)

        return np.array(scores), averages, success_rate, self.saveName

    def get_batch(self):
        mini_batch = random.sample(self.memory, self.batchSize)
        X_batch = np.empty((0, self.stateDim), dtype=np.float64)
        Y_batch = np.empty((0, self.actionDim), dtype=np.float64)

        for mem in mini_batch:  # get original predictions, get q value of next state, and update original
            # predictions, orig state = x, updated preds = y
            q = self.model.predict(np.expand_dims(mem[0], axis=0))  # get prediction from state

            if mem[4]:
                qn = mem[3]
            else:
                qNextMain = self.model.predict(np.expand_dims(mem[2], axis=0))
                qNextTarget = self.targetModel.predict(np.expand_dims(mem[2], axis=0))  # 用target网络预测
                maxAction = np.argmax(qNextMain[0])
                maxQNext = qNextTarget[0][maxAction]
                qn = mem[3] + self.gamma * maxQNext
            q[0][mem[1]] = qn  # replace predicted q values with calculated value for action taken

            X_batch = np.concatenate((X_batch, np.expand_dims(mem[0],axis=0)),axis=0)  # append state to X values
            Y_batch = np.concatenate((Y_batch, q),axis=0)  # append updated predictions to Y values

        return X_batch, Y_batch

    def get_batch_fast(self):
        trans_batch = random.sample(self.memory, self.batchSize)
        trans_batch = np.asarray(trans_batch)
        # (state, action, next_state, reward, done)
        state_batch = trans_batch[:, 0:self.stateDim]
        action_batch = np.array(trans_batch[:, self.stateDim], dtype=int)
        nState_batch = trans_batch[:, self.stateDim + 1:2 * self.stateDim + 1]
        reward_batch = trans_batch[:, 2 * self.stateDim + 1]
        done_batch = np.array(trans_batch[:, 2*self.stateDim+2], dtype=bool)
        q_batch = self.model.predict(state_batch)
        q_next_main_batch = self.model.predict(nState_batch)
        q_next_target_batch = self.targetModel.predict(nState_batch)
        act_max_batch = np.argmax(q_next_main_batch, axis=1)
        # a_onehot = int(np.eye(self.batchSize, self.actionDim)[act_max_batch])
        q_max_next_batch = q_next_target_batch[np.arange(self.batchSize), act_max_batch]

        for i in range(self.batchSize):
            if done_batch[i]:
                qn = reward_batch[i]
            else:
                qn = reward_batch[i] + q_max_next_batch[i] * self.gamma
            q_batch[i][action_batch[i]] = qn
        return state_batch, q_batch

    def train(self):
        #t1 = time.time()

        X_batch, Y_batch = self.get_batch()
        #print(X_batch.shape, Y_batch.shape)
        #t2 = time.time()
        #print("Bat time: {0}".format(t2-t1))
        self.model.fit(X_batch, Y_batch, verbose=0)
        #t3 = time.time()
        #print("Fit time: {0}".format(t3-t2))

        self.memoryCount += 1
        if self.memoryCount >= self.resetTargetMemories:
            self.targetModel.set_weights(self.model.get_weights())  # update target model to current model
            self.memoryCount = 0
            print("reset weights")

class envWrapper():

    def __init__(self, params):  # Set required parameters
        self.actionPublisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.goalX, self.goalY = 0, 0
        self.goalDistanceOld = None
        self.rewardDirection = params["Reward Direction"]
        self.scanRatio = params["Scan Ratio"]
        self.crashPenalty = params["Crash Penalty"]
        self.goalReward = params["Goal Reward"]
        self.directionScalar = params["Direction Scalar"]
        self.maxScanRange = params["Max Scan Range"]
        self.scanRewardScalar = params["Scan Reward Scaler"]
        self.usePause = params["Pause"]
        self.modelX, self.modelY = None, None
        self.stateDim = params["State Dim"]
        self.actionDim = params["Action Dim"]
        print("stateDim: {}".format(self.stateDim))
    def getState(self, ranges):  # read laser ranges, get goal info
        # ranges = [range2State(r) for i, r in enumerate(ranges) if not i % self.scanRatio]
        ranges = [(self.maxScanRange if str(r) == 'inf' else min(r, self.maxScanRange))
                  for i, r in enumerate(ranges) if not i % self.scanRatio]
        goalInfo = self.getGoalOdomStateInfo()
        return np.asarray(ranges + goalInfo).reshape(self.stateDim,)

    def getGoalOdomStateInfo(self):  # get distance and angle to goal
        odomData = None
        while odomData is None:
            try:
                odomData = rospy.wait_for_message('odom', Odometry, timeout=5)
            except Exception as e:
                pass

        # get goal data
        self.modelX = odomData.pose.pose.position.x
        self.modelY = odomData.pose.pose.position.y

        goalDistance = math.hypot(self.goalX - self.modelX, self.goalY - self.modelY)
        goalAngle = math.atan2(self.goalY - self.modelY, self.goalX - self.modelX)
        modelAngle = odomData.pose.pose.orientation

        yaw = math.atan2(+2.0 * (modelAngle.w * modelAngle.z + modelAngle.x * modelAngle.y),
                         1.0 - 2.0 * (modelAngle.y * modelAngle.y + modelAngle.z * modelAngle.z))

        heading = goalAngle - yaw
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi

        # get odom data
        # modelZ = odomData.twist.twist.angular.z

        return [odomData.twist.twist.angular.z, goalDistance, heading]

    def _pause(self):
        # Pause Simulation
        if self.usePause:
            rospy.wait_for_service('/gazebo/pause_physics')
            try:
                rospy.ServiceProxy('/gazebo/pause_physics', Empty)()
            except rospy.ServiceException:
                print("/gazebo/pause_physics Service Call Faild !")

    def _unpause(self):
        if self.usePause:
            rospy.wait_for_service('/gazebo/unpause_physics')

            try:
                rospy.ServiceProxy('/gazebo/unpause_physics', Empty)()
            except rospy.ServiceException:
                print("/gazebo/unpause_physics Service Call Faild !")

    def step(self, action):  # publish action and read new state, reward, and done variables

        self._unpause()
        state = self.getState(self.getScan())
        self.actionPublisher.publish(genTwist(action))

        self._pause()
        reward, done, doneWithGoal = self.getReward(state)

        return state, reward, done, doneWithGoal

    def deleteGoal(self):
        goalDeleted = False
        while not goalDeleted:
            deleteModel("goal")
            goalDeleted = not checkModelPresent("goal")

    def spawnGoal(self):
        self.goalX, self.goalY = getGoalCoord(-1, 0)
        goalSpawned = False
        while not goalSpawned:
            spawnModel("goal", self.goalX, self.goalY)
            goalSpawned = checkModelPresent("goal")
        print("	New goal at {0}, {1}!".format(self.goalX, self.goalY))

    def getScan(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5).ranges
            except:
                pass
        return data

    def resetState(self,doneWithGoal):  # respawn turtle and goal models, get initial state
        print("	Start!")
        self.deleteGoal()
        # Spawn Goal
        self.spawnGoal()
        #if not doneWithGoal:
        x = [--1.5, 1.5, 0, 0]
        y = [0, 0, -1.5, 1.5]
        s = random.randint(0,3)
        self.teleportModel(TURTLEBOT_NAME, x[s],y[s])
        self._unpause()
        state = self.getState(self.getScan())
        self._pause()
        return state

    def getReward(self,
                  state):  # check if crash occured, if goal was reached, if distance to target increased / angle towards target, return reward + doneState accordingly

        # pdb.set_trace()
        if min(state[1:23 - 3]) < SCAN_MIN_DISTANCE or state[0] < SCAN_MIN_DISTANCE_FRONT:
            print("	Crashed!")
            return self.crashPenalty, True, False

        if state[self.stateDim - 2] < GOAL_MIN_DISTANCE:
            print(" Reached!")
            return self.goalReward, True, True

        if self.rewardDirection:  # reward based on heading
            return self.directionScalar * math.cos(state[self.stateDim - 1]), False, False
        else:  # reward based on current distance vs prev distance
            return self.directionScalar * (1 if state[self.stateDim - 2] < self.goalDistanceOld else -1), False, False

    def scanReward(self, state):
        return (1 - min(state[1:self.stateDim - 3]) / self.maxScanRange) * self.scanRewardScalar

    def teleportModel(self, modelName, x, y):  # used as workaround to reset turtlebot model
        rospy.wait_for_service('gazebo/set_model_state')
        apparate = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        smsReq = SetModelStateRequest()
        smsReq.model_state.model_name = modelName
        smsReq.model_state.pose.position.x = x
        smsReq.model_state.pose.position.y = y
        if modelName != "goal":
            self.actionPublisher.publish(Twist())  # stop current twist command
        apparate(smsReq)


def deleteModel(modelName):  # using delete on turtlebot3 crashes gazebo, check if goal exists, delete if present
    while True:
        rospy.wait_for_service('gazebo/get_world_properties')
        get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        resp = get_world_properties()
        if "goal" in resp.model_names:
            break
        else:
            return

    rospy.wait_for_service('gazebo/delete_model')
    del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
    del_model_prox(modelName)


def spawnModel(modelName, x, y):  # use service to spawn model
    rospy.wait_for_service('gazebo/spawn_sdf_model')
    spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
    modelPath = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))) + "/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_"
    if modelName == TURTLEBOT_NAME:
        modelPath = TURTLEBOT_NAME.split("_")[1]
    elif modelName == "goal":
        modelPath += "square/goal_box"
    else:
        raise ValueError("Required model name not available")
    with open(modelPath + '/model.sdf', 'r') as xml_file:
        model_xml = xml_file.read().replace('\n', '')
    spawnPose = Pose()
    spawnPose.position.x = x
    spawnPose.position.y = y
    spawn_model_prox(modelName, model_xml, '', spawnPose, "world")


def checkModelPresent(modelName):
    rospy.wait_for_service('gazebo/get_world_properties')
    get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
    resp = get_world_properties()
    return modelName in resp.model_names


def genTwist(index):  # build twist to be sent to gazebo
    retTwist = Twist()
    retTwist.linear.x = MODEL_SPEED
    retTwist.angular.z = ACTION_SPACE[index]
    return retTwist


def range2State(r):  # convert 'inf' to float value
    return (3.5 if str(r) == 'inf' else r)


def getGoalCoord(removeX=None, removeY=None):  # generate new coordinates for goal
    while True:
        x = random.choice([-1.5, -.5, .5, 1.5])
        y = random.choice([-1.5, -.5, .5, 1.5])
        if not (x == removeX and y == removeY):
            break
    return x, y


def genModel(optimizer, loss, lr, first, hidden, last, stateDim, actionDim, initializer):  # build neural net to be used for DDQN
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, input_shape=(stateDim,), activation=first,
                                              kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(32, activation=hidden, kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(16, activation=hidden, kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(actionDim, activation=last))
    model.compile(optimizer=optimizer(lr=lr), loss=loss)
    return model


def saveParams(path, dict):
    with open(path, "w") as f:
        f.write(dictToStr(dict))


def dictToStr(dict):
    str = ""
    for key, value in sorted(dict.iteritems()):
        str += "{0}: {1}\n".format(key, value)
    return str


def movingAverage(a):
    ret = np.cumsum(a, dtype=float)
    ret[ROLLING_AVERAGE_SAMPLES:] = ret[ROLLING_AVERAGE_SAMPLES:] - ret[:-ROLLING_AVERAGE_SAMPLES]
    return ret[ROLLING_AVERAGE_SAMPLES - 1:] / ROLLING_AVERAGE_SAMPLES


def cartProd(paramDict, fn, **kwargs):
    keys, values = zip(*paramDict.items())
    for v in product(*values):
        experiment = dict(zip(keys, v))  # generate hyperparameter combination
        print("Using hyperparams: " + str(experiment))
        fn(experiment)


if __name__ == '__main__':
    rospy.init_node('qLearner')
    try:
        SESSION_NAME = "mySession - " + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        hyperParameterList = {  # set hyperparameter combinations
            "Episodes": [100],
            "Episode Length": [350],
            "Crash Penalty": [-2000],
            "Goal Reward": [200],
            "Reward Direction": [True],
            # if true, reward is given in proportion to angle towards goal, if false, reward is given if bot 'stepped' closer
            "Epsilon Initial": [1],
            "Epsilon Decay": [.992],
            "Epsilon Min": [.05],
            "Reset Target": [500],  # memories examined count to sync target net to model net
            "Gamma": [.99], # 折扣因子
            "Scan Ratio": [18],  # how wany of the 360 scans are read, larger number = less scan count
            "Max Scan Range": [1],  # how far each scan ray sees, max 3.5
            "Scan Reward Scaler": [1],
            "Learning Rate": [0.0002],
            "Optimizer": [tf.keras.optimizers.RMSprop],
            "Loss": [tf.keras.losses.Huber()],
            "Batch Size": [64],  # memories examined per step
            "Memory Length": [1000000],
            "Direction Scalar": [1],  # changes value of reward for non-terminal steps
            "First Activation": [tf.keras.activations.relu],
            "Hidden Activations": [tf.keras.activations.relu],
            "Last Activation": [tf.keras.activations.linear],
            "Initializer": [tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None)],
            "Load Model": [False],
            "Double Q Network": [True],
            "Pause": [False]
        }

        FINAL_SLICE = (hyperParameterList["Episodes"] // 2 if FINAL_SLICE >= hyperParameterList[
            "Episodes"] else FINAL_SLICE)  # set and save session vars
        hyperParameterList["State Dim"] = [360 // r + 3 for r in hyperParameterList["Scan Ratio"]]  # 360/12 + 3 = 33
        hyperParameterList["Action Dim"] = [len(ACTION_SPACE)]


        folderPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/dqnmodels/{}".format(SESSION_NAME)

        if not os.path.exists(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/dqnmodels/" + SESSION_NAME):
            os.makedirs(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/dqnmodels/" + SESSION_NAME)

        saveParams(folderPath + '/allParams.txt', hyperParameterList)
        saveParams(folderPath + '/envVars.txt', {
            "Action Space": ACTION_SPACE,
            "Turtlebot Name": TURTLEBOT_NAME,
            "Goal Min Distance": GOAL_MIN_DISTANCE,
            "Scan Min Distance": SCAN_MIN_DISTANCE,
            "Model Speed": MODEL_SPEED,
            "Final Slice": FINAL_SLICE,
            "Rolling Average Samples": ROLLING_AVERAGE_SAMPLES
        })

        keys, values = zip(*hyperParameterList.items())
        maxAvg, maxFinalAvg = -1000000, -1000000
        bestParams, bestFinalParams = {}, {}
        experNum = 0

        for v in product(*values):
            experNum += 1
            experiment = dict(zip(keys, v))  # generate hyperparameter combination
            print("Using hyperparams: " + str(experiment))

            env = envWrapper(experiment)
            model = modelClass(experiment, env)
            hyperSetScores, averages, success_rate, setName = model.playEpisodes()

            if np.mean(hyperSetScores) > maxAvg:  # get most succesful combination
                maxAvg = np.mean(hyperSetScores)
                bestParams = experiment

            if np.mean(hyperSetScores[len(hyperSetScores) - FINAL_SLICE::]) > maxFinalAvg:  # over last x episodes
                maxFinalAvg = np.mean(hyperSetScores[len(hyperSetScores) - FINAL_SLICE::])
                bestFinalParams = experiment

            plt.figure(1)
            plt.subplot(311)  # throws incorrect warning, see https://github.com/matplotlib/matplotlib/issues/12513
            plt.plot(hyperSetScores, label=u"ex {0}".format(experNum))
            plt.subplot(312)
            plt.plot(np.arange(ROLLING_AVERAGE_SAMPLES - 1, experiment["Episodes"]), averages, label=u"ex {0}".format(experNum))
            plt.subplot(313)
            plt.plot(np.arange(ROLLING_AVERAGE_SAMPLES - 1, experiment["Episodes"]), success_rate, label=u"ex {0}".format(experNum))

        plt.subplot(311)
        plt.ylabel("Scores")
        # plt.xlabel("Episodes")
        plt.subplot(312)
        plt.ylabel("Average")
        plt.subplot(313)
        plt.ylabel("Sucess Rate")
        plt.xlabel("Episodes")
        plt.legend()
        plt.savefig(folderPath + '/plot.png', bbox_inches='tight')
        plt.close()

        saveParams(folderPath + '/bestParams.txt', bestParams)
        saveParams(folderPath + '/bestFinalParams.txt', bestFinalParams)
    except rospy.ROSInterruptException:
        pass