__author__ = 'jxlllx'
import numpy as np
from QClass import *
import gym

if __name__ == "__main__":
   # theta = [0.0 for i in 9]
    #print(theta)
    # initializtion
    qclass = QClass(ActionNum = 3,BasisNum = 6, epsilon = 0.05)
    # initial Q table
    # initial Q table MountainCar
    env = gym.make('MountainCar-v0')
    print(env.action_space)
    print(env.observation_space)
    env.reset()
    for i_episode in range(20000):
        observation = env.reset()
        print("Episode {}".format(i_episode))
        print('Qtheta',qclass.theta)
        #Choose A from S using policy derived from Q (e.g., -greedy)
        env.render()
        feature = qclass.PolynomialBasisCal22(observation, 2)
        action = qclass.epsilon_greedy(feature)
        #for t in range(10000):#SARSA
        t = 0;
        while(1):
            t = t + 1;
            env.render()
            #Take action A, observe R, S'
            observation1, reward, done, info = env.step(action)
            reward = (observation1[0]+0.5)*(observation1[0]+0.5)+observation1[1]*observation1[1]
            #Choose A' from S' using policy derived from Q (e.g., -greedy)
            # next action epsilon_greedy A
            if done:
                # calculate true value basing on SARSA evaluation
                tvalue = reward
                # update theta
                alpha = 0.01
                qclass.update(feature, action, tvalue, alpha)
                print("Episode finished after {} timesteps".format(t))
                break
            else:
                feature1 = qclass.PolynomialBasisCal22(observation1, 2)
                action1 = qclass.epsilon_greedy(feature1)
                #print('action',action)
                # calculate true value basing on SARSA evaluation
                tvalue = reward + qclass.gamma * qclass.qfunc(feature1, action1)
                #tvalue = reward
                # update theta
                alpha = 0.01
                qclass.update(feature, action, tvalue, alpha)
                # get current state S
                #print(observation1)
                observation = observation1
                feature = feature1
                action = action1

