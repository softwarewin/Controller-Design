__author__ = 'jxlllx'
import numpy as np
import random
class QClass:
    def __init__(self, ActionNum,BasisNum,epsilon):
        self.actions = [i for i in range(ActionNum)]#离散动作空间编号
        self.rewards = 0
        self.theta = [0.0 for i in range(BasisNum*ActionNum)]
        self.theta = np.array(self.theta)
        self.theta = np.transpose(self.theta)
        self.epsilon = epsilon
        self.gamma = 0.8

    def PolynomialBasisCal22(self, States, Order):
        #(X1+X2..+Xm1+1)^n base function
        #X1^2+X2^2+X3^2+2X1X2+2X1X3+2X2X3
        StatesNum = len(States)
        BasisNum = (StatesNum+1)*Order
        BasisValue = [0.0 for i in range(BasisNum)]
        BasisValue[0] = States[0]**2
        BasisValue[1] = States[1]**2
        BasisValue[2] = 1
        BasisValue[3] = States[0]*States[1]
        BasisValue[4] = States[0]
        BasisValue[5] = States[1]
        return BasisValue

    #获取基于动作获取对应特征向量（关系到组合）
    def get_fea_vec(self, feature, action):
        f = np.array([0.0 for i in range(len(self.theta))])#特征向量清零
        idx = 0
        for i in range(len(self.actions)):
            if action == self.actions[i]: idx = i;
        for i in range(len(feature)):
            f[i + idx * len(feature)] = feature[i];#对应动作位置的特征向量赋值
        return f

    def qfunc(self, feature, action):
            f = self.get_fea_vec(feature, action);
            return np.dot(f, self.theta);

    def update(self, feature, action, tvalue, alpha):
        pvalue = self.qfunc(feature, action);
        error = pvalue - tvalue;
        fea = self.get_fea_vec(feature, action);
        self.theta -= alpha * error * fea;

    def epsilon_greedy(self, feature):
        epsilon = self.epsilon
    # select action with maximum Q(s,a)
        amax = 0
        qmax = self.qfunc(feature,self.actions[0])
        for i in range(len(self.actions)):
            a = self.actions[i]
            q = self.qfunc(feature, a)
            if qmax < q:
                qmax  = q
                amax  = i
    # define probability for each action
        pro = [0.0 for i in range(len(self.actions))]
        pro[amax] += 1- epsilon
        for i in range(len(self.actions)):
            pro[i] += epsilon / len(self.actions)
    # select random action
        r = random.random()
        s = 0.0
        for i in range(len(self.actions)):
            s += pro[i]
            if s >= r: return self.actions[i]
        return self.actions[len(self.actions)-1]