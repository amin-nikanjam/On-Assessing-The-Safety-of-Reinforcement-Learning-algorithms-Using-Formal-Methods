
import math
import os, sys, random, operator
import numpy as np
import statistics
import pandas as pd
import random
from csv import writer,reader
import csv
import matplotlib.pyplot as plt
from pysmt.shortcuts import Symbol, LE, GE, Int, And, Equals, Plus, Solver
from pysmt.typing import INT
from pysmt.shortcuts import Symbol, And, Not, is_sat, GE, Int, TRUE, Equals, Or, FALSE, is_valid, Times, LE, Bool


class Environment:


    def __init__(self, Ny=6, Nx=6):
        # Define state space
        self.Ny = Ny  # y grid size
        self.Nx = Nx  # x grid size
        self.state_dim = (Ny, Nx)
        self.state = (0, 0)
        self.stateA = (0, 0)
        # Define action space
        self.action_dim = (4,)  # up, right, down, left
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}

        self.action_coords = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # translations
        # Define rewards table
        self.R = self._build_rewards()  # R(s,a) agent rewards

        self.obs_dict = {0:(0,0),1:(0,1), 2:(0,2), 3:(0,3),4:(0,4),5:(0,5),
                          6:(1, 0),  7:(1, 1),  8:(1, 2),  9:(1, 3),  10:(1, 4),  11:(1, 5),
                          12:(2, 0), 13:(2, 1),  14:(2, 2),  15:(2, 3),  16:(2, 4),  17:(2, 5),
                          18:(3, 0),  19:(3, 1),  20:(3, 2),  21:(3, 3),  22:(3, 4),  23:(3, 5),
                          24:(4, 0),  25:(4, 1),  26:(4, 2),  27:(4, 3),  28:(4, 4),  29:(4, 5),
                          30:(5, 0),  31:(5, 1),  32:(5, 2),  33:(5, 3),  34:(5, 4),  35:(5, 5),
                         }
        self.rew_dict = {(0, 0):0.3, (0, 1):0.4, (0, 2):0.4, (0, 3):0.4, (0, 4):0.5, (0, 5):0.3,
                         (1, 0):0.4, (1, 1):0.5, (1, 2):0.5, (1, 3):0.4, (1, 4):0.5, (1, 5):0.4,
                         (2, 0):0.3, (2, 1):0.5, (2, 2):0.5, (2, 3):0.5, (2, 4):0.5, (2, 5):0.4,
                         (3, 0):0.3, (3, 1):0.4, (3, 2):0.5, (3, 3):0.5, (3, 4):0.5, (3, 5):0.4,
                         (4, 0):0.4, (4, 1):0.5, (4, 2):0.5, (4, 3):0.4, (4, 4):0.4, (4, 5):0.5,
                         (5, 0):0.3, (5, 1):0.3, (5, 2):0.4, (5, 3):0.5, (5, 4):0.5, (5, 5):0.1,
                         }
        self.obs_dictIn = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4, (0, 5): 5,
                           (1, 0): 6, (1, 1): 7, (1, 2): 8, (1, 3): 9, (1, 4): 10, (1, 5): 11,
                           (2, 0): 12, (2, 1): 13, (2, 2): 14, (2, 3): 15, (2, 4): 16, (2, 5): 17,
                           (3, 0): 18, (3, 1): 19, (3, 2): 20, (3, 3): 21, (3, 4): 22, (3, 5): 23,
                           (4, 0): 24, (4, 1): 25, (4, 2): 26, (4, 3): 27, (4, 4): 28, (4, 5): 29,
                           (5, 0): 30, (5, 1): 31, (5, 2): 32, (5, 3): 33, (5, 4): 34, (5, 5): 35,
                           }
        # Check action space consistency
        if len(self.action_dict.keys()) != len(self.action_coords):
            exit("err: inconsistent actions given")

    def reset(self):
        # Reset agent state to top-left grid corner
        self.state = (0, 0)
        return self.state

    def resetA(self):
        # Reset agent state to top-left grid corner
        self.stateA = (0, 0)
        return self.stateA

    def step(self, action, actionA):
        # Evolve agent state
        state_next = (self.state[0] + self.action_coords[action][0],
                      self.state[1] + self.action_coords[action][1])
        state_nextA = (self.stateA[0] + self.action_coords[actionA][0],
                       self.stateA[1] + self.action_coords[actionA][1])
        # Collect reward
        # reward = self.R[self.state + (action,)]-self.rew_dict[state_next]
        self.R = self._build_rewards()

        # Terminate if we reach bottom-right grid corner

        reward = self.R[self.state + (action,)]

        rewA = 0

        self.state = state_next
        self.stateA = state_nextA
        done = (state_next[0] == self.Ny - 1) and (state_next[1] == self.Nx - 1)
        return state_next, state_nextA, reward, done, rewA, self.stateA


    def allowed_actions(self):

        actions_allowed = []
        actions_allowedA=[]
        y, x = self.state[0], self.state[1]
        w,z=self.stateA[0], self.stateA[1]
        if (w > 0):
            actions_allowedA.append(self.action_dict["up"])
        if (w < self.Ny - 1):
            actions_allowedA.append(self.action_dict["down"])
        if (z > 0):
            actions_allowedA.append(self.action_dict["left"])
        if (z < self.Nx - 1):
            actions_allowedA.append(self.action_dict["right"])


        if (y > 0):
            actions_allowed.append(self.action_dict["up"])
        if (y < self.Ny - 1):
            actions_allowed.append(self.action_dict["down"])
        if (x > 0):
            actions_allowed.append(self.action_dict["left"])
        if (x < self.Nx - 1):
            actions_allowed.append(self.action_dict["right"])

        actions_allowed = np.array(actions_allowed, dtype=int)
        actions_allowedA = np.array(actions_allowedA, dtype=int)
        return actions_allowed,actions_allowedA

    def _build_rewards(self):

        r_goal = 100  # reward for arriving at terminal state (bottom-right corner)
        r_nongoal = -1  # penalty for not reaching terminal state



        R = r_nongoal * np.ones(self.state_dim + self.action_dim, dtype=float)  # R[s,a]

        for elt in range(4):
            if (self.state[0] + self.action_coords[elt][0] == self.stateA[0] and self.state[1] +
                    self.action_coords[elt][1] == self.stateA[1]):
                R[self.state[0], self.state[1], elt] = -100

        R[self.Ny - 2, self.Nx - 1, self.action_dict["down"]] = r_goal # arrive from above
        R[self.Ny - 1, self.Nx - 2, self.action_dict["right"]] = r_goal # arrive from the left


        return R


class Agent:

    def __init__(self, env):

        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.epsilon = 0.001
        self.epsilon_decay = 0.99
        self.beta = 0.01
        self.gamma = 0.6

        self.Q = np.zeros(self.state_dim + self.action_dim, dtype=float)
        self.QObs = np.zeros(self.state_dim + self.action_dim, dtype=float)
        self.QA = np.zeros(self.state_dim + self.action_dim, dtype=float)

    def get_action(self, env):

        x, y = env.allowed_actions()
        state = env.state
        stateA = env.stateA
        obs = 0
        ran = 0
        stateA = env.stateA
        if random.uniform(0, 1) < self.epsilon:

            actA = np.random.choice(y)
        else:


            actions_allowedA = y
            Q_sA = self.QA[stateA[0], stateA[1], actions_allowedA]
            actions_greedyA = actions_allowedA[np.flatnonzero(Q_sA == np.max(Q_sA))]
            actA = np.random.choice(actions_greedyA)

        if random.uniform(0, 1) < self.epsilon:
            # explore
            act = np.random.choice(x)

            ran = 1

        else:


            actions_allowed = x

            Q_s = self.Q[state[0], state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            act = np.random.choice(actions_greedy)

        if (state[0] + env.action_coords[act][0] == stateA[0] + env.action_coords[actA][0] and state[1] +
                env.action_coords[act][1] == stateA[1] + env.action_coords[actA][1] and ran == 0):
            saa = state + (act,)

            saaN = (state[0] + env.action_coords[act][0], state[1] + env.action_coords[act][1])
            saaAd = (stateA[0] + env.action_coords[actA][0], stateA[1] + env.action_coords[actA][1])
            self.QObs[saa] += self.beta * (
                    -100 + self.gamma * np.max(self.QObs[saaN]) - self.QObs[saa])

            obs = 1
        if obs == 1 and ran == 0:
            Q_s = self.QObs[state[0], state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            act = np.random.choice(actions_greedy)

        return act, actA, obs



    def train(self, memory):

        (state, action, state_next, reward, done,stateA,actionA,state_nextA,rewA) = memory
        sa = state + (action,)
        sad=stateA + (actionA,)


        self.Q[sa] += self.beta * (reward + self.gamma * np.max(self.Q[state_next]) - self.Q[sa])
        self.QA[sad] += self.beta * (rewA + self.gamma * np.max(self.QA[state_nextA]) - self.QA[sad])
        self.QObs[sa] += self.beta * (reward + self.gamma * np.max(self.QObs[state_next]) - self.QObs[sa])
        return state,stateA







# Settings
class Learn:
    def debut(self):
        for iteration in range(10):
                env = Environment(Ny=6, Nx=6)

                agent = Agent(env)
                action_coords = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                self.obs_dict = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (0, 4), 5: (0, 5),
                                 6: (1, 0), 7: (1, 1), 8: (1, 2), 9: (1, 3), 10: (1, 4), 11: (1, 5),
                                 12: (2, 0), 13: (2, 1), 14: (2, 2), 15: (2, 3), 16: (2, 4), 17: (2, 5),
                                 18: (3, 0), 19: (3, 1), 20: (3, 2), 21: (3, 3), 22: (3, 4), 23: (3, 5),
                                 24: (4, 0), 25: (4, 1), 26: (4, 2), 27: (4, 3), 28: (4, 4), 29: (4, 5),
                                 30: (5, 0), 31: (5, 1), 32: (5, 2), 33: (5, 3), 34: (5, 4), 35: (5, 5),
                                 }
                self.obs_dictIn = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4, (0, 5): 5,
                                   (1, 0): 6, (1, 1): 7, (1, 2): 8, (1, 3): 9, (1, 4): 10, (1, 5): 11,
                                   (2, 0): 12, (2, 1): 13, (2, 2): 14, (2, 3): 15, (2, 4): 16, (2, 5): 17,
                                   (3, 0): 18, (3, 1): 19, (3, 2): 20, (3, 3): 21, (3, 4): 22, (3, 5): 23,
                                   (4, 0): 24, (4, 1): 25, (4, 2): 26, (4, 3): 27, (4, 4): 28, (4, 5): 29,
                                   (5, 0): 30, (5, 1): 31, (5, 2): 32, (5, 3): 33, (5, 4): 34, (5, 5): 35,
                                   }




                # Open our existing CSV file in append mode
                # Create a file object for this file
                file = open('statProb'+str(iteration)+'.csv', 'w', newline='')
                with file:
                    # identifying header
                    header = ['episode Number','Cumulative reward per episode', 'collision per episode','Cumulative reward per episode of adv','fin','occPr']
                    writer = csv.DictWriter(file, fieldnames=header)

                    # writing data row-wise into the csv file
                    writer.writeheader()

                N_episodes = 2100
                A = []
                E = []
                R = []

                ES = []
                cumR = []
                cumRA = []
                num = 0
                nbColl = []

                nj=0
                h=0


                InitAPos=(5,4)
                self.MylistofProb = np.zeros((6, 6))
                self.MylistofColl = np.zeros((6, 6))
                self.MylistofNonColl = np.zeros((6, 6))

                listAdv = [(5, 4), (4, 4), (4, 5)]
                advPos = listAdv[0]
                cumReward=0
                cumRewardA = 0
                DepOccur = np.zeros([36, 4, 2])
                DepOccurA = np.zeros([36, 4, 2])
                for episode in range(N_episodes):

                    # Generate an episode

                    x=episode%50000
                    ob = []


                    #if x==0:
                        #print(episode,j)
                        #A.append(j)
                    iter_episode, reward_episode = 0, 0
                    reward_episodeA = 0
                    state = env.reset()  # starting state

                    env.R = env._build_rewards()
                    stateA = env.resetA()


                    E.append(episode + num)
                    ES.append(1)
                    j=0
                    fin=0

                    while True:

                        #for elt in range(4):

                            #if (state[0]+action_coords[elt][0]==stateA[0] and state[1]+action_coords[elt][1]==stateA[1]):
                                #env.R[state[0], state[1], elt] = -100

                        action, actionA,obsr = agent.get_action(env)
                        #A.append(actionA)

                        state_next, state_nextA, reward, done, rewA, stateAdv = env.step(action, actionA)
                        if episode>2000:
                            DepOccur[self.obs_dictIn[state], action, 0] = DepOccur[self.obs_dictIn[state], action, 0]+1
                        state = state_next
                        stateAprec = stateA
                        stateA = state_nextA
                        if episode > 2000:
                            if obsr == 1:
                                DepOccurA[self.obs_dictIn[stateAprec], actionA, 0] += 1
                        if state == stateA and state!=(0,0):

                            reward = -100

                            rewA = 100


                        potential = -reward * (1 - self.MylistofProb[state[0]][state[1]]) + \
                                    agent.gamma * reward * (1 - self.MylistofProb[state_next[0]][
                            state_next[1]])
                        #potential = -agent.gamma * math.sqrt(
                            #math.pow(state_next[0] - 5, 2) + math.pow(state_next[1] - 5, 2)) + math.sqrt(
                            #math.pow(state[0] - 5, 2) + math.pow(state[1] - 5, 2))
                        reward = reward + potential
                        agent.train((state, action, state_next, reward, done, stateA, actionA, state_nextA, rewA))

                        if state == stateA and state != (0, 0):


                            j=j+1

                            self.MylistofColl[state[0]][state[1]]+=1

                            for z in range(6):
                                for w in range(6):

                                    if self.MylistofColl[z][w] == 0 and self.MylistofNonColl[z][w] == 0:
                                        self.MylistofProb[z][w] = 0

                                    elif self.MylistofColl[z][w] > self.MylistofNonColl[z][w]:

                                        self.MylistofProb[z][w] = 1

                                    elif self.MylistofNonColl[z][w] == 0:

                                        self.MylistofProb[z][w] = 1

                                    elif self.MylistofColl[z][w] < self.MylistofNonColl[z][w]:

                                        self.MylistofProb[z][w] = self.MylistofColl[z][w] / self.MylistofNonColl[z][w]

                                    else:

                                        self.MylistofProb[z][w] = self.MylistofColl[z][w] / self.MylistofNonColl[z][w]








                        else:
                            self.MylistofNonColl[state[0]][state[1]] += 1
                            for z in range(6):
                                for w in range(6):

                                    if self.MylistofColl[z][w] == 0 and self.MylistofNonColl[z][w] == 0:
                                        self.MylistofProb[z][w] = 0

                                    elif self.MylistofColl[z][w] > self.MylistofNonColl[z][w]:

                                        self.MylistofProb[z][w] = 1

                                    elif self.MylistofNonColl[z][w] == 0:

                                        self.MylistofProb[z][w] = 1

                                    elif self.MylistofColl[z][w] < self.MylistofNonColl[z][w]:

                                        self.MylistofProb[z][w] = self.MylistofColl[z][w] / self.MylistofNonColl[z][w]

                                    else:

                                        self.MylistofProb[z][w] = self.MylistofColl[z][w] / self.MylistofNonColl[z][w]

                            #print(episode,self.MylistofProb)
                            nj = nj + 1



                        # train agent
                        # print(state_next, reward)
                        #print(self.MylistofNonColl,'coll',self.MylistofColl)
                        iter_episode += 1

                        reward_episode += reward
                        cumReward += reward
                        cumRewardA += rewA
                        reward_episodeA += rewA
                        # train agent


                        #R.append(rewA)
                        # R.append(reward_episodeA)

                        if state_next == state_nextA:
                            break
                        if (iter_episode > 100):
                            break
                        if done:
                            fin=1
                            break
                        #if state_nextA == state_next:

                            # env.state=(0,0)



                        state = state_next

                        stateA = state_nextA

                    Lst= [episode,reward_episode, int(j), reward_episodeA,fin]


                    file = open('statProb'+str(iteration)+'.csv', 'a+', newline='')

                    # writing the data into the file
                    with file:
                        write = csv.writer(file)
                        write.writerow(Lst)


                    h += 1
                    if h<3:
                        advPos= listAdv[h]

                    if h==3:
                        temp=listAdv[0]
                        listAdv[0]=listAdv[2]
                        listAdv[2]=temp
                        h=0
                    if agent.epsilon >= 0.0001:
                        agent.epsilon *= agent.epsilon_decay
                    cumRA.append(reward_episodeA)
                    cumR.append(reward_episode)

                for count1 in range(36):
                        for count2 in range(4):
                            Total = DepOccur[count1, 0, 0] + DepOccur[count1, 1, 0] + DepOccur[count1, 2, 0] + DepOccur[
                                count1, 3, 0]
                            if Total != 0:
                                DepOccur[count1, count2, 1] = DepOccur[count1, count2, 0] / Total
                for count3 in range(36):
                        TotalA = DepOccurA[count3, 0, 0] + DepOccurA[count3, 1, 0] + DepOccurA[count3, 2, 0] + DepOccurA[
                            count3, 3, 0]
                        for count4 in range(4):

                            if TotalA != 0:
                                DepOccurA[count3, count4, 1] = DepOccurA[count3, count4, 0] / TotalA

                num += N_episodes
                df = pd.read_csv('statProb'+str(iteration)+'.csv')
                Dictt={}
                DicttA={}
                for count5 in range(36):
                    D = []
                    DA=[]
                    for count6 in range(4):
                        D.append(DepOccur[count5,count6,0])
                        DA.append(DepOccurA[count5, count6, 0])
                    Dictt[count5]=D
                    DicttA[count5]=DA
                df1=pd.DataFrame(Dictt)
                df2=pd.DataFrame(DicttA)
                result = pd.concat([df, df1,df2], axis=1)
                result.to_csv('statProb'+str(iteration)+'.csv')




Start=Learn()
Start.debut()

