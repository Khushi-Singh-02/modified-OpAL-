'''
import numpy as np

class dynaPRL_empirical:
    def __init__(self, res_table):  #, reward_manipulate=False):
        self.name = "dynaPRL from empirical"
        #self.reward_manipulate = reward_manipulate

        pleft = res_table['pleft']
        pright = res_table['pright']

        self.switch_tag = res_table['switch_tag']
        self.n_trials = len(pleft)
        self.reward_maps = np.array([pleft, pright]) / 10000
        print('res_table', res_table)

    def choice(self, t, c):
        reward_prob = self.reward_maps[t, :]
        h_side = np.argmax(reward_prob) + 1  # Python indexing starts from 0
        correct = c == h_side
        transition = correct

        # In this task, good_r = outcome
        good_r = np.random.rand() < reward_prob[c - 1]  # Python indexing starts from 0
        outcome = int(good_r)
        
        print('reward_prob', reward_prob)
        return transition, outcome, good_r, correct
'''
'''
        if self.reward_manipulate:
            if good_r:
                if np.random.rand() <= 0.2:
                    outcome = 2  # 20% of rewarded trials will have 2 times reward delivery
                else:
                    outcome = 1
            else:
                outcome = 0
        else:
            outcome = int(good_r)
'''

import numpy as np
import scipy.stats as stats
import random



class DynaPRL:
    def __init__(self, minblock_len=25, maxblock_len=40, block_rand=0.1, n_trials=1000, p_high=[0.45, 0.6, 0.8, 0.8], p_low=[0.45, 0.3, 0.1, 0.2], reward_manipulate=0):
        self.name = "dynaPRL"
        self.minblock_len = minblock_len
        self.maxblock_len = maxblock_len
        self.block_rand = block_rand
        self.n_trials = n_trials
        self.p_high = p_high
        self.p_low = p_low
        self.reward_manipulate = reward_manipulate

    def get_rewardprobs(self, probs):
        reward_map = [0, 0]
        b_type = self.rand_select(probs)
        high_side = self.rand_select([0.5, 0.5]) ############################# CHANGE BACK TO 0.5, 0.5!!!!!!!!!!!!!!!!!!!!!!!!
        reward_map[high_side] = self.p_high[b_type]
        reward_map[1 - high_side] = self.p_low[b_type]
        return reward_map, b_type

    def block_switch(self, last_b_type, n):
        if self.rand_select([0.35, 0.65]) == 0:
            # switch to block N
            # but if last block is also N
            # then force switch to other blocks
            if last_b_type == 0: 
                reward_map, b_type = self.get_rewardprobs([0, 0.5, 0.5, 0])
            else:
                reward_map, b_type = self.get_rewardprobs([1, 0, 0, 0])
        else:
            # switch to non N blocks
            reward_map, b_type = self.get_rewardprobs([0, 0.5, 0.5, 0])
            # see if new reward_map has the same hr side
        if reward_map.index(max(reward_map)) == self.reward_maps[n - 1].index(max(self.reward_maps[n - 1])):
                # the side the same then simply reverse and assign
            reward_map = reward_map[::-1]
        return reward_map, b_type

    def set_task(self, n_trials=None):
        if n_trials is not None:
            # override current n trials value
            self.n_trials = n_trials

        self.reward_maps = [[0, 0] for _ in range(self.n_trials)]
        reward_map, b_type = self.get_rewardprobs([0, 0.5, 0.5, 0])
        last_b_type = b_type
        blockprogress = 1
        warmup=random.random()
        for n in range(self.n_trials):
            if n<50:        ############################### n<0.2*self.n_trials ################ HARD CODED FOR NOW, BUT CAN ALSO USE...........
                if warmup>0.5:
                    self.reward_maps[n] = [0.8, 0.2]
                else:
                    self.reward_maps[n] = [0.2, 0.8]
                blockprogress=self.maxblock_len+1
                last_b_type = 3

            else:
                if blockprogress <= self.minblock_len:
                    self.reward_maps[n] = reward_map
                    blockprogress += 1
                    
                elif blockprogress >self.minblock_len and blockprogress <= self.maxblock_len:
                    if self.rand_select([1-self.block_rand, self.block_rand]) == 1:       
                        # switch
                        reward_map, b_type = self.block_switch(last_b_type, n)
                        self.reward_maps[n] = reward_map
                        last_b_type = b_type
                        blockprogress = 1
                    else:
                        # no switch
                        self.reward_maps[n] = reward_map
                        blockprogress += 1
                elif blockprogress > self.maxblock_len:
                    # forceSwitch
                    reward_map, b_type = self.block_switch(last_b_type, n)
                    self.reward_maps[n] = reward_map
                    last_b_type = b_type
                    blockprogress = 1

        self.switch_tag = [self.reward_maps[i][0] != self.reward_maps[i - 1][0] for i in range(1, len(self.reward_maps))] + [0]

    def choice(self, t, c):
        reward_prob = self.reward_maps[t]
        transition = reward_prob[c - 1] == max(reward_prob)
        good_r = random.random() < reward_prob[c - 1]  # 1 choose left and 2 choose right

        if self.reward_manipulate:
            if good_r:
                if random.random() <= 0.2:
                    outcome = 2  # 20% of rewarded trial will have 2 times reward delivery
                else:
                    outcome = 1
            else:
                outcome = 0
        else:
            outcome = int(good_r)

        correct_choice = reward_prob.index(max(reward_prob)) + 1
        correct = c == correct_choice
        return transition, outcome, good_r, correct
    
    @staticmethod
    def rand_select(probs):
        x= random.choices(range(len(probs)), weights=probs)[0]
        return x
    
