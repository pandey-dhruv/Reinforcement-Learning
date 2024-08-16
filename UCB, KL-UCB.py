"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# START EDITING HERE
# You can use this space to define any helper functions that you need
def KL(p, q):
	if p == 1.0:
		return p*np.log(p/q)
	elif p == 0.0:
		return (1-p)*np.log((1-p)/(1-q))
	else:
		return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
#binary seaech to solve the root
def solve_q(empirical_mean, RHS):
    start = empirical_mean
    end = 1.0
    while(start < end):
        middle = start + (end - start)/2
        kl_divergence = KL(empirical_mean, middle)
        if(abs(kl_divergence - RHS) < 10**(-2)):
            return middle
        elif(kl_divergence > RHS):
            end = middle
        elif(kl_divergence < RHS):
            start = middle

# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        #self.epirical_mean = np.zeros(num_arms)  #create an array with initial probability  = 0
        self.ucb_arm = np.zeros(num_arms, dtype = np.float32)
        self.counts = np.zeros(num_arms, dtype = np.int32)
        self.values = np.zeros(num_arms, dtype = np.float32)
        self.total_pulls = 0   #stores the total number of pulls till that instant of time
        #this basically defines arrays for counts, values and empirical means to store the values for each bandit arm
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        #this method should return the index that you want to pull
        #if all the arms have not been pulled yet, we have to pull each arm atleast once and then start the process
        if(self.total_pulls < self.num_arms):
            #means that no arm has been pulled till now, thus return any random arm(uniform distribution over the arms)
            return self.total_pulls
        for x in range(self.num_arms):
            #first we calculate the ucb value for each arm
            self.ucb_arm[x] = self.values[x] + np.sqrt(2*math.log(self.total_pulls)/self.counts[x])
        #once we have calculated the ucb for each arm, return the maximum
        return np.argmax(self.ucb_arm)
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.total_pulls = self.total_pulls + 1
        self.counts[arm_index] = self.counts[arm_index]+1   #increment number of pulls for the arm by 1
        n = self.counts[arm_index]
        value = self.values[arm_index]   #the previous emprirical mean
        new_value = ((n-1)*value + reward)*(1/n)    #new empirical mean after pulling the arms
        self.values[arm_index] = new_value

        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(self.num_arms, dtype = np.int32)         #counts of the number of pulls for each arm
        self.values = np.zeros(self.num_arms, dtype = np.float32)       #value of the empirical mean of each arm
        self.ucb_values = np.zeros(self.num_arms, dtype = np.float32)
        self.total_pulls = 0                                            #value of the total number of bandit pulls
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # first we need to solve the value of q for which the equation given in the slides holds
        while(self.total_pulls < self.num_arms):
            return self.total_pulls
        for i in range(self.num_arms):
            val = math.log(self.total_pulls)/self.counts[i]
            self.ucb_values[i] = solve_q(self.values[i], val)
        return np.argmax(self.ucb_values)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.total_pulls = self.total_pulls + 1
        self.counts[arm_index] = self.counts[arm_index]+1
        n = self.counts[arm_index]
        self.values[arm_index] = ((n-1)*self.values[arm_index] + reward)*(1/n)
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.success = np.zeros(self.num_arms, dtype = np.int32)
        self.failure = np.zeros(self.num_arms, dtype = np.int32)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        #for every arm we need to pull it once in our mind and then store tha value
        beta_arms = np.zeros(self.num_arms, dtype = np.float32)
        for x in range(self.num_arms):
            #now we sample in our mind
            beta_arms[x] = np.random.beta(self.success[x]+1, self.failure[x]+1)
        #now we actually sample the arm with the maxiumum value
        return np.argmax(beta_arms)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        #we just need to update the value in success and failure
        if(reward == 0):
            self.failure[arm_index] = self.failure[arm_index]+1
        elif(reward == 1):
            self.success[arm_index] = self.success[arm_index]+1
        # END EDITING HERE
