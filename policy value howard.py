import numpy as np
import argparse
import pulp

def policy_evaluation(mdp_trans, discount, action_space, numStates, numActions, end, policy):
    #creating the arrays for the curr_value function (to store the previous iteration) and value_func to store the final
    curr_val = np.zeros(numStates, dtype = np.float64)
    value_func = np.zeros(numStates, dtype = np.float64)
    while(True):
        curr_val = value_func.copy()
        for state in range(numStates):
            #if the state is a terminal state, then we cannot move any furthur thus, we continue
            #if state in end:
            #    continue
            val_current_iter = 0
            #action has to be taken according to the policy
            for s2, r, p in mdp_trans[state][policy[state]]:
                val_current_iter = val_current_iter + p*(r + discount*value_func[s2])     
            value_func[state] = val_current_iter
                    
        #break condition if the solution converges
        if np.linalg.norm(value_func-curr_val) <= 1e-10:
            # print("Total: ", i)
            return value_func
            
def value_iteration(mdp_trans, discount, action_space, numStates, numActions, end):
    #np.float32 gives wrong answers maybe due to decimal accuracy
    value_func = np.zeros(numStates, dtype = np.float64)
    policy = np.zeros(numStates, dtype = np.float64)
    curr_val = np.zeros(numStates, dtype = np.float64)
    
    while True:
        #storing the previous value of the value function in curr_val and then we will calculate the new value
        curr_val = value_func.copy()
        for state in range(numStates):

            optimal_action = 0
            val_max = float('-inf')
            #for every possible action, we find the value and then compare them to find the maximum value
            for action in sorted(action_space[state]):
                val_current_iter = 0
                for s2, r, p in mdp_trans[state][action]:
                    val_current_iter = val_current_iter + p*(r + discount*curr_val[s2])
                    
                if val_current_iter > val_max:
                    val_max = val_current_iter
                    optimal_action = action
                
            if len(action_space[state]) == 0:
                val_max = 0
            
            policy[state] = optimal_action  
            value_func[state] = val_max
            
        #convergence criteria
        if np.linalg.norm(curr_val-value_func) <= 1e-10:
            return value_func, policy
        
        

def howard_policy_iteration(mdp_trans, gamma, action_space, numStates, numActions, end):
    policy = np.zeros(numStates, dtype = np.int32)
    i = 0  # TODO check this
    while True:
        #first we evaluate the current policy
        value_func = policy_evaluation(mdp_trans, gamma, action_space, numStates, numActions, end, policy)
        
        #now we create a new policy 
        new_policy = policy.copy()
        for state in range(numStates):
            #if we have a terminal state, we cannot move anywhere thus we have to continue
            if state in end:
                continue
            curr_Q = value_func[state] 
            max_Q = curr_Q
            optimal_action = policy[state]

            for action in sorted(action_space[state]):
                #initializing the state action value function
                Qsa = 0
                for s2, r, p in mdp_trans[state][action]:
                    Qsa += p*(r + gamma*value_func[s2])

                if(abs(Qsa - curr_Q) >= 1e-10 and Qsa > max_Q):
                    optimal_action = action
                    max_Q = Qsa
            new_policy[state] = optimal_action
            
        #if the new policy is same as the previous policy, then we have reached the optimal policy
        #othwerwise, update the current policy
        if np.array_equal(policy, new_policy):
            return value_func, policy
        else:
            policy = new_policy



def linear_programming(mdp_trans, action_space, gamma, numStates, numActions):
    #here I have used pulp to solve the linear program 
    prob = pulp.LpProblem("ValueFn", pulp.LpMinimize)
    
    #creating the decision variables V1, V2, V3, V4, V5..... and so on
    value_function = []
    for i in range(numStates):
        value_function.append(pulp.LpVariable('V'+str(i)))
    #this prob now begins collecting problem data. This has been reference from the tutorial video
    prob += pulp.lpSum(value_function)

    #now we create the objective function
    for state in range(numStates):
        if len(action_space[state]) == 0:
            if(value_function[state]>=0):
                #adding the constraints
                prob += value_function[state]>=0
            
        #if the action space for that particular state contains a non zero number of possible actions
        for action in action_space[state]:
            qsaprob = []
            for s2, r, p in mdp_trans[state][action]:
                qsaprob.append(p*(r + gamma*value_function[s2]))
            if len(mdp_trans[state][action]) > 0:
                #adding the constraints as taught in class
                prob += value_function[state] >= pulp.lpSum(qsaprob)

    optresult = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    optimal_value_function = [pulp.value(x) for x in value_function]

    #obtaining the optimal policy from the optimal value function
    policy = np.zeros(numStates, dtype = np.int32)
    for state in range(numStates):
        
        Qs = np.zeros(numActions, dtype = np.int32)
        for action in action_space[state]:
            Qs[action] = 0
            for s2, rew, prob in mdp_trans[state][action]:
                Qs[action] += prob * (rew + gamma*optimal_value_function[s2])
                
        policy[state] = np.argmax(Qs)
    return optimal_value_function, policy

#driver code
if __name__ == "__main__":
    #Initialize the parser
    parser = argparse.ArgumentParser()
    #taking in as input the algorithm to be used and the MDP file.
    parser.add_argument("--mdp", help = "specify the path to the input MDP (Markov Decision Process) file", required = True)
    parser.add_argument("--algorithm", help = "Specify the algorithm to be used from value iteration, Howard's policy iteration and Linear Programming",
                        required = False, choices = ["vi", "hpi", "lp"], default = "lp")
    parser.add_argument("--policy", help = "Enter the policy for which the value function has to be found", required = False)
    args = parser.parse_args()
    policy = args.policy
    
    #for processing the Markov Decision Process file
    lines = []
    with open(args.mdp, mode = "r") as f:
        lines = f.readlines()
    #print(lines)
    discount = 0
    action_space = {}   #set of all actions that can be taken
    for line in lines:
        x = line.split()
        #for the MDP file, if the first word is Transition, then it specifies the Transition probability
        #if the first word is numStates, then we get numStates and so on
        if(x[0] == "numStates"):
            numStates = int(x[1])
            for i in range(numStates):
                action_space[i] = set()   #initializing the set of all actions for the ith state
            
        elif(x[0] == "numActions"):
            numActions = int(x[1])
            #we create a nested list for all possible state action combinations but poupulate only the ones actually possbile
            mdp_trans = [[[] for action in range(numActions)] for state in range(numStates)]
            
        elif(x[0] == "transition"):
            s1 = int(x[1])
            ac = int(x[2])
            s2 = int(x[3])
            r = float(x[4])             #reward
            p = float(x[5])             #transition probability
            action_space[s1].add(ac)    #because we now know that we can take action ac in state s1
            mdp_trans[s1][ac].append((s2, r, p))
            #this line means that if we are in state s1, take action ac, then we land in s2, get reward r with transition prob = p

        elif(x[0] == "end"):
            end = list(map(int, x[1:]))
            #this list contains the final state for an episodic task
            
        elif(x[0] == "mdptype"):
            mdptype = x[1]
            
        elif(x[0] == "discount"):
            discount = float(x[1])   #gamma will be a fraction
    #if the policy is supplied, then we need to do policy evaluation
    if(policy != None):
        policy = []
        with open(args.policy, mode = "r") as p:
            lines = p.readlines()
            for line in lines:
                x = line.split()
                # print(x)
                policy.append(int(x[0]))
        #in this case, we already know the policy and we only need to find the value funciton for that policy
        value = policy_evaluation(mdp_trans, discount, action_space, numStates, numActions, end, policy)
    else:
        alg = args.algorithm
        if(alg == "vi"):
            value, policy = value_iteration(mdp_trans, discount, action_space, numStates, numActions, end)
        elif(alg == "hpi"):
            value, policy = howard_policy_iteration(mdp_trans, discount, action_space, numStates, numActions, end)
        elif(alg == "lp"):
            value, policy = linear_programming(mdp_trans, action_space, discount, numStates, numActions)
    #to print the optimal value function in the format specified in the problem statement
    for i in range(len(value)):
        print(f"{value[i]} {policy[i]}")
       
