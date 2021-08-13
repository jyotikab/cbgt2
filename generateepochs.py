import numpy as np
import random


def define_reward(opt_p, n_trials=100, reward_mu=1, reward_std=0):

    trial_index = np.arange(n_trials)
    
    #define suboptimal choice reward probability
    subopt_p = 1 - opt_p

    #sample rewards
    reward_values = np.random.normal(loc=reward_mu, scale=reward_std, size=n_trials)

    #calculate n_trials based on probabilities 
    n_opt_reward_trials = int(opt_p * n_trials)
    n_subopt_reward_trials = int(subopt_p * n_trials)

    #find indices for optimal and suboptimal choices 
    opt_reward_idx = np.random.choice(trial_index, size=n_opt_reward_trials, replace=False)
    subopt_reward_idx = np.setxor1d(trial_index, opt_reward_idx) #return the sorted, unique values that are in only one (not both) of the input arrays
    
    #intialize reward vectors
    reward_t1, reward_t2 = np.zeros((n_trials)),np.zeros((n_trials))

    #assign rewards
    reward_t1[opt_reward_idx] = reward_values[opt_reward_idx] 
    reward_t2[subopt_reward_idx] = reward_values[subopt_reward_idx]
    
    return reward_t1, reward_t2
    


def define_changepoints(n_trials, reward_t1, reward_t2, cp_lambda):

    
    #what is cp_lambda? - frequency of changing points 

    n_cps = int(n_trials/cp_lambda) #find approximate number of change points
    cp_base = np.cumsum(np.random.poisson(lam=cp_lambda,size=n_cps)) #calculate cp indices
    #cumsum - return the cumulative sum of the elements along a given axis 
    
    cp_idx = np.insert(cp_base,0,0) #add 0
    cp_idx = np.append(cp_idx,n_trials-1) #add 0
    
    cp_idx = cp_idx[cp_idx < n_trials] 
    
    #to remove possible equal elements 
    cp_idx = list(set(cp_idx))
    
    cp_indicator = np.zeros(n_trials)
    cp_indicator[cp_idx] = 1
    
    return cp_idx, cp_indicator 


def define_epochs(n_trials, reward_t1, reward_t2, cp_idx, opt_p):  
    
    t1_epochs = []
    t2_epochs = []
    
    
    epoch_number = []
    epoch_trial = []
    epoch_length = []
    
    reward_p = []
    
    volatile_pattern = []
    
    subopt_p = 1 - opt_p
    
    #remove or not? not needed for the moment 
    p_id_solution = [] #female greeble is always first 
    f_greeble = ord('f') #returns an integer representing the unicode character 
    m_greeble = ord('m')

    current_target = True
    #treat all the changepoints except for the last one 
    for i in range(len(cp_idx)-1):
        if current_target:
            volatile_pattern.append(np.repeat(0., cp_idx[i+1]-cp_idx[i]))
            t1_epochs.append(reward_t1[cp_idx[i]:cp_idx[i+1]])
            t2_epochs.append(reward_t2[cp_idx[i]:cp_idx[i+1]])
            #reward_p.append(np.repeat(opt_p, cp_idx[i+1]-cp_idx[i]))
            #p_id_solution.append(np.repeat(f_greeble, cp_idx[i+1]-cp_idx[i])) 
        else: 
            volatile_pattern.append(np.repeat(1., cp_idx[i+1]-cp_idx[i]))
            t1_epochs.append(reward_t2[cp_idx[i]:cp_idx[i+1]])
            t2_epochs.append(reward_t1[cp_idx[i]:cp_idx[i+1]])
            #reward_p.append(np.repeat(subopt_p, cp_idx[i+1]-cp_idx[i]))
            #p_id_solution.append(np.repeat(m_greeble, cp_idx[i+1]-cp_idx[i]))
        
        #epoch_number.append(np.repeat(i, cp_idx[i+1]-cp_idx[i]))
        
        current_target = not(current_target)
        
        #consider the last changepoint 
        if i == len(cp_idx)-2:
            if current_target:
                volatile_pattern.append(np.repeat(0., cp_idx[i+1]-cp_idx[i]))
                t1_epochs.append(reward_t1[cp_idx[i+1]:])
                t2_epochs.append(reward_t2[cp_idx[i+1]:])
                #reward_p.append(opt_p)
                #p_id_solution.append(f_greeble)
            else:
                volatile_pattern.append(np.repeat(1., cp_idx[i+1]-cp_idx[i]))
                t1_epochs.append(reward_t2[cp_idx[i+1]:])
                t2_epochs.append(reward_t1[cp_idx[i+1]:])
                #reward_p.append(subopt_p)
                #p_id_solution.append(m_greeble)
    
            #epoch_number.append(i+1)

    #save flaten arrays 
    #epoch_number = np.hstack(epoch_number).astype('float')
    t1_epochs = np.hstack(t1_epochs)
    t2_epochs = np.hstack(t2_epochs)
    #reward_p = np.hstack(reward_p).astype('float')
    #p_id_solution = np.hstack(p_id_solution)
    volatile_pattern = np.hstack(volatile_pattern)
    noisy_pattern = [min([.00001,abs(x)])*100000 for x in t1_epochs]
    #volatile_pattern = [x%2 for x in epoch_number] - if we need to compute epoch_number 
    
    return t1_epochs, t2_epochs, noisy_pattern, volatile_pattern #, epoch_number, reward_p, p_id_solution
    
