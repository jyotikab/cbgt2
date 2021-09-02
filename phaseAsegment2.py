
# generating reward schedule for conflict/volatility tasks
# important output variables
# - t1_epochs
# - t2_epochs
# - noisy_pattern
# - volatile_pattern


import numpy as np
import random
from frontendhelpers import *


def define_reward(opt_p, n_trials=400, reward_mu=3, reward_std=1):

    trial_index = np.arange(n_trials)

    # define suboptimal choice reward probability
    subopt_p = 1 - opt_p

    # sample rewards
    reward_values = np.random.normal(
        loc=reward_mu, scale=reward_std, size=n_trials)

    # calcualte n_trials based on probabilities
    n_opt_reward_trials = int(opt_p * n_trials)
    n_subopt_reward_trials = int(subopt_p * n_trials)

    # find indices for optimal and suboptimal choices
    opt_reward_idx = np.random.choice(
        trial_index, size=n_opt_reward_trials, replace=False)
    subopt_reward_idx = np.setxor1d(trial_index, opt_reward_idx)

    # intialize reward vectors
    reward_t1, reward_t2 = np.zeros((n_trials)), np.zeros((n_trials))

    # assign rewards
    reward_t1[opt_reward_idx] = reward_values[opt_reward_idx]
    reward_t2[subopt_reward_idx] = reward_values[subopt_reward_idx]

    return reward_t1, reward_t2


def define_changepoints(n_trials, reward_t1, reward_t2, cp_lambda):

    # find approximate number of change points
    n_cps = int(n_trials / cp_lambda)
    cp_base = np.cumsum(
        np.random.poisson(
            lam=cp_lambda,
            size=n_cps))  # calculate cp indices

    cp_idx = np.insert(cp_base, 0, 0)  # add 0
    cp_idx = np.append(cp_idx, n_trials - 1)  # add 0

    cp_idx = cp_idx[cp_idx < n_trials]

    cp_indicator = np.zeros(n_trials)
    cp_indicator[cp_idx] = 1

    return cp_idx, cp_indicator


def define_epochs(n_trials, reward_t1, reward_t2, cp_idx, opt_p):

    t1_epochs = []
    t2_epochs = []

    subopt_p = 1 - opt_p

    epoch_number = []
    epoch_trial = []
    epoch_length = []

    reward_p = []

    p_id_solution = []  # female greeble is always first

    f_greeble = ord('f')
    m_greeble = ord('m')

    current_target = True
    for i in range(len(cp_idx) - 1):
        if current_target:
            t1_epochs.append(reward_t1[cp_idx[i]:cp_idx[i + 1]])
            t2_epochs.append(reward_t2[cp_idx[i]:cp_idx[i + 1]])
            reward_p.append(np.repeat(opt_p, cp_idx[i + 1] - cp_idx[i]))
            p_id_solution.append(
                np.repeat(f_greeble, cp_idx[i + 1] - cp_idx[i]))
        else:
            t1_epochs.append(reward_t2[cp_idx[i]:cp_idx[i + 1]])
            t2_epochs.append(reward_t1[cp_idx[i]:cp_idx[i + 1]])
            reward_p.append(np.repeat(subopt_p, cp_idx[i + 1] - cp_idx[i]))
            p_id_solution.append(
                np.repeat(m_greeble, cp_idx[i + 1] - cp_idx[i]))

        epoch_number.append(np.repeat(i, cp_idx[i + 1] - cp_idx[i]))
        epoch_trial.append(np.arange(cp_idx[i + 1] - cp_idx[i]))
        epoch_length.append(np.repeat(len(np.arange(
            cp_idx[i + 1] - cp_idx[i])), repeats=len(np.arange(cp_idx[i + 1] - cp_idx[i]))))

        if i == len(cp_idx) - 2:
            if current_target:
                t1_epochs.append(reward_t1[-1])
                t2_epochs.append(reward_t2[-1])
                reward_p.append(opt_p)
                p_id_solution.append(f_greeble)
            else:
                t1_epochs.append(reward_t2[-1])
                t2_epochs.append(reward_t1[-1])
                reward_p.append(opt_p)
                p_id_solution.append(m_greeble)

            epoch_number.append(i)

        current_target = not(current_target)

    epoch_length[-1] = epoch_length[-1] + 1
    # flatten
    epoch_number = np.hstack(epoch_number).astype('float')
    epoch_trial = np.hstack(epoch_trial).astype('float')
    epoch_length = np.hstack(epoch_length).astype('float')

    epoch_trial = np.append(epoch_trial, (epoch_trial[-1] + 1))
    epoch_length = np.append(epoch_length, epoch_length[-1])

    t1_epochs = np.hstack(t1_epochs)
    t2_epochs = np.hstack(t2_epochs)
    reward_p = np.hstack(reward_p).astype('float')
    reward_p[-1] = reward_p[-2]
    p_id_solution = np.hstack(p_id_solution)

    # Matthew: new code
    t1_epochs = np.divide(t1_epochs, 3)
    t2_epochs = np.divide(t2_epochs, 3)
    for i in range(0, len(t1_epochs)):
        if random.uniform(0, 1) > opt_p:
            temp = t1_epochs[i]
            t1_epochs[i] = t2_epochs[i]
            t2_epochs[i] = temp

    return t1_epochs, t2_epochs, epoch_number, reward_p, p_id_solution, epoch_trial, epoch_length


n_trials = 600
volatility = 30
conflict = 0.75


(rt1, rt2) = define_reward(1, n_trials)
(cp_idx, cp_indicator) = define_changepoints(n_trials, rt1, rt2, volatility)
(t1_epochs, t2_epochs, epoch_number, reward_p, p_id_solution, epoch_trial,
 epoch_length) = define_epochs(10, rt1, rt2, cp_idx, conflict)
noisy_pattern = [min([.00001, abs(x)]) * 100000 for x in t1_epochs]
volatile_pattern = [x % 2 for x in epoch_number]
