from frontendhelpers import *
from tracetype import *
import copy
import pdb
import numpy as np
import scipy.stats as sp_st

def helper_cellparams(params=None):

    celldefaults = ParamSet('celldefaults', {'N': 75,
                                             'C': 0.5,
                                             'Taum': 20,
                                             'RestPot': -70,
                                             'ResetPot': -55,
                                             'Threshold': -50,
                                             'RestPot_ca': -85,
                                             'Alpha_ca': 0.5,
                                             'Tau_ca': 80,
                                             'Eff_ca': 0.0,
                                             'tauhm': 20,
                                             'tauhp': 100,
                                             'V_h': -60,
                                             'V_T': 120,
                                             'g_T': 0,
                                             'g_adr_max': 0,
                                             'Vadr_h': -100,
                                             'Vadr_s': 10,
                                             'ADRRevPot': -90,
                                             'g_k_max': 0,
                                             'Vk_h': -34,
                                             'Vk_s': 6.5,
                                             'tau_k_max': 8,
                                             'n_k': 0,
                                             'h': 1, })

    if params is not None:
        
        celldefaults = ModifyViaSelector(celldefaults, params)
        
    return celldefaults


def helper_popspecific(pops=dict()):

    popspecific = {'LIP': {'N': 204},
                   'FSI': {'C': 0.2, 'Taum': 10},
                   # should be 10 but was 20 due to bug
                   'GPeP': {'N': 750, 'g_T': 0.06, 'Taum': 20},
                   'STNE': {'N': 750, 'g_T': 0.06},
                   'LIPI': {'N': 186, 'C': 0.2, 'Taum': 10},
                   'Th': {'Taum': 27.78}}

    if pops is not None:
        for key in pops.keys():
            for item in pops[key].keys():
                popspecific[key][item] = pops[key][item]

    return popspecific


def helper_receptor(receps=None):

    receptordefaults = ParamSet('receptordefaults', {'Tau_AMPA': 2,
                                                     'RevPot_AMPA': 0,
                                                     'Tau_GABA': 5,
                                                     'RevPot_GABA': -70,
                                                     'Tau_NMDA': 100,
                                                     'RevPot_NMDA': 0, })

    if receps is not None:
        receptordefaults = ModifyViaSelector(receptordefaults, receps)

    return receptordefaults


def helper_basestim(base=dict()):

    basestim = {'FSI': {
        'FreqExt_AMPA': 3.6,
        'MeanExtEff_AMPA': 1.55,
        'MeanExtCon_AMPA': 800},
        'LIPI': {
        'FreqExt_AMPA': 1.05,
        'MeanExtEff_AMPA': 1.2,
        'MeanExtCon_AMPA': 640},
        'GPi': {
        'FreqExt_AMPA': 0.8,
        'MeanExtEff_AMPA': 5.9,
        'MeanExtCon_AMPA': 800},
        'STNE': {
        'FreqExt_AMPA': 4.45,
        'MeanExtEff_AMPA': 1.65,
        'MeanExtCon_AMPA': 800},
        'GPeP': {
        'FreqExt_AMPA': 4,
        'MeanExtEff_AMPA': 2,
        'MeanExtCon_AMPA': 800,
        'FreqExt_GABA': 2,
        'MeanExtEff_GABA': 2,
        'MeanExtCon_GABA': 2000},
        'D1STR': {
        'FreqExt_AMPA': 1.3,
        'MeanExtEff_AMPA': 4,
        'MeanExtCon_AMPA': 800},
        'D2STR': {
        'FreqExt_AMPA': 1.3,
        'MeanExtEff_AMPA': 4,
        'MeanExtCon_AMPA': 800},
        'LIP': {
        'FreqExt_AMPA': 2.2,
        'MeanExtEff_AMPA': 2,
        'MeanExtCon_AMPA': 800},
        'Th': {
        'FreqExt_AMPA': 2.2,
        'MeanExtEff_AMPA': 2.5,
        'MeanExtCon_AMPA': 800}, }

    if base is not None:
        for key in base.keys():
            for item in base[key].keys():
                basestim[key][item] = base[key][item]

    return basestim


def helper_dpmn(dpmns=None):

    dpmndefaults = ParamSet('dpmndefaults', {'dpmn_tauDOP': 2,
                                             'dpmn_alpha': 0.3,
                                             'dpmn_DAt': 0.0,
                                             'dpmn_taum': 1e100,
                                             'dpmn_dPRE': 0.8,
                                             'dpmn_dPOST': 0.04,
                                             'dpmn_tauE': 15,
                                             'dpmn_tauPRE': 15,
                                             'dpmn_tauPOST': 6,
                                             'dpmn_wmax': 0.3,
                                             'dpmn_w': 0.1286,
                                             'dpmn_Q1': 0.0,
                                             'dpmn_Q2': 0.0,
                                             'dpmn_m': 1.0,
                                             'dpmn_E': 0.0,
                                             'dpmn_DAp': 0.0,
                                             'dpmn_APRE': 0.0,
                                             'dpmn_APOST': 0.0,
                                             'dpmn_XPRE': 0.0,
                                             'dpmn_XPOST': 0.0})

    if dpmns is not None:
        dpmnsdefaults = ModifyViaSelector(dpmndefaults, dpmns)

    return dpmndefaults


def helper_d1(d1=None):

    d1defaults = ParamSet('d1defaults', {'dpmn_type': 1,
                                         'dpmn_alphaw': 55 / 3.0,  # ???
                                         'dpmn_a': 1.0,
                                         'dpmn_b': 0.1,
                                         'dpmn_c': 0.05, })
    if d1 is not None:
        d1defaults = ModifyViaSelector(d1defaults, d1)

    return d1defaults


def helper_d2(d2=None):

    d2defaults = ParamSet('d2defaults', {'dpmn_type': 2,
                                         'dpmn_alphaw': -45 / 3.0,
                                         'dpmn_a': 0.5,
                                         'dpmn_b': 0.005,
                                         'dpmn_c': 0.05, })
    if d2 is not None:
        d2defaults = ModifyViaSelector(d2defaults, d2)

    return d2defaults


def helper_actionchannels(channels=None):
    
    actionchannels = ParamSet('actionchannels', {'action': [1, 2]},  )

    if channels is not None:
        actionchannels = ModifyViaSelector(actionchannels, channels)
    
    return actionchannels

# At some point we have to replace one variable/action(eg t1_epochs, t2_epochs ) by a single data structure. 
# Change this function accordingly. We also have to decide if the format is n_trials x channels or the otherway round

# And maybe this function does not belong in init_params.py. It felt too specific for frontendhelpers.py too. Find a place
def get_reward_value(t1_epochs,t2_epochs,chosen_action,trial_num):
    print("get_reward_value")
    rew_epochs = np.vstack((t1_epochs,t2_epochs)).T
    
    
    # Assuming a n_trials x channels array, ideally this should be a data frame ? so that we do not have to convert chosen action to index chosen_action-1
    reward_val = rew_epochs[trial_num][chosen_action-1]
    print(reward_val)
    return reward_val
    

# Change the reward_value, chosen action to arrays, trial numbers
def helper_init_Q_support_params(q_support=None):
    print("helper_init_Q_support_params")
    Q_support_params = ParamSet('Q_support_params',{'bayes_unif_min':0.,'bayes_unif_max':2.0, 'bayes_H':0.05, 'bayes_sF':1.25, 'q_alpha': 0.45, 'dpmn_CPP_scale':15.,'reward_value' :-1., 'chosen_action': 1})
    
    if q_support is not None:
        Q_support_params = ModifyViaSelector(Q_support_params,q_support)
    
    print(Q_support_params)
    return Q_support_params

def helper_update_Q_support_params(Q_support_params,reward_val,chosen_action):
    print("helper_update_Q_support_params")
    Q_support_params = untrace(Q_support_params)
    
    Q_support_params.reward_value = reward_val
    Q_support_params.chosen_action = chosen_action
    
    print(Q_support_params)
    return Q_support_params
    
# Q_df should have columns are actions, rows are trial numbers
def helper_init_Q_df(actionchannels,q_df=None):
    print("helper_init_Q_df")
    # q_df should be a n_trial+1 x action channels array
    # Start with only dataframe an merge with a new one every trial number - merge that in update Q_df
    # Another Q_val function to initialize the q_val specifically, whether it is same for all actions or not, 0.5
    
    num_actions = len(actionchannels["action"])
    print("num_actions",num_actions)
    Q_df = pd.DataFrame(columns=[actionchannels.iloc[na]["action"] for na in np.arange(num_actions)])
    print("Q_df",Q_df)
    Q_df = Q_df.append({actionchannels.iloc[na]["action"]:0.5 for na in np.arange(num_actions)},ignore_index=True)
    print("Q_df",Q_df)
    # Different initial values for Q_df should be taken care when calling this function with q_df and non-None value
    # eg. q_df = pd.DataFrame({1: 0.5, 2: 0.6})
    
    
    if q_df is not None:
        Q_df = pd.DataFrame(columns=[actionchannels.iloc[na]["action"] for na in np.arange(num_actions)])
        print("Q_df",Q_df)
        Q_df = Q_df.append({actionchannels.iloc[na]["action"]:0.5 for na in np.arange(num_actions)},ignore_index=True)

        Q_df = ModifyViaSelector(Q_df,q_df)
    
    return Q_df


# At this point we assume that the chosen_action has been updated in Q_support_params
def helper_update_Q_df(Q_df, Q_support_params,dpmndefaults,trial_num): 
    print("In update_Q_df") 

    Q_support_params = untrace(Q_support_params)
    #Q_df = untrace(Q_df)
    
    print("Q_support_params")
    print(Q_support_params)
    print("Q_df")
    print(Q_df)
    print("trial_num",trial_num)
    trial_wise_q_df = Q_df.iloc[trial_num] # is this the convention ?, or start with trial_num=0
    trial_wise_chosen_action = Q_support_params.chosen_action
    
    u_val = sp_st.uniform.pdf(Q_support_params.reward_value ,Q_support_params.bayes_unif_min, Q_support_params.bayes_unif_max)

      
    #q_val_chosen = trial_wise_q_df.loc[trial_wise_q_df["action"]==trial_wise_chosen_action]["Q_val"]
    q_val_chosen = trial_wise_q_df[trial_wise_chosen_action]
    
    n_val = sp_st.norm.pdf(Q_support_params.reward_value, q_val_chosen, Q_support_params.bayes_sF)
    
    bayes_CPP = (u_val * Q_support_params.bayes_H) / ((u_val * Q_support_params.bayes_H) + (n_val * (1 - Q_support_params.bayes_H)))

    q_error = Q_support_params.reward_value - q_val_chosen.values
   
    q_val_updated = q_val_chosen.values + Q_support_params.q_alpha.values * q_error
        
    # Copy the updated q value back into thie Q data frame
    
    
    # First append an empty dataframe for the new trial
    #Q_df = Q_df.append({na:Q_df  for na in list(Q_df.columns)},ignore_index=True) # Replace nan with previous trial Q-value
    # Duplicate the last row of df
    new_data = pd.DataFrame(Q_df[-1:].values, columns=Q_df.columns)
    Q_df = Q_df.append(new_data)
    # Update the correct value with q_val_updated
    Q_df.iloc[trial_num+1][trial_wise_chosen_action] = q_val_updated


    dpmndefaults.dpmn_DAp = q_error * bayes_CPP * Q_support_params.dpmn_CPP_scale
    print("====================================================================")
    print("Q_support_params")
    print(Q_support_params)

    print("Q_df")
    print(Q_df)

    print("dpmndefaults")
    print(dpmndefaults)

    return Q_df, Q_support_params, dpmndefaults
    #return dpmndefaults, #Q_support_params
    
    
    
    
    
