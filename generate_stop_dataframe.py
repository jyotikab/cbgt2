import numpy as np
import random
import pandas as pd




def define_stop(stop_signal_probability, actionchannels, n_trials, stop_signal_channel,stop_signal_amplitude,stop_signal_onset,stop_signal_present ):
    #print(actionchannels)

    stop_df = pd.DataFrame() 
    stop_df["stop_signal_amplitude"] = [stop_signal_amplitude]
    stop_df["stop_signal_onset"] = [stop_signal_onset]
    stop_df["stop_signal_present"] = [stop_signal_present]
    stop_df["stop_signal_probability"] = [stop_signal_probability]
    stop_df["stop_signal_channel"] = [stop_signal_channel]
    
    print(stop_df)
    trial_index = np.arange(n_trials)
    stop_channels_df = pd.DataFrame(columns=list(actionchannels.action.values)+["trial_num"])
    stop_channels_df["trial_num"] = trial_index
    for act in list(actionchannels.action.values):
        stop_channels_df[act] = False
         
    trial_indices = np.zeros(n_trials)

    print(type(stop_signal_probability))
    if isinstance(stop_signal_probability,float) == True:
        trials_with_stop_signal = np.random.choice(trial_index,int(n_trials*stop_signal_probability), replace=False)
    elif type(stop_signal_probability) == list:
        trials_with_stop_signal = stop_signal_probability
    
    print(trials_with_stop_signal)        
    for n in np.arange(n_trials):
        
        if stop_signal_channel == "any":
            channels_stop = np.random.choice(list(actionchannels.action.values),1, replace=False)
        elif stop_signal_channel == "all":
            channels_stop = list(actionchannels.action.values)
        
        if n in trials_with_stop_signal:
            for col in channels_stop:
                stop_channels_df.loc[n,col] = True
    
    return stop_df, stop_channels_df #reward_t1, reward_t2


def GenStopSchedule(stop_signal_probability, actionchannels, n_trials, stop_signal_channel, stop_signal_amplitude, stop_signal_onset, stop_signal_present):
    
    #reward_t1, reward_t2
    stop_df, stop_channels_df = define_stop(
        stop_signal_probability, actionchannels, n_trials, stop_signal_channel,stop_signal_amplitude, stop_signal_onset,stop_signal_present)
    
    print("stop_df")
    print(stop_df)
    
    print("stop_channels_df")
    print(stop_channels_df)
    return stop_df, stop_channels_df
