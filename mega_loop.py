# 1: IMPORTING SCRIPTS

import cbgt as cbgt
from frontendhelpers import * 
from tracetype import *
import init_params as par 
import popconstruct as popconstruct
import qvalues as qval
import generateepochs as gen
from agentmatrixinit import *
from agent_timestep import timestep_mutator, multitimestep_mutator
import pipeline_creation as pl_creat

def mega_loop(slf):
    slf.AMPA_con,slf.AMPA_eff = CreateSynapses(slf.popdata,slf.connectivity_AMPA,slf.meaneff_AMPA,slf.plastic_AMPA)
    slf.GABA_con,slf.GABA_eff = CreateSynapses(slf.popdata,slf.connectivity_GABA,slf.meaneff_GABA,slf.plastic_GABA)
    slf.NMDA_con,slf.NMDA_eff = CreateSynapses(slf.popdata,slf.connectivity_NMDA,slf.meaneff_NMDA,slf.plastic_NMDA)

    popdata = slf.popdata
    actionchannels = slf.actionchannels
    agent = initializeAgent(popdata)
    slf.agent = agent

    agent.AMPA_con,agent.AMPA_eff = slf.AMPA_con,slf.AMPA_eff
    agent.GABA_con,agent.GABA_eff = slf.GABA_con,slf.GABA_eff
    agent.NMDA_con,agent.NMDA_eff = slf.NMDA_con,slf.NMDA_eff
    agent.LastConductanceNMDA = CreateAuxiliarySynapseData(popdata,slf.connectivity_NMDA)

    agent.FRs = agent.rollingbuffer.mean(1) / untrace(list(popdata['N'])) / agent.dt * 1000

    multitimestep_mutator(agent,popdata,1000)

    agent.phase = 0
    agent.phasetimer = 0
    agent.motor_queued = None
    agent.dpmn_queued = None
    agent.gain = np.ones(len(actionchannels))
    agent.extstim = np.zeros(len(actionchannels))
    agent.ramping_extstim = np.zeros(len(actionchannels))
    agent.in_popids = np.where(popdata['name'] == 'LIP')[0]
    agent.out_popids = np.where(popdata['name'] == 'Th')[0]

    presented_stimulus = 1
    slf.chosen_action = None

    while slf.trial_num < slf.n_trials:
        agent.extstim = agent.gain * presented_stimulus * 4.0  # TODO: make 3.0 a param
        agent.ramping_extstim = agent.ramping_extstim * 0.9 + agent.extstim * 0.1
        for action_idx in range(len(actionchannels)):
            popid = agent.in_popids[action_idx]
            agent.FreqExt_AMPA[popid] = np.ones(len(agent.FreqExt_AMPA[popid])) * agent.ramping_extstim[action_idx]
        multitimestep_mutator(agent,popdata,5)
        agent.phasetimer += 1 # 1 ms = 5 * dt
        #agent.FRs = np.stack((agent.FRs,agent.rollingbuffer.mean(1) / untrace(list(popdata['N'])) / agent.dt * 1000))

        if agent.phase == 0:
            gateFRs = agent.rollingbuffer[agent.out_popids].mean(1) / untrace(list(popdata['N'][agent.out_popids])) / agent.dt * 1000
            thresholds_crossed = np.where(gateFRs > 30)[0]
            if len(thresholds_crossed) > 0 or agent.phasetimer > 1000:
                agent.phase = 1
                agent.phasetimer = 0
                agent.gain = np.zeros(len(actionchannels))
                if len(thresholds_crossed) > 0:
                    agent.motor_queued = thresholds_crossed[0]
                    agent.gain[agent.motor_queued] = 0.5
                else:
                    agent.motor_queued = -1

        if agent.phase == 1:
            if agent.phasetimer > 100:
                agent.phase = 2
                agent.phasetimer = 0
                agent.gain = np.zeros(len(actionchannels))
                print(actionchannels)
                slf.chosen_action = untrace(actionchannels.iloc[agent.motor_queued,0])
                print("chosen_action",slf.chosen_action)
                agent.motor_queued = None

        if agent.phase == 2:
            if agent.phasetimer > 100:
                slf.dpmndefaults['dpmn_DAp'] = 0
                slf.trial_num += 1
                agent.phase = 0
                agent.phasetimer = 0


        # environment

        if slf.chosen_action is not None:
            #slf.reward_val = qval.get_reward_value(slf.t1_epochs,slf.t2_epochs,slf.chosen_action,slf.trial_num)
            slf.reward_val = qval.get_reward_value(slf.t_epochs,slf.chosen_action,slf.trial_num)
            slf.Q_support_params = qval.helper_update_Q_support_params(slf.Q_support_params,slf.reward_val,slf.chosen_action)
            (slf.Q_df, slf.Q_support_params, slf.dpmndefaults) = qval.helper_update_Q_df(slf.Q_df,slf.Q_support_params,slf.dpmndefaults,slf.trial_num)
            slf.chosen_action = None

    