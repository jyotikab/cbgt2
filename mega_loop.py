# 1. IMPORTING SCRIPTS

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

# 2. TIMESTEP LOOP

def mega_loop(self):
    self.AMPA_con,self.AMPA_eff = CreateSynapses(self.popdata,self.connectivity_AMPA,self.meaneff_AMPA,self.plastic_AMPA)
    self.GABA_con,self.GABA_eff = CreateSynapses(self.popdata,self.connectivity_GABA,self.meaneff_GABA,self.plastic_GABA)
    self.NMDA_con,self.NMDA_eff = CreateSynapses(self.popdata,self.connectivity_NMDA,self.meaneff_NMDA,self.plastic_NMDA)

    popdata = self.popdata
    actionchannels = self.actionchannels
    agent = initializeAgent(popdata)
    self.agent = agent

    popdata['column'] = popdata.index

    agent.AMPA_con,agent.AMPA_eff = self.AMPA_con,self.AMPA_eff
    agent.GABA_con,agent.GABA_eff = self.GABA_con,self.GABA_eff
    agent.NMDA_con,agent.NMDA_eff = self.NMDA_con,self.NMDA_eff
    agent.LastConductanceNMDA = CreateAuxiliarySynapseData(popdata,self.connectivity_NMDA)


    agent.phase = 0
    agent.globaltimer = 0
    agent.phasetimer = 0
    agent.motor_queued = None
    agent.dpmn_queued = None
    agent.gain = np.ones(len(actionchannels))
    agent.extstim = np.zeros(len(actionchannels))
    agent.ramping_extstim = np.zeros(len(actionchannels))
    agent.in_popids = np.where(popdata['name'] == 'LIP')[0]
    agent.out_popids = np.where(popdata['name'] == 'Th')[0]

    presented_stimulus = 1
    self.chosen_action = None

    for action_idx in range(len(actionchannels)):
        popid = agent.in_popids[action_idx]
        agent.FreqExt_AMPA[popid] = np.zeros(len(agent.FreqExt_AMPA[popid]))
    multitimestep_mutator(agent,popdata,5000)
    agent.FRs = [agent.rollingbuffer.mean(1) / untrace(list(popdata['N'])) / agent.dt * 1000]

    datatables_decision = None
    datatables_stimulusstarttime = agent.globaltimer
    datatables_decisiontime = None
    datatables_decisionduration = None
    datatables_decisiondurationplusdelay = None
    datatables_rewardtime = None
    datatables_correctdecision = None
    datatables_reward = None

    self.datatables = pd.DataFrame([], columns=["decision", "stimulusstarttime", "decisiontime", "decisionduration", "decisiondurationplusdelay", "rewardtime", "correctdecision", "reward"])
    self.datatables.index.name = 'trial'


    while self.trial_num < self.n_trials:
        agent.extstim = agent.gain * presented_stimulus * self.maxstim  # TODO: make 3.0 a param
        agent.ramping_extstim = agent.ramping_extstim * 0.9 + agent.extstim * 0.1
        for action_idx in range(len(actionchannels)):
            popid = agent.in_popids[action_idx]
            agent.FreqExt_AMPA[popid] = np.ones(len(agent.FreqExt_AMPA[popid])) * agent.ramping_extstim[action_idx]
        multitimestep_mutator(agent,popdata,5)
        agent.phasetimer += 1 # 1 ms = 5 * dt
        agent.globaltimer += 1 # 1 ms = 5 * dt
        agent.FRs = np.concatenate((agent.FRs,[agent.rollingbuffer.mean(1) / untrace(list(popdata['N'])) / agent.dt * 1000]))

        if agent.phase == 0:
            gateFRs = agent.rollingbuffer[agent.out_popids].mean(1) / untrace(list(popdata['N'][agent.out_popids])) / agent.dt * 1000
            thresholds_crossed = np.where(gateFRs > 30)[0]
            if len(thresholds_crossed) > 0 or agent.phasetimer > 1000:

                print('phasetimer',agent.phasetimer)
                print('globaltimer',agent.globaltimer)
                print('gateFRs',gateFRs)
                print('thresholds_crossed',thresholds_crossed)
                agent.phase = 1
                agent.phasetimer = 0
                agent.gain = np.zeros(len(actionchannels))
                datatables_decisiontime = agent.globaltimer
                datatables_decisionduration = agent.globaltimer - datatables_stimulusstarttime
                if len(thresholds_crossed) > 0:
                    agent.motor_queued = thresholds_crossed[0]
                    datatables_decision = agent.motor_queued

                    agent.gain[agent.motor_queued] = 0.75 # sustained fraction in old network
                else:
                    agent.motor_queued = -1

        if agent.phase == 1:
            if agent.phasetimer > 300:
                agent.phase = 2
                print('phasetimer',agent.phasetimer)
                print('globaltimer',agent.globaltimer)
                agent.phasetimer = 0
                agent.gain = np.zeros(len(actionchannels))
                print(actionchannels)
                datatables_rewardtime = agent.globaltimer
                datatables_decisiondurationplusdelay = agent.globaltimer - datatables_stimulusstarttime
                if agent.motor_queued == -1:
                    self.chosen_action = None
                else:
                    self.chosen_action = untrace(actionchannels.iloc[agent.motor_queued,0])
                print("chosen_action",self.chosen_action)
                agent.motor_queued = None

        if agent.phase == 2:
            if agent.phasetimer > 300:
                self.dpmndefaults['dpmn_DAp'] = 0
                self.trial_num += 1
                agent.phase = 0
                agent.phasetimer = 0
                agent.gain = np.ones(len(actionchannels))

                datatablesrow = pd.DataFrame([[
                    datatables_decision,
                    datatables_stimulusstarttime,
                    datatables_decisiontime,
                    datatables_decisionduration,
                    datatables_decisiondurationplusdelay,
                    datatables_rewardtime,
                    datatables_correctdecision,
                    datatables_reward,
                ]], columns=["decision", "stimulusstarttime", "decisiontime", "decisionduration", "decisiondurationplusdelay", "rewardtime", "correctdecision", "reward"])
                datatablesrow.index.name = 'trial'

                self.datatables = pd.concat([self.datatables,datatablesrow], ignore_index=True)

                datatables_decision = None
                datatables_stimulusstarttime = agent.globaltimer
                datatables_decisiontime = None
                datatables_decisionduration = None
                datatables_decisiondurationplusdelay = None
                datatables_rewardtime = None
                datatables_correctdecision = None
                datatables_reward = None

        # environment

        if self.chosen_action is not None:
            #self.reward_val = qval.get_reward_value(self.t1_epochs,self.t2_epochs,self.chosen_action,self.trial_num)
            self.reward_val = qval.get_reward_value(self.t_epochs,self.chosen_action,self.trial_num)
            datatables_reward = np.sign(self.reward_val)
            self.Q_support_params = qval.helper_update_Q_support_params(self.Q_support_params,self.reward_val,self.chosen_action)
            (self.Q_df, self.Q_support_params, self.dpmndefaults) = qval.helper_update_Q_df(self.Q_df,self.Q_support_params,self.dpmndefaults,self.trial_num)
            self.chosen_action = None

    self.popfreqs = pd.DataFrame(agent.FRs)
    self.popfreqs['Time (ms)'] = self.popfreqs.index
