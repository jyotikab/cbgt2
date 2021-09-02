import numpy as np

def multitimestep_mutator(agent,popdata,numsteps):
    for i in range(numsteps):
        timestep_mutator(agent,popdata)

def timestep_mutator(agent,popdata):

    for popid in range(len(popdata)):
        agent.ExtMuS_AMPA[popid] = agent.MeanExtEff_AMPA[popid] * agent.FreqExt_AMPA[popid] * .001 * agent.MeanExtCon_AMPA[popid] * agent.Tau_AMPA[popid]
        agent.ExtSigmaS_AMPA[popid] = agent.MeanExtEff_AMPA[popid] * np.sqrt(agent.Tau_AMPA[popid] * .5 * agent.FreqExt_AMPA[popid] * .001 * agent.MeanExtCon_AMPA[popid])
        agent.ExtS_AMPA[popid] += agent.dt / agent.Tau_AMPA[popid] * (-agent.ExtS_AMPA[popid] + agent.ExtMuS_AMPA[popid]) # + agent.ExtSigmaS_AMPA[popid] * sqrt(agent.dt * 2. / agent.Tau_AMPA[popid]) * gasdev()
        agent.LS_AMPA[popid] *= np.exp(-agent.dt / agent.Tau_AMPA[popid])

    for src_popid in range(len(popdata)):
        for dest_popid in range(len(popdata)):
            if agent.AMPA_con[src_popid][dest_popid] is not None:
                for src_neuron in agent.spikes[src_popid]:
                    agent.LS_AMPA[dest_popid] += agent.AMPA_eff[src_popid][dest_popid][src_neuron] * agent.AMPA_con[src_popid][dest_popid][src_neuron]

    for popid in range(len(popdata)):
        agent.ExtMuS_GABA[popid] = agent.MeanExtEff_GABA[popid] * agent.FreqExt_GABA[popid] * .001 * agent.MeanExtCon_GABA[popid] * agent.Tau_GABA[popid]
        agent.ExtSigmaS_GABA[popid] = agent.MeanExtEff_GABA[popid] * np.sqrt(agent.Tau_GABA[popid] * .5 * agent.FreqExt_GABA[popid] * .001 * agent.MeanExtCon_GABA[popid])
        agent.ExtS_GABA[popid] += agent.dt / agent.Tau_GABA[popid] * (-agent.ExtS_GABA[popid] + agent.ExtMuS_GABA[popid]) # + agent.ExtSigmaS_GABA[popid] * sqrt(agent.dt * 2. / agent.Tau_GABA[popid]) * gasdev()
        agent.LS_GABA[popid] *= np.exp(-agent.dt / agent.Tau_GABA[popid])

    for src_popid in range(len(popdata)):
        for dest_popid in range(len(popdata)):
            if agent.GABA_con[src_popid][dest_popid] is not None:
                for src_neuron in agent.spikes[src_popid]:
                    agent.LS_GABA[dest_popid] += agent.GABA_eff[src_popid][dest_popid][src_neuron] * agent.GABA_con[src_popid][dest_popid][src_neuron]

    for popid in range(len(popdata)):
        agent.ExtMuS_NMDA[popid] = agent.MeanExtEff_NMDA[popid] * agent.FreqExt_NMDA[popid] * .001 * agent.MeanExtCon_NMDA[popid] * agent.Tau_NMDA[popid]
        agent.ExtSigmaS_NMDA[popid] = agent.MeanExtEff_NMDA[popid] * np.sqrt(agent.Tau_NMDA[popid] * .5 * agent.FreqExt_NMDA[popid] * .001 * agent.MeanExtCon_NMDA[popid])
        agent.ExtS_NMDA[popid] += agent.dt / agent.Tau_NMDA[popid] * (-agent.ExtS_NMDA[popid] + agent.ExtMuS_NMDA[popid]) # + agent.ExtSigmaS_NMDA[popid] * sqrt(agent.dt * 2. / agent.Tau_NMDA[popid]) * gasdev()
        agent.LS_NMDA[popid] *= np.exp(-agent.dt / agent.Tau_NMDA[popid])
        agent.timesincelastspike[popid] += agent.dt

    for src_popid in range(len(popdata)):
        for dest_popid in range(len(popdata)):
            if agent.NMDA_con[src_popid][dest_popid] is not None:
                for src_neuron in agent.spikes[src_popid]:
                    agent.LastConductanceNMDA[src_popid][dest_popid][src_neuron] *= np.exp(-agent.timesincelastspike[src_popid][src_neuron]*agent.Tau_NMDA[dest_popid])
                    agent.LS_NMDA[dest_popid] += agent.NMDA_eff[src_popid][dest_popid][src_neuron] * agent.NMDA_con[src_popid][dest_popid][src_neuron]
                    agent.LastConductanceNMDA[src_popid][dest_popid][src_neuron] += 0.6332 * (1 - agent.LastConductanceNMDA[src_popid][dest_popid][src_neuron])

    for popid in range(len(popdata)):
        for neuron in agent.spikes[popid]:
            agent.timesincelastspike[popid][neuron] = 0

    for popid in range(len(popdata)):
        agent.cond[popid] = (agent.V[popid] < agent.V_h[popid]).astype(int)
        agent.h[popid] = agent.h[popid] + agent.dt * (agent.cond[popid] - agent.h[popid]) / agent.tauhp[popid]
        agent.g_rb[popid] = agent.g_T[popid] * agent.h[popid] * (1 - agent.cond[popid])

    for popid in range(len(popdata)):
        agent.cond[popid] = (agent.V[popid] <= agent.Threshold[popid]).astype(int) * (agent.RefrState[popid] == 0).astype(int)

        agent.V[popid] -= (agent.V[popid] - agent.ResetPot[popid]) * (1 - agent.cond[popid])
        agent.RefrState[popid] -= np.sign(agent.RefrState[popid])

        agent.g_adr[popid] = agent.g_adr_max[popid] / (1 + np.exp((agent.V[popid]-agent.Vadr_h[popid]) / agent.Vadr_s[popid]))

        agent.dv[popid] = agent.V[popid] + 55
        agent.tau_n[popid] = agent.tau_k_max[popid] / (np.exp(-1 * agent.dv[popid] / 30) + np.exp(agent.dv[popid] / 30))
        agent.n_inif[popid] = 1 / (1 + np.exp(-(agent.V[popid] - agent.Vk_h[popid]) / agent.Vk_s[popid]))
        agent.n_k[popid] = agent.n_k[popid] + agent.cond[popid] * -agent.dt / agent.tau_n[popid] * (agent.n_k[popid] - agent.n_inif[popid])
        agent.g_k[popid] = agent.g_k_max[popid] * agent.n_k[popid]

        agent.V[popid] = agent.V[popid] + agent.cond[popid] * -agent.dt * (1 / agent.Taum[popid] * (agent.V[popid] - agent.RestPot[popid]) + agent.Ca[popid] * agent.g_ahp[popid] / agent.C[popid] * 0.001 * (agent.V[popid] - agent.Vk[popid]) + agent.g_adr[popid] / agent.C[popid] * (agent.V[popid] - agent.ADRRevPot[popid]) + agent.g_k[popid] / agent.C[popid] * (agent.V[popid] - agent.ADRRevPot[popid]) + agent.g_rb[popid] / agent.C[popid] * (agent.V[popid] - agent.V_T[popid]))
        agent.Ca[popid] = agent.Ca[popid] - agent.cond[popid] * agent.Ca[popid] * agent.dt / agent.Tau_ca[popid]

        agent.Vaux[popid] = np.minimum(agent.V[popid],agent.Threshold[popid])

        agent.V[popid] = agent.V[popid] + agent.cond[popid] * agent.dt * (agent.RevPot_NMDA[popid] - agent.Vaux[popid]) * .001 * (agent.LS_NMDA[popid] + agent.ExtS_NMDA[popid]) / agent.C[popid] / (1. + np.exp(-0.062 * agent.Vaux[popid] / 3.57))
        agent.V[popid] = agent.V[popid] + agent.cond[popid] * agent.dt * (agent.RevPot_AMPA[popid] - agent.Vaux[popid]) * .001 * (agent.LS_AMPA[popid] + agent.ExtS_AMPA[popid]) / agent.C[popid]
        agent.V[popid] = agent.V[popid] + agent.cond[popid] * agent.dt * (agent.RevPot_GABA[popid] - agent.Vaux[popid]) * .001 * (agent.LS_GABA[popid] + agent.ExtS_GABA[popid]) / agent.C[popid]

    for popid in range(len(popdata)):
        agent.spikes[popid] = list(np.nonzero(agent.V[popid] > agent.Threshold[popid])[0])
        for neuron in agent.spikes[popid]:
            agent.V[popid][neuron] = 0
            agent.Ca[popid][neuron] += agent.alpha_ca[popid][neuron]
            #agent.dpmn_XPOST[popid] = spikes

    for popid in range(len(popdata)):
        agent.rollingbuffer[popid][agent.bufferpointer] = len(agent.spikes[popid])
    agent.bufferpointer += 1
    if agent.bufferpointer >= agent.bufferlength:
        agent.bufferpointer = 0
