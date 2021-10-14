import numpy as np

def multitimestep_mutator(agent,popdata,numsteps):
    for i in range(numsteps):
        timestep_mutator(agent,popdata)


def timestep_mutator_2(a,popdata):
    Npop = len(popdata)

    newspikes = []
    for i in range(Npop):
        newspikes.append([])

    # 1337-1345
    for p in range(Npop):
        if a.dpmn_type[p][0] > 0:
            pass
            #a.dpmn_XPRE[p] *= 0
            #a.dpmn_XPOST[p] *= 0

    # 1347-1392
    # Compute the decay of the total conductances and add external input
    for p in range(Npop):
        selected = np.random.randint(len(a.N[p]))
        for i in range(len(a.N[p])):
            # AMPA
            freq = a.FreqExt_AMPA[p][i]
            efficacy = a.MeanExtEff_AMPA[p][i]
            a.ExtMuS_AMPA[p][i] = freq * .001 * efficacy * a.MeanExtCon_AMPA[p][i] * a.Tau_AMPA[p][i]
            a.ExtSigmaS_AMPA[p][i] = np.sqrt(a.Tau_AMPA[p][i] * .5 * freq * .001 * efficacy * efficacy * a.MeanExtCon_AMPA[p][i])
            s = a.ExtSigmaS_AMPA[p][i]
            if s > 0:
                a.ExtS_AMPA[p][i] += a.dt / a.Tau_AMPA[p][i] * (-a.ExtS_AMPA[p][i] + a.ExtMuS_AMPA[p][i]) + s * np.sqrt(a.dt * 2. / a.Tau_AMPA[p][i]) * np.random.normal()
            else:
                a.ExtS_AMPA[p][i] += a.dt / a.Tau_AMPA[p][i] * (-a.ExtS_AMPA[p][i] + a.ExtMuS_AMPA[p][i])
            a.LS_AMPA[p][i] *= np.exp(-a.dt / a.Tau_AMPA[p][i]) # decay

            # GABA
            freq = a.FreqExt_GABA[p][i]
            efficacy = a.MeanExtEff_GABA[p][i]
            a.ExtMuS_GABA[p][i] = freq * .001 * efficacy * a.MeanExtCon_GABA[p][i] * a.Tau_GABA[p][i]
            a.ExtSigmaS_GABA[p][i] = np.sqrt(a.Tau_GABA[p][i] * .5 * freq * .001 * efficacy * efficacy * a.MeanExtCon_GABA[p][i])
            s = a.ExtSigmaS_GABA[p][i]
            if s > 0:
                a.ExtS_GABA[p][i] += a.dt / a.Tau_GABA[p][i] * (-a.ExtS_GABA[p][i] + a.ExtMuS_GABA[p][i]) + s * np.sqrt(a.dt * 2. / a.Tau_GABA[p][i]) * np.random.normal()
            else:
                a.ExtS_GABA[p][i] += a.dt / a.Tau_GABA[p][i] * (-a.ExtS_GABA[p][i] + a.ExtMuS_GABA[p][i])
            a.LS_GABA[p][i] *= np.exp(-a.dt / a.Tau_GABA[p][i]) # decay

            # NMDA
            freq = a.FreqExt_NMDA[p][i]
            efficacy = a.MeanExtEff_NMDA[p][i]
            a.ExtMuS_NMDA[p][i] = freq * .001 * efficacy * a.MeanExtCon_NMDA[p][i] * a.Tau_NMDA[p][i]
            a.ExtSigmaS_NMDA[p][i] = np.sqrt(a.Tau_NMDA[p][i] * .5 * freq * .001 * efficacy * efficacy * a.MeanExtCon_NMDA[p][i])
            s = a.ExtSigmaS_NMDA[p][i]
            if s > 0:
                a.ExtS_NMDA[p][i] += a.dt / a.Tau_NMDA[p][i] * (-a.ExtS_NMDA[p][i] + a.ExtMuS_NMDA[p][i]) + s * np.sqrt(a.dt * 2. / a.Tau_NMDA[p][i]) * np.random.normal()
            else:
                a.ExtS_NMDA[p][i] += a.dt / a.Tau_NMDA[p][i] * (-a.ExtS_NMDA[p][i] + a.ExtMuS_NMDA[p][i])
            a.LS_NMDA[p][i] *= np.exp(-a.dt / a.Tau_NMDA[p][i]) # decay

            a.timesincelastspike[p][i] += a.dt

    # 1396-1449
    # Update the total conductances (changes provoked by the spikes)
    for p in range(Npop):
        for sourceneuron in a.spikes[p]:
            for tp in range(Npop):
                # AMPA
                if a.AMPA_con[p][tp] is not None:
                    for tn in range(len(a.N[tp])):
                        pathway_strength = a.AMPA_eff[p][tp][sourceneuron,tn]
                        a.LS_AMPA[tp][tn] += pathway_strength

                # GABA
                if a.GABA_con[p][tp] is not None:
                    for tn in range(len(a.N[tp])):
                        pathway_strength = a.GABA_eff[p][tp][sourceneuron,tn]
                        a.LS_GABA[tp][tn] += pathway_strength

                # NMDA
                if a.NMDA_con[p][tp] is not None:
                    for tn in range(len(a.N[tp])):
                        pathway_strength = a.NMDA_eff[p][tp][sourceneuron,tn]
                        ALPHA = 0.6332
                        a.LastConductanceNMDA[p][tp][sourceneuron,tn] *= np.exp(-a.timesincelastspike[p][sourceneuron] / a.Tau_NMDA[tp][tn])
                        a.LS_NMDA[tp][tn] += pathway_strength * ALPHA * (1. - a.LastConductanceNMDA[p][tp][sourceneuron,tn])
                        a.LastConductanceNMDA[p][tp][sourceneuron,tn] += ALPHA * (1. - a.LastConductanceNMDA[p][tp][sourceneuron,tn])

                # dopaminergic learning
                #if a.dpmn_cortex[p][sourceneuron] > 0 and a.dpmn_type[tp][tn] > 0:
                #    a.dpmn_XPRE[tp][tn] = 1

    for p in range(Npop):
        for i in range(len(a.N[p])):
            g_rb = 0
            if a.V[p][i] < a.V_h[p][i]:
                a.h[p][i] += (1.0 - a.h[p][i]) * a.dt / a.tauhp[p][i]
            else:
                a.h[p][i] += -a.h[p][i] * a.dt / a.tauhm[p][i]
                g_rb = a.g_T[p][i] * a.h[p][i]

            if a.V[p][i] > a.Threshold[p][i]:
                a.V[p][i] = a.ResetPot[p][i]
                a.RefrState[p][i] -= 1
                continue

            if a.RefrState[p][i] > 0:
                a.RefrState[p][i] -= 1
                continue

            g_adr = 0
            if a.g_adr_max[p][i] != 0:
                g_adr = a.g_adr_max[p][i] / (1 + np.exp((a.V[p][i] - a.Vadr_h[p][i]) / a.Vadr_s[p][i]))

            g_k = 0
            if a.g_k_max[p][i] != 0:
                tau_max = a.tau_k_max[p][i]
                dv = a.V[p][i] + 55.0
                tau_n = tau_max / (np.exp(-1 * dv / 30) + np.exp(dv / 30))
                n_inif = 1 / (1 + np.exp(-(a.V[p][i] - a.Vk_h[p][i]) / a.Vk_s[p][i]))
                a.n_k[p][i] += -a.dt / tau_n * (a.n_k[p][i] - n_inif)
                g_k = a.g_k_max[p][i] * a.n_k[p][i]

            if a.g_ahp[p][i] == 0:
                a.V[p][i] += -a.dt * (1 / a.Taum[p][i] * (a.V[p][i] - a.RestPot[p][i]) + g_adr / a.C[p][i] * (a.V[p][i] - a.ADRRevPot[p][i]) + g_k / a.C[p][i] * (a.V[p][i] - a.ADRRevPot[p][i]) + g_rb / a.C[p][i] * (a.V[p][i] - a.V_T[p][i]))
            else:
                a.V[p][i] += -a.dt * (1 / a.Taum[p][i] * (a.V[p][i] - a.RestPot[p][i]) +  a.Ca[p][i] * a.g_ahp[p][i] / a.C[p][i] * 0.001 * (a.V[p][i] - a.Vk[p][i]) + g_adr / a.C[p][i] * (a.V[p][i] - a.ADRRevPot[p][i]) + g_k / a.C[p][i] * (a.V[p][i] - a.ADRRevPot[p][i]) + g_rb / a.C[p][i] * (a.V[p][i] - a.V_T[p][i]))

            a.Ca[p][i] += -a.Ca[p][i] * a.dt / a.Tau_ca[p][i]

            Vaux = a.V[p][i]
            if Vaux > a.Threshold[p][i]:
                Vaux = a.Threshold[p][i]

            # AMPA
            a.V[p][i] += a.dt * (a.RevPot_AMPA[p][i] - Vaux) * .001 * (a.LS_AMPA[p][i] + a.ExtS_AMPA[p][i]) / a.C[p][i]
            # GABA
            a.V[p][i] += a.dt * (a.RevPot_GABA[p][i] - Vaux) * .001 * (a.LS_GABA[p][i] + a.ExtS_GABA[p][i]) / a.C[p][i]
            # NMDA
            a.V[p][i] += a.dt * (a.RevPot_NMDA[p][i] - Vaux) * .001 * (a.LS_NMDA[p][i] + a.ExtS_NMDA[p][i]) / a.C[p][i] / (1. + np.exp(-0.062 * Vaux / 3.57))

            if a.V[p][i] > a.Threshold[p][i]:
                #print("a spike occured!")
                newspikes[p].append(i)

                a.V[p][i] = a.ResetPot[p][i]
                a.timesincelastspike[p][i] = 0

                a.V[p][i] = 0
                a.RefrState[p][i] = 10 # 2/dt

                a.Ca[p][i] += a.alpha_ca[p][i]

    a.spikes = newspikes

    for p in range(Npop):
        a.rollingbuffer[p][a.bufferpointer] = len(a.spikes[p])
    a.bufferpointer += 1
    if a.bufferpointer >= a.bufferlength:
        a.bufferpointer = 0



def timestep_mutator(a,popdata):

    Npop = len(popdata)

    newspikes = []
    for i in range(Npop):
        newspikes.append([])

    for popid in range(len(popdata)):
        a.ExtMuS_AMPA[popid] = a.MeanExtEff_AMPA[popid] * a.FreqExt_AMPA[popid] * .001 * a.MeanExtCon_AMPA[popid] * a.Tau_AMPA[popid]
        a.ExtSigmaS_AMPA[popid] = a.MeanExtEff_AMPA[popid] * np.sqrt(a.Tau_AMPA[popid] * .5 * a.FreqExt_AMPA[popid] * .001 * a.MeanExtCon_AMPA[popid])
        a.ExtS_AMPA[popid] += a.dt / a.Tau_AMPA[popid] * (-a.ExtS_AMPA[popid] + a.ExtMuS_AMPA[popid]) + a.ExtSigmaS_AMPA[popid] * np.sqrt(a.dt * 2. / a.Tau_AMPA[popid]) * np.random.normal(size=len(a.Tau_AMPA[popid]))
        a.LS_AMPA[popid] *= np.exp(-a.dt / a.Tau_AMPA[popid])

    for src_popid in range(len(popdata)):
        for dest_popid in range(len(popdata)):
            if a.AMPA_con[src_popid][dest_popid] is not None:
                for src_neuron in a.spikes[src_popid]:
                    a.LS_AMPA[dest_popid] += a.AMPA_eff[src_popid][dest_popid][src_neuron] * a.AMPA_con[src_popid][dest_popid][src_neuron]

    for popid in range(len(popdata)):
        a.ExtMuS_GABA[popid] = a.MeanExtEff_GABA[popid] * a.FreqExt_GABA[popid] * .001 * a.MeanExtCon_GABA[popid] * a.Tau_GABA[popid]
        a.ExtSigmaS_GABA[popid] = a.MeanExtEff_GABA[popid] * np.sqrt(a.Tau_GABA[popid] * .5 * a.FreqExt_GABA[popid] * .001 * a.MeanExtCon_GABA[popid])
        a.ExtS_GABA[popid] += a.dt / a.Tau_GABA[popid] * (-a.ExtS_GABA[popid] + a.ExtMuS_GABA[popid]) + a.ExtSigmaS_GABA[popid] * np.sqrt(a.dt * 2. / a.Tau_GABA[popid]) * np.random.normal(size=len(a.Tau_AMPA[popid]))
        a.LS_GABA[popid] *= np.exp(-a.dt / a.Tau_GABA[popid])

    for src_popid in range(len(popdata)):
        for dest_popid in range(len(popdata)):
            if a.GABA_con[src_popid][dest_popid] is not None:
                for src_neuron in a.spikes[src_popid]:
                    a.LS_GABA[dest_popid] += a.GABA_eff[src_popid][dest_popid][src_neuron] * a.GABA_con[src_popid][dest_popid][src_neuron]

    for popid in range(len(popdata)):
        a.ExtMuS_NMDA[popid] = a.MeanExtEff_NMDA[popid] * a.FreqExt_NMDA[popid] * .001 * a.MeanExtCon_NMDA[popid] * a.Tau_NMDA[popid]
        a.ExtSigmaS_NMDA[popid] = a.MeanExtEff_NMDA[popid] * np.sqrt(a.Tau_NMDA[popid] * .5 * a.FreqExt_NMDA[popid] * .001 * a.MeanExtCon_NMDA[popid])
        a.ExtS_NMDA[popid] += a.dt / a.Tau_NMDA[popid] * (-a.ExtS_NMDA[popid] + a.ExtMuS_NMDA[popid]) + a.ExtSigmaS_NMDA[popid] * np.sqrt(a.dt * 2. / a.Tau_NMDA[popid]) * np.random.normal(size=len(a.Tau_AMPA[popid]))
        a.LS_NMDA[popid] *= np.exp(-a.dt / a.Tau_NMDA[popid])
        a.timesincelastspike[popid] += a.dt
        a.Ptimesincelastspike[popid] += a.dt

    for src_popid in range(len(popdata)):
        for dest_popid in range(len(popdata)):
            if a.NMDA_con[src_popid][dest_popid] is not None:
                for src_neuron in a.spikes[src_popid]:
                    ALPHA = 0.6332
                    a.LastConductanceNMDA[src_popid][dest_popid][src_neuron] *= np.exp(-a.Ptimesincelastspike[src_popid][src_neuron]/a.Tau_NMDA[dest_popid])
                    a.LS_NMDA[dest_popid] += a.NMDA_eff[src_popid][dest_popid][src_neuron] * a.NMDA_con[src_popid][dest_popid][src_neuron] * ALPHA * (1 - a.LastConductanceNMDA[src_popid][dest_popid][src_neuron])
                    a.LastConductanceNMDA[src_popid][dest_popid][src_neuron] += ALPHA * (1 - a.LastConductanceNMDA[src_popid][dest_popid][src_neuron])

    # # 1347-1392
    # # Compute the decay of the total conductances and add external input
    # for p in range(Npop):
    #     selected = np.random.randint(len(a.N[p]))
    #     for i in range(len(a.N[p])):
    #         # AMPA
    #         freq = a.FreqExt_AMPA[p][i]
    #         efficacy = a.MeanExtEff_AMPA[p][i]
    #         a.ExtMuS_AMPA[p][i] = freq * .001 * efficacy * a.MeanExtCon_AMPA[p][i] * a.Tau_AMPA[p][i]
    #         a.ExtSigmaS_AMPA[p][i] = np.sqrt(a.Tau_AMPA[p][i] * .5 * freq * .001 * efficacy * efficacy * a.MeanExtCon_AMPA[p][i])
    #         s = a.ExtSigmaS_AMPA[p][i]
    #         if s > 0:
    #             a.ExtS_AMPA[p][i] += a.dt / a.Tau_AMPA[p][i] * (-a.ExtS_AMPA[p][i] + a.ExtMuS_AMPA[p][i]) + s * np.sqrt(a.dt * 2. / a.Tau_AMPA[p][i]) * np.random.normal()
    #         else:
    #             a.ExtS_AMPA[p][i] += a.dt / a.Tau_AMPA[p][i] * (-a.ExtS_AMPA[p][i] + a.ExtMuS_AMPA[p][i])
    #         a.LS_AMPA[p][i] *= np.exp(-a.dt / a.Tau_AMPA[p][i]) # decay
    #
    #         # GABA
    #         freq = a.FreqExt_GABA[p][i]
    #         efficacy = a.MeanExtEff_GABA[p][i]
    #         a.ExtMuS_GABA[p][i] = freq * .001 * efficacy * a.MeanExtCon_GABA[p][i] * a.Tau_GABA[p][i]
    #         a.ExtSigmaS_GABA[p][i] = np.sqrt(a.Tau_GABA[p][i] * .5 * freq * .001 * efficacy * efficacy * a.MeanExtCon_GABA[p][i])
    #         s = a.ExtSigmaS_GABA[p][i]
    #         if s > 0:
    #             a.ExtS_GABA[p][i] += a.dt / a.Tau_GABA[p][i] * (-a.ExtS_GABA[p][i] + a.ExtMuS_GABA[p][i]) + s * np.sqrt(a.dt * 2. / a.Tau_GABA[p][i]) * np.random.normal()
    #         else:
    #             a.ExtS_GABA[p][i] += a.dt / a.Tau_GABA[p][i] * (-a.ExtS_GABA[p][i] + a.ExtMuS_GABA[p][i])
    #         a.LS_GABA[p][i] *= np.exp(-a.dt / a.Tau_GABA[p][i]) # decay
    #
    #         # NMDA
    #         freq = a.FreqExt_NMDA[p][i]
    #         efficacy = a.MeanExtEff_NMDA[p][i]
    #         a.ExtMuS_NMDA[p][i] = freq * .001 * efficacy * a.MeanExtCon_NMDA[p][i] * a.Tau_NMDA[p][i]
    #         a.ExtSigmaS_NMDA[p][i] = np.sqrt(a.Tau_NMDA[p][i] * .5 * freq * .001 * efficacy * efficacy * a.MeanExtCon_NMDA[p][i])
    #         s = a.ExtSigmaS_NMDA[p][i]
    #         if s > 0:
    #             a.ExtS_NMDA[p][i] += a.dt / a.Tau_NMDA[p][i] * (-a.ExtS_NMDA[p][i] + a.ExtMuS_NMDA[p][i]) + s * np.sqrt(a.dt * 2. / a.Tau_NMDA[p][i]) * np.random.normal()
    #         else:
    #             a.ExtS_NMDA[p][i] += a.dt / a.Tau_NMDA[p][i] * (-a.ExtS_NMDA[p][i] + a.ExtMuS_NMDA[p][i])
    #         a.LS_NMDA[p][i] *= np.exp(-a.dt / a.Tau_NMDA[p][i]) # decay
    #
    #         a.timesincelastspike[p][i] += a.dt
    #
    # # 1396-1449
    # # Update the total conductances (changes provoked by the spikes)
    # for p in range(Npop):
    #     for sourceneuron in a.spikes[p]:
    #         for tp in range(Npop):
    #             # AMPA
    #             if a.AMPA_con[p][tp] is not None:
    #                 for tn in range(len(a.N[tp])):
    #                     pathway_strength = a.AMPA_eff[p][tp][sourceneuron,tn]
    #                     a.LS_AMPA[tp][tn] += pathway_strength
    #
    #             # GABA
    #             if a.GABA_con[p][tp] is not None:
    #                 for tn in range(len(a.N[tp])):
    #                     pathway_strength = a.GABA_eff[p][tp][sourceneuron,tn]
    #                     a.LS_GABA[tp][tn] += pathway_strength
    #
    #             # NMDA
    #             if a.NMDA_con[p][tp] is not None:
    #                 for tn in range(len(a.N[tp])):
    #                     pathway_strength = a.NMDA_eff[p][tp][sourceneuron,tn]
    #                     ALPHA = 0.6332
    #                     a.LastConductanceNMDA[p][tp][sourceneuron,tn] *= np.exp(-a.timesincelastspike[p][sourceneuron] / a.Tau_NMDA[tp][tn])
    #                     a.LS_NMDA[tp][tn] += pathway_strength * ALPHA * (1. - a.LastConductanceNMDA[p][tp][sourceneuron,tn])
    #                     a.LastConductanceNMDA[p][tp][sourceneuron,tn] += ALPHA * (1. - a.LastConductanceNMDA[p][tp][sourceneuron,tn])
    #
    #             # dopaminergic learning
    #             #if a.dpmn_cortex[p][sourceneuron] > 0 and a.dpmn_type[tp][tn] > 0:
    #             #    a.dpmn_XPRE[tp][tn] = 1

    for popid in range(len(popdata)):
        a.cond[popid] = (a.V[popid] < a.V_h[popid]).astype(int)
        # true (cond = 1)
        a.h[popid] = a.h[popid] + a.cond[popid] * a.dt * (1 - a.h[popid]) / a.tauhp[popid]
        # false (cond = 0)
        a.h[popid] = a.h[popid] + (1 - a.cond[popid]) * a.dt * (-a.h[popid]) / a.tauhm[popid]
        # mix
        a.g_rb[popid] = a.g_T[popid] * a.h[popid] * (1 - a.cond[popid])

    for popid in range(len(popdata)):
        # 0 = 1st continue, 1 = proceed
        a.cond[popid] = (a.V[popid] <= a.Threshold[popid]).astype(int)
        a.V[popid] -= (a.V[popid] - a.ResetPot[popid]) * (1 - a.cond[popid])
        # 0 = 1st or 2nd continue, 1 = proceed
        a.cond[popid] = a.cond[popid] * (a.RefrState[popid] == 0).astype(int)
        a.RefrState[popid] -= np.sign(a.RefrState[popid]) * (1 - a.cond[popid])

        a.g_adr[popid] = a.g_adr_max[popid] / (1 + np.exp((a.V[popid]-a.Vadr_h[popid]) / a.Vadr_s[popid]))

        a.dv[popid] = a.V[popid] + 55
        a.tau_n[popid] = a.tau_k_max[popid] / (np.exp(-1 * a.dv[popid] / 30) + np.exp(a.dv[popid] / 30))
        a.n_inif[popid] = 1 / (1 + np.exp(-(a.V[popid] - a.Vk_h[popid]) / a.Vk_s[popid]))
        a.n_k[popid] = a.n_k[popid] + a.cond[popid] * -a.dt / a.tau_n[popid] * (a.n_k[popid] - a.n_inif[popid])
        a.g_k[popid] = a.g_k_max[popid] * a.n_k[popid]

        a.V[popid] = a.V[popid] + a.cond[popid] * -a.dt * (1 / a.Taum[popid] * (a.V[popid] - a.RestPot[popid]) + a.Ca[popid] * a.g_ahp[popid] / a.C[popid] * 0.001 * (a.V[popid] - a.Vk[popid]) + a.g_adr[popid] / a.C[popid] * (a.V[popid] - a.ADRRevPot[popid]) + a.g_k[popid] / a.C[popid] * (a.V[popid] - a.ADRRevPot[popid]) + a.g_rb[popid] / a.C[popid] * (a.V[popid] - a.V_T[popid]))
        a.Ca[popid] = a.Ca[popid] - a.cond[popid] * a.Ca[popid] * a.dt / a.Tau_ca[popid]

        a.Vaux[popid] = np.minimum(a.V[popid],a.Threshold[popid])

        a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_NMDA[popid] - a.Vaux[popid]) * .001 * (a.LS_NMDA[popid] + a.ExtS_NMDA[popid]) / a.C[popid] / (1. + np.exp(-0.062 * a.Vaux[popid] / 3.57))
        a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_AMPA[popid] - a.Vaux[popid]) * .001 * (a.LS_AMPA[popid] + a.ExtS_AMPA[popid]) / a.C[popid]
        a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_GABA[popid] - a.Vaux[popid]) * .001 * (a.LS_GABA[popid] + a.ExtS_GABA[popid]) / a.C[popid]

    for popid in range(len(popdata)):
        newspikes[popid] = list(np.nonzero(a.V[popid] > a.Threshold[popid])[0])
        for neuron in newspikes[popid]:
            a.V[popid][neuron] = 0
            a.Ca[popid][neuron] += a.alpha_ca[popid][neuron]
            a.RefrState[popid][neuron] = 10
            a.Ptimesincelastspike[popid][neuron] = a.timesincelastspike[popid][neuron]
            a.timesincelastspike[popid][neuron] = 0
            #a.dpmn_XPOST[popid] = spikes

    a.spikes = newspikes

    # for p in range(Npop):
    #     for i in range(len(a.N[p])):
    #         g_rb = 0
    #         if a.V[p][i] < a.V_h[p][i]:
    #             a.h[p][i] += (1.0 - a.h[p][i]) * a.dt / a.tauhp[p][i]
    #         else:
    #             a.h[p][i] += -a.h[p][i] * a.dt / a.tauhm[p][i]
    #             g_rb = a.g_T[p][i] * a.h[p][i]
    #
    #         if a.V[p][i] > a.Threshold[p][i]:
    #             a.V[p][i] = a.ResetPot[p][i]
    #             a.RefrState[p][i] -= 1
    #             continue
    #
    #         if a.RefrState[p][i] > 0:
    #             a.RefrState[p][i] -= 1
    #             continue
    #
    #         g_adr = 0
    #         if a.g_adr_max[p][i] != 0:
    #             g_adr = a.g_adr_max[p][i] / (1 + np.exp((a.V[p][i] - a.Vadr_h[p][i]) / a.Vadr_s[p][i]))
    #
    #         g_k = 0
    #         if a.g_k_max[p][i] != 0:
    #             tau_max = a.tau_k_max[p][i]
    #             dv = a.V[p][i] + 55.0
    #             tau_n = tau_max / (np.exp(-1 * dv / 30) + np.exp(dv / 30))
    #             n_inif = 1 / (1 + np.exp(-(a.V[p][i] - a.Vk_h[p][i]) / a.Vk_s[p][i]))
    #             a.n_k[p][i] += -a.dt / tau_n * (a.n_k[p][i] - n_inif)
    #             g_k = a.g_k_max[p][i] * a.n_k[p][i]
    #
    #         if a.g_ahp[p][i] == 0:
    #             a.V[p][i] += -a.dt * (1 / a.Taum[p][i] * (a.V[p][i] - a.RestPot[p][i]) + g_adr / a.C[p][i] * (a.V[p][i] - a.ADRRevPot[p][i]) + g_k / a.C[p][i] * (a.V[p][i] - a.ADRRevPot[p][i]) + g_rb / a.C[p][i] * (a.V[p][i] - a.V_T[p][i]))
    #         else:
    #             a.V[p][i] += -a.dt * (1 / a.Taum[p][i] * (a.V[p][i] - a.RestPot[p][i]) +  a.Ca[p][i] * a.g_ahp[p][i] / a.C[p][i] * 0.001 * (a.V[p][i] - a.Vk[p][i]) + g_adr / a.C[p][i] * (a.V[p][i] - a.ADRRevPot[p][i]) + g_k / a.C[p][i] * (a.V[p][i] - a.ADRRevPot[p][i]) + g_rb / a.C[p][i] * (a.V[p][i] - a.V_T[p][i]))
    #
    #         a.Ca[p][i] += -a.Ca[p][i] * a.dt / a.Tau_ca[p][i]
    #
    #         Vaux = a.V[p][i]
    #         if Vaux > a.Threshold[p][i]:
    #             Vaux = a.Threshold[p][i]
    #
    #         # AMPA
    #         a.V[p][i] += a.dt * (a.RevPot_AMPA[p][i] - Vaux) * .001 * (a.LS_AMPA[p][i] + a.ExtS_AMPA[p][i]) / a.C[p][i]
    #         # GABA
    #         a.V[p][i] += a.dt * (a.RevPot_GABA[p][i] - Vaux) * .001 * (a.LS_GABA[p][i] + a.ExtS_GABA[p][i]) / a.C[p][i]
    #         # NMDA
    #         a.V[p][i] += a.dt * (a.RevPot_NMDA[p][i] - Vaux) * .001 * (a.LS_NMDA[p][i] + a.ExtS_NMDA[p][i]) / a.C[p][i] / (1. + np.exp(-0.062 * Vaux / 3.57))
    #
    #         if a.V[p][i] > a.Threshold[p][i]:
    #             #print("a spike occured!")
    #             newspikes[p].append(i)
    #
    #             a.V[p][i] = a.ResetPot[p][i]
    #             a.timesincelastspike[p][i] = 0
    #
    #             a.V[p][i] = 0
    #             a.RefrState[p][i] = 10 # 2/dt
    #
    #             a.Ca[p][i] += a.alpha_ca[p][i]
    #
    # a.spikes = newspikes

    for popid in range(len(popdata)):
        a.rollingbuffer[popid][a.bufferpointer] = len(a.spikes[popid])
    a.bufferpointer += 1
    if a.bufferpointer >= a.bufferlength:
        a.bufferpointer = 0