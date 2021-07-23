from frontendhelpers import *
import copy
import pdb


def helper_cellparams(params=None):

    print("helper_cellparams")
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
        #cellparams = celldefaults.copy(deep=True)

    print(celldefaults)
    return celldefaults


def helper_popspecific(pops=dict()):

    print("helper_popspecific")
    popspecific = {'LIP': {'N': 204},
                   'FSI': {'C': 0.2, 'Taum': 10},
                   # should be 10 but was 20 due to bug
                   'GPeP': {'N': 750, 'g_T': 0.06, 'Taum': 20},
                   'STNE': {'N': 750, 'g_T': 0.06},
                   'LIPI': {'N': 186, 'C': 0.2, 'Taum': 10},
                   'Th': {'Taum': 27.78}}

    # if len(pops) ==0:
    #    return popspecific
    if len(pops) != 0:
        for key in pops.keys():
            for item in pops[key].keys():
                popspecific[key][item] = pops[key][item]

    print(popspecific)
    return popspecific


def helper_receptor(receps=None):

    print("helper_receptor")
    receptordefaults = ParamSet('receptordefaults', {'Tau_AMPA': 2,
                                                     'RevPot_AMPA': 0,
                                                     'Tau_GABA': 5,
                                                     'RevPot_GABA': -70,
                                                     'Tau_NMDA': 100,
                                                     'RevPot_NMDA': 0, })

    if receps is not None:
        receptordefaults = ModifyViaSelector(receptordefaults, receps)

    print(receptordefaults)
    return receptordefaults


def helper_basestim(base=dict()):

    print("helper_basestim")
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

    if len(base) != 0:
        for key in base.keys():
            for item in base[key].keys():
                basestim[key][item] = base[key][item]

    print(basestim)
    return basestim


def helper_dpmn(dpmns=None):

    print("helper_dpmn")
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

    print(dpmndefaults)
    return dpmndefaults


def helper_d1(d1=None):

    print("helper_d1")
    d1defaults = ParamSet('d1defaults', {'dpmn_type': 1,
                                         'dpmn_alphaw': 55 / 3.0,  # ???
                                         'dpmn_a': 1.0,
                                         'dpmn_b': 0.1,
                                         'dpmn_c': 0.05, })
    if d1 is not None:
        d1defaults = ModifyViaSelector(d1defaults, d1)

    print(d1defaults)
    return d1defaults


def helper_d2(d2=None):

    print("helper_d2")
    d2defaults = ParamSet('d2defaults', {'dpmn_type': 2,
                                         'dpmn_alphaw': -45 / 3.0,
                                         'dpmn_a': 0.5,
                                         'dpmn_b': 0.005,
                                         'dpmn_c': 0.05, })
    if d2 is not None:
        d2defaults = ModifyViaSelector(d2defaults, d2)

    print(d2defaults)
    return d2defaults


def helper_actionchannels(channels=None):
    
    print("helper_actionchannels")
    actionchannels = ParamSet('actionchannels', {'action': [1, 2]},)

    if channels is not None:
        actionchannels = ModifyViaSelector(actionchannels, channels)
    print(actionchannels)
    return actionchannels