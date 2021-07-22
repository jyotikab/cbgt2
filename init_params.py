from frontendhelpers import *
import copy


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

    if params is None:
        return celldefaults

    else:
        celldefaults_modified = ModifyViaSelector(celldefaults_modified, params)
        return celldefaults_modified     

    
def helper_popspecific(pops=dict()):

    
    popspecific = {'LIP': {'N': 204},
                   'FSI': {'C': 0.2, 'Taum': 10},
                   # should be 10 but was 20 due to bug
                   'GPeP': {'N': 750, 'g_T': 0.06, 'Taum': 20},
                   'STNE': {'N': 750, 'g_T': 0.06},
                   'LIPI': {'N': 186, 'C': 0.2, 'Taum': 10},
                   'Th': {'Taum': 27.78}}

    if len(pops) ==0:
        return popspecific
    else:
        for key in pops.keys():
            for item in pops[key].keys():
                popspecific_modified[key][item] = pops[key][item]

        return popspecific_modified


def helper_receptor(receps=None):

    receptordefaults = ParamSet('receptordefaults', {'Tau_AMPA': 2,
                                                     'RevPot_AMPA': 0,
                                                     'Tau_GABA': 5,
                                                     'RevPot_GABA': -70,
                                                     'Tau_NMDA': 100,
                                                     'RevPot_NMDA': 0, })

    if receps is None:
        return receptordefaults
    else:
        receptordefaults_modified = ModifyViaSelector(receptordefaults_modified, receps)
        return receptordefaults_modified


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

    if len(base) !=0:
        return basestim
    else:
        for key in base.keys():
            for item in base[key].keys():
                basestim_modified[key][item] = base[key][item]

        return basestim_modified


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

    if dpmns is None:
        return dpmndefaults
    else:
        dpmnsdefaults_modified = ModifyViaSelector(dpmndefaults_modified, dpmns)
        return dpmndefaults_modified


def helper_d1(d1=None):

    d1defaults = ParamSet('d1defaults', {'dpmn_type': 1,
                                         'dpmn_alphaw': 55 / 3.0,  # ???
                                         'dpmn_a': 1.0,
                                         'dpmn_b': 0.1,
                                         'dpmn_c': 0.05, })
    if d1 is None:
        return d1defaults
    else:
        d1defaults_modified = ModifyViaSelector(d1defaults_modified, d1)
        return d1defaults_modified


def helper_d2(d2=None):

    d2defaults = ParamSet('d2defaults', {'dpmn_type': 2,
                                         'dpmn_alphaw': -45 / 3.0,
                                         'dpmn_a': 0.5,
                                         'dpmn_b': 0.005,
                                         'dpmn_c': 0.05, })
    if d2 is None:
        return d2defaults
    else:
        d2defaults_modified = ModifyViaSelector(d2defaults_modified, d2)
        return d2defaults_modified


def helper_actionchannels(channels=None):

    actionchannels = ParamSet('actionchannels', {'action': [1, 2]},)

    if channels is None:
        return actionchannels
    else:
        actionchannels_modified = ModifyViaSelector(actionchannels_modified, channels)
        return actionchannels_modified
