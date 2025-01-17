from frontendhelpers import *
from init_params import *
import pandas as pd


# ----------------------  helper_popconstruct FUNCTION  ------------------
# helper_popconstruct sets for each population the corrisponding specific
# parameters with either the defaults or the values passed as arguments
# (through the functions implemented in init_params.py)

# inputs:   channels = action channels related parameters - through helper_actionchannels (init_params.py)
#           popspecific = population specific parameters dictionary - through helper_popspecific (init_params.py)
#           celldefaults = neuron parameters, which are either default values or values set by the user - through helper_cellparams (init_params.py)
#           receptordefaults = receptor specific parameters dictionary - through helper_receptor (init_paras.py)
#           basestim = base stimulus parameter dictionary - through helper_basestim (init_params.py)
#           dpmn_defaults = dopamine related parameters dictionary - through helper_dpmn (init_params.py)
#           d1defaults = dopamine related parameters for D1-MSN - through helper_d1 (init_params.py)
# d2defaults = dopamine related parameters for D2-MSN - through helper_d2
# (init_params.py)

# outputs:  popdata = populations dataframe, each one containing the
# corresponding specific parameters

def helper_popconstruct(
        channels,
        popspecific,
        celldefaults,
        receptordefaults,
        basestim,
        dpmndefaults,
        d1defaults,
        d2defaults):

    popdata = pd.DataFrame()

    popdata['name'] = [
        'GPi',
        'STNE',
        'GPeP',
        'D1STR',
        'D2STR',
        'LIP',
        'Th',
        'FSI',
        'LIPI',
    ]
    popdata = trace(popdata, 'init')

    popdata = ModifyViaSelector(popdata, channels, SelName(
        ['GPi', 'STNE', 'GPeP', 'D1STR', 'D2STR', 'LIP', 'Th']))

    popdata = ModifyViaSelector(popdata, celldefaults)

    for key, data in popspecific.items():
        params = ParamSet('popspecific', data)
        popdata = ModifyViaSelector(popdata, params, SelName(key))

    popdata = ModifyViaSelector(popdata, receptordefaults)

    for key, data in basestim.items():
        params = ParamSet('basestim', data)
        popdata = ModifyViaSelector(popdata, params, SelName(key))

    popdata = ModifyViaSelector(
        popdata, dpmndefaults, SelName(['D1STR', 'D2STR']))
    popdata = ModifyViaSelector(popdata, d1defaults, SelName('D1STR'))
    popdata = ModifyViaSelector(popdata, d2defaults, SelName('D2STR'))

    return popdata

# ----------------------  helper_poppathways FUNCTION  -------------------
# helper_poppathays sets for each connection between populations the
# corrisponding specific parameters with either the defaults or the values
# passed

# inputs:  popdata = populations dataframe, each one containing the corresponding specific parameters
# newpathways = connectivity parameters sent as argument. This is a
# dataframe and allows setting only a subset of parameters.

# outputs:  pathways = dataframe that defines the source and the
# destination of each connection, which is the type of receptor involved,
# the probability of connection and efficiency


def helper_poppathways(popdata, newpathways=None):

    if newpathways is None:
        newpathways = pd.DataFrame()

    dpmn_ratio = 0.5
    dpmn_implied = 0.7

    # broken?
    # simplepathways = pd.DataFrame(
    #     [
    #         ['LIP', 'D1STR', 'AMPA', 'syn', 1, 0.027 * dpmn_ratio / dpmn_implied, True],
    #         ['LIP', 'D1STR', 'NMDA', 'syn', 1, 0.027 * (1 - dpmn_ratio), False],
    #         ['LIP', 'D2STR', 'AMPA', 'syn', 1, 0.027 * dpmn_ratio / dpmn_implied, True],
    #         ['LIP', 'D2STR', 'NMDA', 'syn', 1, 0.027 * (1 - dpmn_ratio), False],
    #         ['LIP', 'FSI', 'AMPA', 'all', 1, 0.198, False],
    #         ['LIP', 'Th', 'AMPA', 'all', 1, 0.035, False],
    #         ['LIP', 'Th', 'NMDA', 'all', 1, 0.035, False],
    #
    #         ['D1STR', 'D1STR', 'GABA', 'syn', 0.45, 0.28, False],
    #         ['D1STR', 'D2STR', 'GABA', 'syn', 0.45, 0.28, False],
    #         ['D1STR', 'GPi', 'GABA', 'syn', 1, 2.09, False],
    #
    #         ['D2STR', 'D2STR', 'GABA', 'syn', 0.45, 0.28, False],
    #         ['D2STR', 'D1STR', 'GABA', 'syn', 0.5, 0.28, False],
    #         ['D2STR', 'GPeP', 'GABA', 'syn', 1, 4.07, False],
    #
    #         ['FSI', 'FSI', 'GABA', 'all', 1, 3.25833, False],
    #         ['FSI', 'D1STR', 'GABA', 'all', 1, 1.7776, False],
    #         ['FSI', 'D2STR', 'GABA', 'all', 1, 1.669867, False],
    #
    #         ['GPeP', 'GPeP', 'GABA', 'all', 0.0667, 1.75, False],
    #         ['GPeP', 'STNE', 'GABA', 'syn', 0.0667, 0.35, False],
    #         ['GPeP', 'GPi', 'GABA', 'syn', 1, 0.06, False],
    #
    #         ['STNE', 'GPeP', 'AMPA', 'syn', 0.161668, 0.07, False],
    #         ['STNE', 'GPeP', 'NMDA', 'syn', 0.161668, 1.51, False],
    #         ['STNE', 'GPi', 'NMDA', 'all', 1, 0.038, False],
    #
    #         ['GPi', 'Th', 'GABA', 'syn', 1, 0.3315, False],
    #
    #         ['Th', 'D1STR', 'AMPA', 'syn', 1, 0.3825, False],
    #         ['Th', 'D2STR', 'AMPA', 'syn', 1, 0.3825, False],
    #         ['Th', 'FSI', 'AMPA', 'all', 0.8334, 0.1, False],
    #         ['Th', 'LIP', 'NMDA', 'all', 0.8334, 0.03, False],
    #
    #         # ramping ctx
    #
    #         ['LIP', 'LIP', 'AMPA', 'all', 0.4335, 0.0127, False],
    #         ['LIP', 'LIP', 'NMDA', 'all', 0.4335, 0.15, False],
    #         ['LIP', 'LIPI', 'AMPA', 'all', 0.241667, 0.113, False],
    #         ['LIP', 'LIPI', 'NMDA', 'all', 0.241667, 0.525, False],
    #
    #         ['LIPI', 'LIP', 'GABA', 'all', 1, 1.75, False],
    #         ['LIPI', 'LIPI', 'GABA', 'all', 1, 3.58335, False],
    #
    #         ['Th', 'LIPI', 'NMDA', 'all', 0.8334, 0.015, False],
    #
    #     ],
    #     columns=['src', 'dest', 'receptor', 'type', 'con', 'eff', 'plastic']
    # )

    # mixture based on 2019 params
    # simplepathways = pd.DataFrame(
    #    [
    #        ['LIP', 'D1STR', 'AMPA', 'syn', 1, 0.297, True],#
    #        ['LIP', 'D1STR', 'NMDA', 'syn', 1, 0.297, False],#
    #        ['LIP', 'D2STR', 'AMPA', 'syn', 1, 0.300, True],#
    #        ['LIP', 'D2STR', 'NMDA', 'syn', 1, 0.300, False],#
    #        ['LIP', 'FSI', 'AMPA', 'all', 1, 0.2475, False],#
    #        ['LIP', 'Th', 'AMPA', 'syn', 1, 0.035, False],#
    #        ['LIP', 'Th', 'NMDA', 'syn', 1, 0.035, False],#

    #        ['D1STR', 'D1STR', 'GABA', 'syn', 0.45, 0.28, False],#
    #        ['D1STR', 'D2STR', 'GABA', 'syn', 0.45, 0.28, False],#
    #        ['D1STR', 'GPi', 'GABA', 'syn', 1, 2.09, False],#

    #        ['D2STR', 'D2STR', 'GABA', 'syn', 0.45, 0.28, False],#
    #        ['D2STR', 'D1STR', 'GABA', 'syn', 0.5, 0.28, False],#
    #        ['D2STR', 'GPeP', 'GABA', 'syn', 1, 4.07, False],#

            
    #        ['FSI', 'FSI', 'GABA', 'all', 1, 3.25833, False],#
    #        ['FSI', 'D1STR', 'GABA', 'all', 1, 2.706, False],#
    #        ['FSI', 'D2STR', 'GABA', 'all', 1, 2.542, False],#

    #        ['GPeP', 'GPeP', 'GABA', 'all', 0.0667, 1.5, False],#
    #        ['GPeP', 'STNE', 'GABA', 'syn', 0.0667, 0.4, False],#
    #        ['GPeP', 'GPi', 'GABA', 'syn', 1, 0.04, False],#

    #        ['STNE', 'GPeP', 'AMPA', 'syn', 0.161666, 0.07, False],#
    #        ['STNE', 'GPeP', 'NMDA', 'syn', 0.161666, 4., False],#
    #        ['STNE', 'GPi', 'NMDA', 'all', 1, 0.108, False],#

    #        ['GPi', 'Th', 'GABA', 'syn', 1, 0.1898333, False],#

    #        ['Th', 'D1STR', 'AMPA', 'syn', 1, 0.51, False],#
    #        ['Th', 'D2STR', 'AMPA', 'syn', 1, 0.51, False],#
    #        ['Th', 'FSI', 'AMPA', 'all', 0.8334, 0.3, False],#
    #        ['Th', 'LIP', 'NMDA', 'all', 0.8334, 0.02, False],#

            # ramping ctx

    #        ['LIP', 'LIP', 'AMPA', 'syn', 0.4335, 0.0127, False],#
    #        ['LIP', 'LIP', 'NMDA', 'syn', 0.4335, 0.15, False],#
    #        ['LIP', 'LIPI', 'AMPA', 'all', 0.241667, 0.013, False],#
    #        ['LIP', 'LIPI', 'NMDA', 'all', 0.241667, 0.125, False],#

    #        ['LIPI', 'LIP', 'GABA', 'all', 1, 1.75, False],#
    #        ['LIPI', 'LIPI', 'GABA', 'all', 1, 3.58333, False],#

    #        ['Th', 'LIPI', 'NMDA', 'all', 0.8334, 0.015, False],#

    #   ],
    #    columns=['src', 'dest', 'receptor', 'type', 'con', 'eff', 'plastic']
    #)
    
    # phenotype study's values
    simplepathways = pd.DataFrame(
        [
            ['LIP', 'D1STR', 'AMPA', 'syn', 1, 0.027, True],#
            ['LIP', 'D1STR', 'NMDA', 'syn', 1, 0.027, False],#
            ['LIP', 'D2STR', 'AMPA', 'syn', 1, 0.027, True],#
            ['LIP', 'D2STR', 'NMDA', 'syn', 1, 0.027, False],#
            ['LIP', 'FSI', 'AMPA', 'all', 1, 0.198, False],#
            ['LIP', 'Th', 'AMPA', 'syn', 1, 0.035, False],#
            ['LIP', 'Th', 'NMDA', 'syn', 1, 0.035, False],#

            ['D1STR', 'D1STR', 'GABA', 'syn', 0.45, 0.28, False],#
            ['D1STR', 'D2STR', 'GABA', 'syn', 0.45, 0.28, False],#
            ['D1STR', 'GPi', 'GABA', 'syn', 1, 2.09, False],#

            ['D2STR', 'D2STR', 'GABA', 'syn', 0.45, 0.28, False],#
            ['D2STR', 'D1STR', 'GABA', 'syn', 0.5, 0.28, False],#
            ['D2STR', 'GPeP', 'GABA', 'syn', 1, 4.07, False],#

            
            ['FSI', 'FSI', 'GABA', 'all', 1, 3.25833, False],#
            ['FSI', 'D1STR', 'GABA', 'all', 1, 1.77760, False],#
            ['FSI', 'D2STR', 'GABA', 'all', 1, 1.66987, False],#

            ['GPeP', 'GPeP', 'GABA', 'all', 0.0667, 1.75, False],#
            ['GPeP', 'STNE', 'GABA', 'syn', 0.0667, 0.35, False],#
            ['GPeP', 'GPi', 'GABA', 'syn', 1, 0.06, False],#

            ['STNE', 'GPeP', 'AMPA', 'syn', 0.161666, 0.07, False],#
            ['STNE', 'GPeP', 'NMDA', 'syn', 0.161666, 1.51, False],#
            ['STNE', 'GPi', 'NMDA', 'all', 1, 0.0380, False],#

            ['GPi', 'Th', 'GABA', 'syn', 1, 0.3315, False],#

            ['Th', 'D1STR', 'AMPA', 'syn', 1, 0.3825, False],#
            ['Th', 'D2STR', 'AMPA', 'syn', 1, 0.3825, False],#
            ['Th', 'FSI', 'AMPA', 'all', 0.8334, 0.1, False],#
            ['Th', 'LIP', 'NMDA', 'all', 0.8334, 0.03, False],#

            # ramping ctx

            ['LIP', 'LIP', 'AMPA', 'syn', 0.4335, 0.0127, False],#
            ['LIP', 'LIP', 'NMDA', 'syn', 0.4335, 0.15, False],#
            ['LIP', 'LIPI', 'AMPA', 'all', 0.241667, 0.113, False],#
            ['LIP', 'LIPI', 'NMDA', 'all', 0.241667, 0.525, False],#

            ['LIPI', 'LIP', 'GABA', 'all', 1, 1.75, False],#
            ['LIPI', 'LIPI', 'GABA', 'all', 1, 3.58333, False],#

            ['Th', 'LIPI', 'NMDA', 'all', 0.8334, 0.015, False],#

        ],
        columns=['src', 'dest', 'receptor', 'type', 'con', 'eff', 'plastic']
    )

    # ejn params
    # simplepathways = pd.DataFrame(
    #     [
    #         ['LIP', 'D1STR', 'AMPA', 'syn', 1, 0.189, True],
    #         ['LIP', 'D1STR', 'NMDA', 'syn', 1, 0.189, False],
    #         ['LIP', 'D2STR', 'AMPA', 'syn', 1, 0.189, True],
    #         ['LIP', 'D2STR', 'NMDA', 'syn', 1, 0.189, False],
    #         ['LIP', 'FSI', 'AMPA', 'all', 1, 0.198, False],
    #         ['LIP', 'Th', 'AMPA', 'all', 1, 0.035, False],
    #         ['LIP', 'Th', 'NMDA', 'all', 1, 0.035, False],
    #
    #         ['D1STR', 'D1STR', 'GABA', 'syn', 0.45, 0.28, False],
    #         ['D1STR', 'D2STR', 'GABA', 'syn', 0.45, 0.28, False],
    #         ['D1STR', 'GPi', 'GABA', 'syn', 1, 2.09, False],
    #
    #         ['D2STR', 'D2STR', 'GABA', 'syn', 0.45, 0.28, False],
    #         ['D2STR', 'D1STR', 'GABA', 'syn', 0.5, 0.28, False],
    #         ['D2STR', 'GPeP', 'GABA', 'syn', 1, 4.07, False],
    #
    #         ['FSI', 'FSI', 'GABA', 'all', 1, 3.25833, False],
    #         ['FSI', 'D1STR', 'GABA', 'all', 1, 2.1648, False],
    #         ['FSI', 'D2STR', 'GABA', 'all', 1, 2.0336, False],
    #
    #         ['GPeP', 'GPeP', 'GABA', 'all', 0.0667, 1.5, False],
    #         ['GPeP', 'STNE', 'GABA', 'syn', 0.0667, 0.4, False],
    #         ['GPeP', 'GPi', 'GABA', 'syn', 1, 0.04, False],
    #
    #         ['STNE', 'GPeP', 'AMPA', 'syn', 0.161666, 0.07, False],
    #         ['STNE', 'GPeP', 'NMDA', 'syn', 0.161666, 4., False],
    #         ['STNE', 'GPi', 'NMDA', 'all', 1, 0.108, False],
    #
    #         ['GPi', 'Th', 'GABA', 'syn', 1, 0.1898333, False],
    #
    #         ['Th', 'D1STR', 'AMPA', 'syn', 1, 0.3825, False],
    #         ['Th', 'D2STR', 'AMPA', 'syn', 1, 0.3825, False],
    #         ['Th', 'FSI', 'AMPA', 'all', 0.8333, 0.3, False],
    #         ['Th', 'LIP', 'NMDA', 'all', 0.8333, 0.02, False],
    #
    #         # ramping ctx
    #
    #         ['LIP', 'LIP', 'AMPA', 'all', 0.4335, 0.0127, False],
    #         ['LIP', 'LIP', 'NMDA', 'all', 0.4335, 0.15, False],
    #         ['LIP', 'LIPI', 'AMPA', 'all', 0.241667, 0.013, False],
    #         ['LIP', 'LIPI', 'NMDA', 'all', 0.241667, 0.125, False],
    #
    #         ['LIPI', 'LIP', 'GABA', 'all', 1, 1.75, False],
    #         ['LIPI', 'LIPI', 'GABA', 'all', 1, 3.58333, False],
    #
    #         ['Th', 'LIPI', 'NMDA', 'all', 0.8333, 0.015, False],
    #
    #     ],
    #     columns=['src', 'dest', 'receptor', 'type', 'con', 'eff', 'plastic']
    # )

    simplepathways = trace(simplepathways, 'init')

    if len(newpathways) != 0:

        simplepathways.update(newpathways)

    pathways = simplepathways.copy()
    pathways['biselector'] = None
    for idx, row in pathways.iterrows():
        if row['type'] == 'syn':
            pathways.loc[idx, 'biselector'] = NamePathwaySelector(
                row['src'], row['dest'], 'action')
        elif row['type'] == 'all':
            pathways.loc[idx, 'biselector'] = NamePathwaySelector(
                row['src'], row['dest'])
    pathways = trace(pathways, 'auto')

    return pathways

# ----------------------  helper_connectivity FUNCTION  ------------------
# helper_connectivity sets three connectivity grids defining,
# correspondingly, the probability of connection, the mean synaptic
# efficacy and the plasticity of connections between each population,
# referring only to AMPA receptors. With plasticity of connection it is
# meant whether or not the weights of that connection could ever change -
# particularly, referring to the Ctx-STR connections modulated by AMPA
# receptors.

# inputs:  receptor = which type of receptor involved (AMPA, GABA, NMDA)
#          popdata = populations dataframe, each one containing the corresponding specific parameters
# pathways = dataframe defining the source and the destination of each
# connection, which is the type of receptor involved, the probability of
# connection and efficiency

# outputs:  connectivity = connectivity matrix defining the probability of connection between each population
#           meanEff = connectivity matrix defining the mean synaptic efficacy of each connection between populations
# plasticity = connectivity matrix defining the plasticity of connections
# between each population (true or false, 0 if not applicable)


def helper_connectivity(receptor, popdata, pathways):

    connectiongrid = constructSquareDf(untrace(popdata['name'].tolist()))
    connectiongrid = trace(connectiongrid, 'init')

    connectivity = connectiongrid.copy()
    meanEff = connectiongrid.copy()
    plasticity = connectiongrid.copy()

    for idx, row in pathways.iterrows():
        if row['receptor'] == receptor:
            biselector = row['biselector']
            receptor = row['receptor']
            con = row['con']
            eff = row['eff']
            plastic = row['plastic']

            connectivity = FillGridSelection(
                connectivity, popdata, biselector, con)
            meanEff = FillGridSelection(
                meanEff, popdata, biselector, eff)
            plasticity = FillGridSelection(
                plasticity, popdata, biselector, plastic)

    return connectivity, meanEff, plasticity


# ----------------------  helper_connectivityAMPA FUNCTION  --------------
# helper_connectivityAMPA sets two connectivity matrices defining the
# probability of connection and the mean synaptic efficacy between each
# population, referring only to AMPA receptors.

# inputs:  popdata = populations dataframe, each one containing the corresponding specific parameters
# pathways = dataframe defining the source and the destination of each
# connection, which is the type of receptor involved, the probability of
# connection and efficiency

# outputs:  connectivity_AMPA = connectivity matrix defining the probability of connection between each population - AMPA receptors
# meanEff_AMPA = connectivity matrix defining the mean synaptic efficacy
# of each connection between populations - AMPA receptors

# def helper_connectivityAMPA(popdata, pathways):
#
#     connectiongrid = constructSquareDf(untrace(popdata['name'].tolist()))
#     connectiongrid = trace(connectiongrid, 'init')
#
#     connectivity_AMPA = connectiongrid.copy()
#     meanEff_AMPA = connectiongrid.copy()
#
#     for idx, row in pathways.iterrows():
#         biselector = row['biselector']
#         receptor = row['receptor']
#         con = row['con']
#         eff = row['eff']
#
#         connectivity_AMPA = FillGridSelection(
#             connectivity_AMPA, popdata, biselector, con)
#         meanEff_AMPA = FillGridSelection(
#             meanEff_AMPA, popdata, biselector, eff)
#
#     return connectivity_AMPA, meanEff_AMPA
#

# ----------------------  helper_connectivityGABA FUNCTION  --------------
# helper_connectivityGABA sets two connectivity matrices defining the
# probability of connection and the mean synaptic efficacy between each
# population, referring only to GABA receptors.

# inputs:  popdata = populations dataframe, each one containing the corresponding specific parameters
# pathways = dataframe defining the source and the destination of each
# connection, which is the type of receptor involved, the probability of
# connection and efficiency

# outputs:  connectivity_GABA = connectivity matrix defining the probability of connection between each population - GABA receptors
# meanEff_GABA = connectivity matrix defining the mean synaptic efficacy
# of each connection between populations - GABA receptors

# def helper_connectivityGABA(popdata, pathways):
#
#     connectiongrid = constructSquareDf(untrace(popdata['name'].tolist()))
#     connectiongrid = trace(connectiongrid, 'init')
#
#     connectivity_GABA = connectiongrid.copy()
#     meanEff_GABA = connectiongrid.copy()
#
#     for idx, row in pathways.iterrows():
#         biselector = row['biselector']
#         receptor = row['receptor']
#         con = row['con']
#         eff = row['eff']
#
#         connectivity_GABA = FillGridSelection(
#             connectivity_GABA, popdata, biselector, con)
#         meanEff_GABA = FillGridSelection(
#             meanEff_GABA, popdata, biselector, eff)
#
#     return connectivity_GABA, meanEff_GABA
#

# ----------------------  helper_connectivityNMDA FUNCTION  --------------
# helper_connectivityNMDA sets two connectivity matrices defining the
# probability of connection and the mean synaptic efficacy between each
# population, referring only to NMDA receptors.

# inputs:  popdata = populations dataframe, each one containing the corresponding specific parameters
# pathways = dataframe defining the source and the destination of each
# connection, which is the type of receptor involved, the probability of
# connection and efficiency

# outputs:  connectivity_NMDA = connectivity matrix defining the probability of connection between each population - NMDA receptors
# meanEff_NMDA = connectivity matrix defining the mean synaptic efficacy
# of each connection between populations - NMDA receptors

# def helper_connectivityNMDA(popdata, pathways):
#
#     connectiongrid = constructSquareDf(untrace(popdata['name'].tolist()))
#     connectiongrid = trace(connectiongrid, 'init')
#
#     connectivity_NMDA = connectiongrid.copy()
#     meanEff_NMDA = connectiongrid.copy()
#
#     for idx, row in pathways.iterrows():
#         biselector = row['biselector']
#         receptor = row['receptor']
#         con = row['con']
#         eff = row['eff']
#
#         connectivity_NMDA = FillGridSelection(
#             connectivity_NMDA, popdata, biselector, con)
#         meanEff_NMDA = FillGridSelection(
#             meanEff_NMDA, popdata, biselector, eff)
#
#     return connectivity_NMDA, meanEff_NMDA
