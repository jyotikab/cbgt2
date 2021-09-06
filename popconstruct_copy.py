from frontendhelpers import *
from init_params import *
import pandas as pd


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


def helper_poppathways(popdata, newpathways=None):

    if newpathways is None:
        newpathways = pd.DataFrame()

    simplepathways = pd.DataFrame(
        [
            ['LIP', 'D1STR', 'AMPA', 'syn', 1, 0.027, True],
            ['LIP', 'D1STR', 'NMDA', 'syn', 1, 0.027, False],
            ['LIP', 'D2STR', 'AMPA', 'syn', 1, 0.027, True],
            ['LIP', 'D2STR', 'NMDA', 'syn', 1, 0.027, False],
            ['LIP', 'FSI', 'AMPA', 'all', 1, 0.198, False],
            ['LIP', 'Th', 'AMPA', 'all', 1, 0.035, False],
            ['LIP', 'Th', 'NMDA', 'all', 1, 0.035, False],

            ['D1STR', 'D1STR', 'GABA', 'syn', 0.45, 0.28, False],
            ['D1STR', 'D2STR', 'GABA', 'syn', 0.45, 0.28, False],
            ['D1STR', 'GPi', 'GABA', 'syn', 1, 2.09, False],

            ['D2STR', 'D2STR', 'GABA', 'syn', 0.45, 0.28, False],
            ['D2STR', 'D1STR', 'GABA', 'syn', 0.5, 0.28, False],
            ['D2STR', 'GPeP', 'GABA', 'syn', 1, 4.07, False],

            ['FSI', 'FSI', 'GABA', 'all', 1, 3.25833, False],
            ['FSI', 'D1STR', 'GABA', 'all', 1, 1.7776, False],
            ['FSI', 'D2STR', 'GABA', 'all', 1, 1.669867, False],

            ['GPeP', 'GPeP', 'GABA', 'all', 0.0667, 1.75, False],
            ['GPeP', 'STNE', 'GABA', 'syn', 0.0667, 0.35, False],
            ['GPeP', 'GPi', 'GABA', 'syn', 1, 0.06, False],

            ['STNE', 'GPeP', 'AMPA', 'syn', 0.161668, 0.07, False],
            ['STNE', 'GPeP', 'NMDA', 'syn', 0.161668, 1.51, False],
            ['STNE', 'GPi', 'NMDA', 'all', 1, 0.038, False],

            ['GPi', 'Th', 'GABA', 'syn', 1, 0.3315, False],

            ['Th', 'D1STR', 'AMPA', 'syn', 1, 0.3825, False],
            ['Th', 'D2STR', 'AMPA', 'syn', 1, 0.3825, False],
            ['Th', 'FSI', 'AMPA', 'all', 0.8334, 0.1, False],
            ['Th', 'LIP', 'NMDA', 'all', 0.8334, 0.03, False],

            # ramping ctx

            ['LIP', 'LIP', 'AMPA', 'all', 0.4335, 0.0127, False],
            ['LIP', 'LIP', 'NMDA', 'all', 0.4335, 0.15, False],
            ['LIP', 'LIPI', 'AMPA', 'all', 0.241667, 0.113, False],
            ['LIP', 'LIPI', 'NMDA', 'all', 0.241667, 0.525, False],

            ['LIPI', 'LIP', 'GABA', 'all', 1, 1.75, False],
            ['LIPI', 'LIPI', 'GABA', 'all', 1, 3.58335, False],

            ['Th', 'LIPI', 'NMDA', 'all', 0.8334, 0.015, False],

        ],
        columns=['src', 'dest', 'receptor', 'type', 'con', 'eff', 'plastic']
    )
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
#
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
#
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
