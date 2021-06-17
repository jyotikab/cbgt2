from popparams import *


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


popdata = ModifyViaSelector(popdata, dpmndefaults, SelName(['D1STR', 'D2STR']))
popdata = ModifyViaSelector(popdata, d1defaults, SelName('D1STR'))
popdata = ModifyViaSelector(popdata, d2defaults, SelName('D2STR'))
