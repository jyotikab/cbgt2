def expandIdByCell(popdata):
    array = []
    for idx, row in popdata.iterrows():
        array.extend([idx] * untrace(row['N']))
    return np.array(array)


def expandParamByCell(popdata, param, fillvalue=np.nan):
    if param not in popdata.columns:
        print(param + " not found, initializing from scratch.")
        return np.ones(popdata.sum()['N']) * fillvalue

    array = []
    for idx, row in popdata.iterrows():
        value = row[param]
        if value.is_nan():
            value = fillvalue
        else:
            value = untrace(value)
        array.extend([value] * untrace(row['N']))
    return np.array(array)


def expandSelectorByCell(popdata, selector):
    array = []
    for idx, row in popdata.iterrows():
        array.extend([selector(row)] * untrace(row['N']))
    return np.array(array)


def createMatrix(popdata):
    totalN = popdata.sum()['N']
    return np.zeros((totalN, totalN))


def expandBiselectorByCell(popdata, biselector):
    matrix = createMatrix(popdata)
    for idx1, row1 in popdata.iterrows():
        for idx2, row2 in popdata.iterrows():
            if biselector(row1, row2):
                maxrow = popdata.cumsum()['N'].loc[idx1]
                minrow = maxrow - popdata['N'].loc[idx1]
                maxcol = popdata.cumsum()['N'].loc[idx2]
                mincol = maxcol - popdata['N'].loc[idx2]
                matrix[minrow:maxrow, mincol:maxcol] = 1
    return matrix


def CreateSynapses(popdata, cons, effs):
    efficacy = createMatrix(popdata)

    for idx1, row1 in popdata.iterrows():
        for idx2, row2 in popdata.iterrows():
            maxrow = popdata.cumsum()['N'].loc[idx1]
            minrow = maxrow - popdata['N'].loc[idx1]
            maxcol = popdata.cumsum()['N'].loc[idx2]
            mincol = maxcol - popdata['N'].loc[idx2]
            eff = effs.iloc[idx1, idx2]
            con = cons.iloc[idx1, idx2]

            if con == 0.0:
                continue

            data = (
                np.random.rand(
                    popdata['N'].loc[idx1],
                    popdata['N'].loc[idx2]) < con).astype(int)
            data = data * untrace(eff)

            efficacy[minrow:maxrow, mincol:maxcol] = data

            print(idx1, idx2, con)
    return efficacy


Efficacy_GABA = CreateSynapses(popdata, Connectivity_GABA, MeanEff_GABA)
Efficacy_AMPA = CreateSynapses(popdata, Connectivity_AMPA, MeanEff_AMPA)
Efficacy_NMDA = CreateSynapses(popdata, Connectivity_NMDA, MeanEff_NMDA)
