import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    print("matplotlib not available, check_nominate is not available")

class idpt_dataframe(object):

    def __init__(self, payload, ret, icpsrs = None):
        if icpsrs:
            self.df = pd.DataFrame(index = icpsrs,
                                   columns = ['start_dim1', 'start_dim2', 'end_dim1', 'end_dim2'])
        else:
            self.df = pd.DataFrame(index = ret['idpt'].keys(),
                                   columns = ['start_dim1', 'start_dim2', 'end_dim1', 'end_dim2'])

        for i in self.df.index.values:
            start = payload['idpt'][i]
            end = ret['idpt'][i]['idpt']
            self.df.set_value(i, 'start_dim1', start[0])
            self.df.set_value(i, 'start_dim2', start[1])
            self.df.set_value(i, 'end_dim1', end[0])
            self.df.set_value(i, 'end_dim2', end[1])

        def onedim_cancor(self, dim = 1):
            y = self.df[:,dim-1]
            lr = sp.stats.linregress(x = self.df[['end_dim1', 'end_dim2']],
                                     y = y)
            return lr
            

def payload_idpt_matrix(ret):
    X = None
    for i,v in enumerate(ret):
        if not X:
            X = np.array([v['idpt']])
        else:
            X = np.concatenate((X, [v['idpt']]), axis = 0)

    return X

def canonical_corr_idpt(payload, ret, icpsrs = None):
    X = None
    for i,v in enumerate(ret):
        if not X:
            X = np.array([v['idpt']])
        else:
            X = np.concatenate((X, [v['idpt']]), axis = 0)
