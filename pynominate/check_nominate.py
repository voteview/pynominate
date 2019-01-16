import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


class idpt_dataframe(object):

    def __init__(self, payload, ret, icpsrs=None):
        cols = [
            "start_dim1",
            "start_dim2",
            "end_dim1",
            "end_dim2",
            "euclidean_dist",
            "dist_dim1",
            "dist_dim2"
        ]
        if icpsrs:
            self.df = pd.DataFrame(index=icpsrs, columns=cols)
        else:
            self.df = pd.DataFrame(index=ret['idpt'].keys(), columns=cols)

        for i in self.df.index.values:
            start = payload['idpt'][i]
            end = ret['idpt'][i]['idpt']
            euc_dist = np.sum(np.square(np.array(start) - np.array(end)))
            dim1_dist = end[0] - start[0]
            dim2_dist = end[1] - start[1]
            self.df.loc[i, cols] = [
                start[0], 
                start[1], 
                end[0], 
                end[1], 
                euc_dist,
                dim1_dist,
                dim2_dist
            ]

    def dim_coefficients(self):
        reg = linear_model.LinearRegression().fit(
            y=self.df[["end_dim1", "end_dim2"]],
            X=self.df[["start_dim1", "start_dim2"]]
        )
        return reg.coef_

    def plot_dim_changes(self, dim=1):
        plt.plot(self.df["start_dim" + str(dim)], self.df["end_dim" + str(dim)], "o")
        plt.plot([-1, 1], [-1, 1], "k-", lw=0.5)
        plt.xlabel("Start Dim " + str(dim))
        plt.ylabel("End Dim " + str(dim))
        plt.show()
        return None
        
    def plot_distance_histograms(self, n_bins=20):
        fig, axs = plt.subplots(1, 3)
        axs[0].hist(self.df["dist_dim1"].values, bins=n_bins)
        axs[1].hist(self.df["dist_dim2"].values, bins=n_bins)
        axs[2].hist(self.df["euclidean_dist"].values, bins=n_bins)
        axs[0].set_xlabel("Dim 1 Dist.")
        axs[1].set_xlabel("Dim 2 Dist.")
        axs[2].set_xlabel("Euclidean Dist.")
        return None
    
    def top_movers(self, measure="euclidean_dist", n=10):
        return self.df.sort_values(measure, ascending=False).head(n)
        
