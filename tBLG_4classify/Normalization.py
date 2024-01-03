import pandas as pd
from matplotlib import pyplot as plt

data_origin = pd.read_csv('./Dataset/tBLG_20-30.csv', sep=",", header=0,index_col=0)

def max_min_normalization(data_value):

    data_nor = (data_origin - data_origin.min(axis=0))/(data_origin.max(axis=0) - data_origin.min(axis=0))
    data_nor.to_csv("./ProcessedData/NormtBLG_20-30.csv")
    print('done')

    plt.plot(data_nor)
    plt.show()

max_min_normalization(data_origin)