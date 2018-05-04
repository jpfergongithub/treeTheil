import pandas as pd
import numpy as np
import treeTheil as tT

bar = pd.DataFrame(pd.read_csv('test_data_multiyear.csv'))
levels = ['year', 'city', 'firm']
groups = ['white', 'black']
theils = np.array([(year, tT.theil(tT.theil_tree(bar[bar['year'] == year],
                                   levels, groups), str(year), 1))
                  for year in bar.year.unique()])
