import pandas as pd
import numpy as np
import treeTheil as tT
import matplotlib.pyplot as plt

bar = pd.DataFrame(pd.read_csv('test_data_multiyear.csv'))
levels = ['year', 'city', 'firm']
groups = ['white', 'black']
years = np.array([year for year in bar.year.unique()])
theils = np.array([tT.theil(tT.theilTree(bar[bar['year'] == year],
                                         levels, groups), str(year), 1)
                  for year in bar.year.unique()])
btws = np.array([tT.btw_theil(tT.theilTree(bar[bar['year'] == year],
                                           levels, groups), str(year))
                for year in bar.year.unique()])
wins = np.array([[tT.win_theil(tT.theilTree(bar[bar['year'] == year],
                                            levels, groups), '%s|%s' % (
                                                year, city))
                  for year in bar.year.unique()]
                 for city in bar.city.unique()])
firmxs = np.array([[tT.cross_theil(tT.theilTree(bar[bar['year'] ==
                                                    year], levels, groups),
                                   firm)
                    for year in bar.year.unique()]
                   for firm in bar.firm.unique()])
