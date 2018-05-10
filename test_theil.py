import pandas as pd
import numpy as np
import treeTheil as tT
import matplotlib.pyplot as plt

bar = pd.DataFrame(pd.read_csv('test_data_multiyear.csv'))
levels = ['year', 'city', 'firm']
groups = ['white', 'black']
ytargs = "bar[bar['year'] == year], %s, %s" % (levels, groups)
years = np.array([year for year in bar.year.unique()])
theils = np.array([tT.theil(tT.theilTree(ytargs), str(year), 1)
                   for year in bar.year.unique()])
btws = np.array([tT.btw_theil(tT.theilTree(ytargs), str(year))
                 for year in bar.year.unique()])
wins = np.array([[tT.win_theil(tT.theilTree(ytargs), '%s|%s' %
                               (year, city), 1)
                  for year in bar.year.unique()]
                 for city in bar.city.unique()])
firmxs = np.array([[tT.xwin_theil(tT.theilTree(ytargs), firm)
                    for year in bar.year.unique()]
                   for firm in bar.firm.unique()])

plt.plot(years, theils)
plt.show()

for city in wins:
    plt.plot(years, city)
plt.show()

cumwins = np.cumsum(wins, axis=0)
cumbtws = cumwins[2, :] + btws
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.fill_between(years, 0, cumwins[0, :])
ax1.fill_between(years, cumwins[0, :], cumwins[1, :])
ax1.fill_between(years, cumwins[1, :], cumwins[2, :])
ax1.fill_between(years, cumwins[2, :], cumbtws)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.annotate(r'New York', xy=(1981.5, .004))
plt.annotate(r'Chicago', xy=(1981.5, .02))
plt.annotate(r'Philadelphia', xy=(1981.5, .05))
plt.annotate(r'Between Cities', xy=(1981.5, .08))
plt.xticks([1980, 1981, 1982])
plt.show()
