"""Example for plotting the temperature_raw time-series"""
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sgdm.osdba import ls_dbas
import glidertools as gt
import cmocean.cm as cmo
import datetime
from sgdm import Dba
from pprint import pprint as pp

log_level = getattr(logging, 'INFO')
log_format = '%(asctime)s:%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
logging.basicConfig(format=log_format, level=log_level)

dt0 = datetime.datetime(2017, 4, 24, 0, 0, 0)
dt1 = datetime.datetime(2017, 4, 25, 0, 0, 0)

# Github repo location
dba_dir = '../data/ru28-20170424T1310'
# localhost location
dba_dir = '/Users/kerfoot/datasets/gliders/rucool/deployments/2017/ru28-20170424T1310/data/in/ascii/dbd'
dbas = ls_dbas(dba_dir, dt0=dt0, dt1=dt1)

process_gps = True
process_ctd = True
index_profiles = True

start_time = datetime.datetime.now()
dba = Dba(dbas, keep_gld_dups=False, gps=process_gps, ctd=process_ctd, profiles=index_profiles)
end_time = datetime.datetime.now()

vmin = 5
vmax = 10

ax = gt.plot.scatter(dba.data.index, dba.data.depth_raw, dba.data.temperature_raw, cmap=cmo.thermal, robust=True)
# Format the x axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# Center the x-axis tick labels and rotate
for xlabel in ax.xaxis.get_ticklabels():
    xlabel.set(rotation=0, horizontalalignment='center')

cb = ax.get_figure().axes[1]

cb.set_ylabel('{:}'.format(dba.column_defs['temperature_raw']['attrs']['units']))

# Title the plot
ax.set_title('temperature_raw: {:} - {:}'.format(dba.profiles.start_time.min().strftime('%Y-%m-%dT%H:%MZ'),
                                                 dba.profiles.end_time.max().strftime('%Y-%m-%dT%H:%MZ')))

plt.show()
