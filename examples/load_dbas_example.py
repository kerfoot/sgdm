"""Scratch space for sgdm developments"""

import logging
import matplotlib.pyplot as plt
from sgdm.osdba import ls_dbas
import glidertools as gt
import cmocean.cm as cmo
import datetime
from sgdm import Dba
from pprint import pprint as pp

log_level = getattr(logging, 'DEBUG')
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

elapsed_time = end_time - start_time
print('Load time: {:} seconds'.format(elapsed_time.total_seconds()))
