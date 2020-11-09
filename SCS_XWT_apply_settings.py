#!/usr/bin/env python
'''File name: Denver-Water_XWT.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 16.04.2018
    Date last modified: 16.04.2018

    ############################################################## 
    Purpos:
    Contains the setup for extreme weather typing (XWT) for
    Denver Water watersheds

'''
from pdb import set_trace as stop
import numpy as np
import os
import pandas as pd
import datetime

    
# ###################################################

dStartDayPR=datetime.datetime(1990, 1, 1, 0)
dStopDayPR=datetime.datetime(2016, 12, 31, 23)
rgdTime = pd.date_range(dStartDayPR, end=dStopDayPR, freq='d')

iMonths=[12, 1, 2]

# ---------
# Setup clustering algorithm
ClusterMeth='hdbscan'  # current options are ['HandK','hdbscan']
ClusterBreakup = 1     # breakes up clusters that are unproportionally large (only for hdbscan)
# ---------

# DENVER WATER REGIONS
sPlotDir='/glade/work/fossell/preevents/XWT_Kates_code/plot_test/'
sDataDir='/glade/work/fossell/preevents/XWT_Kates_code/data_test/'
DW_Regions=['MDW.poly']
sRegion=0
# sRegion=DW_Regions.index(Region)
Region=DW_Regions[sRegion]
sSubregionSCS='/glade/u/home/fossell/MET/met/data/poly/'+DW_Regions[sRegion]

Metrics=['PSS', 'MRD', 'MRR', 'APR', 'PEX']
Dimensions=['Variables', 'Extreme Nr.', 'Domain Size', 'Annual Cycle', 'Smoothing', 'Split Sample', 'Metrics']

# rgsWTvars= ['var151','u',   'v',     'UV', 'tcw',    'q', 'q'   , 'z','cape', '06Shear', 'FLX','cape-shear','cape']
# VarsFullName=['PSL','U850','V850','UV850',  'PW', 'Q850', 'Q500', 'ZG500','CAPE', 'VS06', 'FLX850','CAPE-Shear','MAX_CAPE']
# rgsWTfolders=['/glade/scratch/prein/ERA-Interim/PSL/fin_PSL-sfc_ERA-Interim_12-0_',\
#               '/glade/scratch/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
#               '/glade/scratch/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
#               '/glade/scratch/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
#               '/glade/scratch/prein/ERA-Interim/TCW/fin_TCW-sfc_ERA-Interim_12-0_',\
#               '/glade/scratch/prein/ERA-Interim/Q850/Q850_daymean_',\
#               '/glade/scratch/prein/ERA-Interim/Q500/Q500_daymean_',\
#               '/glade/scratch/prein/ERA-Interim/Z500/Z500_daymean_',\
#               '/glade/scratch/prein/ERA-Interim/CAPE_ECMWF/fin_CAPE-ECMWF-sfc_ERA-Interim_12-9_',\
#               '/glade/scratch/prein/ERA-Interim/Shear06/Shear06/fin_0-6km-Shear_ERA-Interim_12-0_',\
#               '/glade/scratch/prein/ERA-Interim/UV850xTCW/fin_FLX-pl_ERA-Interim_',\
#               '/glade/scratch/prein/ERA-Interim/Shear06/Shear06/fin_0-6km-Shear_ERA-Interim_12-0_',\
#               '/glade/scratch/fossell/ERA-Interim/CAPE-Shear/MAX/fin_CAPE-Shear_ERA-Interim_12-9_',\
#               '/glade/scratch/fossell/ERA-Interim/CAPE_ECMWF/MAX_CAPE/fin_CAPE-MAX-sfc_ERA-Interim_12-9_']


# #MDW optimum - M domain size
# rgsWTvars= ['v', 'UV', 'cape']
# VarsFullName=['V850', 'UV850', 'CAPE']
# rgsWTfolders=['/glade/scratch/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
#               '/glade/scratch/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
#               '/glade/scratch/prein/ERA-Interim/CAPE_ECMWF/fin_CAPE-ECMWF-sfc_ERA-Interim_12-9_']

# #MDW optimum - S domain size
# rgsWTvars= ['cape','cape-shear']
# VarsFullName=['CAPE','CAPE-Shear']
# rgsWTfolders=['/glade/scratch/prein/ERA-Interim/CAPE_ECMWF/fin_CAPE-ECMWF-sfc_ERA-Interim_12-9_',\
#               '/glade/scratch/fossell/ERA-Interim/CAPE-Shear/MAX/fin_CAPE-Shear_ERA-Interim_12-9_']
# rgrNrOfExtremes=20 #[6,10,15,30]
# WT_Domains='S'     # ['S','M','L','XXL'] 
# DomDegreeAdd=2   # [2,   5,  10, 20] 
# Annual_Cycle='1'   # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
# SpatialSmoothing=0.5 #[0,0.5,1]

#MDW optimum - S domain size & relaxed cluster settings
#rgsWTvars= ['q','cape','cape-shear']
#VarsFullName=['Q500','CAPE','CAPE-Shear']
#rgsWTfolders=['/glade/scratch/prein/ERA-Interim/Q500/Q500_daymean_',\
#              '/glade/scratch/prein/ERA-Interim/CAPE_ECMWF/fin_CAPE-ECMWF-sfc_ERA-Interim_12-9_',\
#              '/glade/scratch/fossell/ERA-Interim/CAPE-Shear/MAX/fin_CAPE-Shear_ERA-Interim_12-9_']

rgsWTvars= ['z','06Shear','cape']
VarsFullName=['ZG500', 'VS06','MAX_CAPE']
rgsWTfolders=['/glade/scratch/prein/ERA-Interim/Z500/Z500_daymean_',\
              '/glade/scratch/prein/ERA-Interim/Shear06/Shear06/fin_0-6km-Shear_ERA-Interim_12-0_',\
              '/glade/scratch/fossell/ERA-Interim/CAPE_ECMWF/MAX_CAPE/fin_CAPE-MAX-sfc_ERA-Interim_12-9_']

rgrNrOfExtremes=30 #[6,10,15,30]
WT_Domains='S'     # ['S','M','L','XXL'] 
DomDegreeAdd=2   # [2,   5,  10, 20] 
Annual_Cycle='1'   # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
SpatialSmoothing=0.5 #[0,0.5,1]

