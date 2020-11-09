#!/usr/bin/env python
'''File name: ExtremeEvent-WeatherTyping.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 16.04.2018
    Date last modified: 20.04.2018
    
    This version of the program does not use two loops to find the 
    best variables but rather test all possible combintions of up to
    4 variables.

    ############################################################## 
    Purpos:
    Classifies weather patterns that cause precipitation extremes

    1) read in shape file for area under cosideration

    2) read in precipitation data from PRISM

    3) identify the N-days that had highest rainfall records

    4) read in ERA-Interim data for these days

    5) remove the 30-year mooving average from time series and
       normalize the variables

    5) run clustering algorithm on extreme WT patterns

    6) search for the extreme WT centroids in the full record


'''

from dateutil import rrule
import datetime
import glob
from netCDF4 import Dataset
import sys, traceback
import dateutil.parser as dparser
import string
from pdb import set_trace as stop
import numpy as np
import numpy.ma as ma
import os
from mpl_toolkits import basemap
import pickle
import subprocess
import pandas as pd
from scipy import stats
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import pylab as plt
import random
import scipy.ndimage as ndimage
import scipy
import shapefile
import matplotlib.path as mplPath
from matplotlib.patches import Polygon as Polygon2
# Cluster specific modules
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.vq import kmeans2, vq, whiten
from scipy.ndimage import gaussian_filter
# import seaborn as sns
# import metpy.calc as mpcalc
import shapefile as shp
import sys
from itertools import combinations 

from Functions_Extreme_WTs import XWT
from Functions_Extreme_WTs import MRR, MRD, perkins_skill

# ###################################################
# This information comes from the setup file

from SCS_search_settings import rgdTime, iMonths, sPlotDir, sDataDir, Region, sSubregionSCS, rgsWTvars, VarsFullName, rgsWTfolders, rgrNrOfExtremes, WT_Domains, DomDegreeAdd, Annual_Cycle, SpatialSmoothing, Metrics, Dimensions, ClusterMeth, ClusterBreakup

ss='-'
sMonths=ss.join([str(iMonths[ii]) for ii in range(len(iMonths))])
print(sMonths)
sDomains = ss.join([str(WT_Domains[ii]) for ii in range(len(WT_Domains))])
sExtremes = ss.join([str(rgrNrOfExtremes[ii]) for ii in range(len(rgrNrOfExtremes))])

# create all possible combinations of variables
Combinations1=np.array(list(combinations(np.array(list(range(len(VarsFullName)))), 1)))
Combinations2=np.squeeze(np.array(list(combinations(np.array(list(range(len(VarsFullName)))), 2))))
Combinations3=np.squeeze(np.array(list(combinations(np.array(list(range(len(VarsFullName)))), 3))))
Combinations4=np.squeeze(np.array(list(combinations(np.array(list(range(len(VarsFullName)))), 4))))
Combinations=list(Combinations1)+list(Combinations2)+list(Combinations3) #+list(Combinations4)

# create nessesary directories
if not os.path.exists(sDataDir):
    os.makedirs(sDataDir)
if not os.path.exists(sPlotDir):
    os.makedirs(sPlotDir)
sRegion=Region


# ###################################################
# use setup to generate data
rgiYears=np.unique(rgdTime.year)
YYYY_stamp=str(rgdTime.year[0])+'-'+str(rgdTime.year[-1])
rgiSeasonWT=np.isin(rgdTime.month, iMonths)
rgdTime=rgdTime[rgiSeasonWT]

SPLIT=np.where(rgdTime.year <= rgiYears[int(len(rgiYears)/2)])[0][-1]
SkillScores_All=np.zeros((len(Combinations), len(rgrNrOfExtremes), len(WT_Domains), len(Annual_Cycle), len(SpatialSmoothing), 2, len(Metrics))); SkillScores_All[:]=np.nan

SaveStats=sDataDir+sRegion+'_'+YYYY_stamp+'-'+sMonths+'_'+sDomains+'_'+sExtremes+'.npz'

if os.path.isfile(SaveStats) == 0:

    # ###################################################
    print('    Read the SCS data')
    
    #ncid=Dataset('/glade/work/fossell/preevents/grid_obs/monthly/reports_201604.nc', mode='r') # open the netcdf file
    ncid=Dataset('/glade/work/fossell/preevents/grid_obs/monthly_sig/reports_201604.nc', mode='r') # open the netcdf file
    rgrLatSCS=np.squeeze(ncid.variables['Lats'][:])[0,:,:]
    rgrLonSCS=np.squeeze(ncid.variables['Lons'][:])[0,:,:]
    ncid.close()
    rgrGridCells=[(rgrLatSCS.ravel()[ii], rgrLonSCS.ravel()[ii]) for ii in range(len(rgrLonSCS.ravel()))]
    rgrSRactP=np.zeros((rgrLonSCS.shape[0]*rgrLonSCS.shape[1]))
    # read in mask ploygon for subregion selection
    Subregion=np.array(pd.read_csv(sSubregionSCS).values)
    Subregion=np.array([Subregion[ii][0].split(' ') for ii in range(len(Subregion))])
    Subregion=Subregion.astype('float')
    grSCSregion=mplPath.Path(Subregion)
    ctr=Subregion
    TMP=np.array(grSCSregion.contains_points(rgrGridCells))
    rgrSRactP[TMP == 1]=1
    rgrSRactP=np.reshape(rgrSRactP, (rgrLatSCS.shape[0], rgrLatSCS.shape[1]))
    
    rgiSrSCS=np.array(np.where(rgrSRactP == True))
    iLatMaxP=rgiSrSCS[0,:].max()+1
    iLatMinP=rgiSrSCS[0,:].min()
    iLonMaxP=rgiSrSCS[1,:].max()+1
    iLonMinP=rgiSrSCS[1,:].min()
    rgrSCSdata=np.zeros((sum(rgiSeasonWT), iLatMaxP-iLatMinP, iLonMaxP-iLonMinP))
    for yy in range(len(rgiYears)):
        for mo in range(len(iMonths)):
            iDays=((rgdTime.year == rgiYears[yy]) & (rgdTime.month == iMonths[mo]))
           #ncid=Dataset('/glade/work/fossell/preevents/grid_obs/monthly/reports_'+str(rgiYears[0]+yy)+str("%02d" %  iMonths[mo])+'.nc', mode='r')
            ncid=Dataset('/glade/work/fossell/preevents/grid_obs/monthly_sig/reports_'+str(rgiYears[0]+yy)+str("%02d" %  iMonths[mo])+'.nc', mode='r')
            DATA=np.squeeze(ncid.variables['OSR'][:, iLatMinP:iLatMaxP, iLonMinP:iLonMaxP])
            try:
                rgrSCSdata[iDays,:,:]=DATA
            except:
               #print('/glade/work/fossell/preevents/grid_obs/monthly/reports_'+str(rgiYears[0]+yy)+str("%02d" %  iMonths[mo])+'.nc')
                print(('/glade/work/fossell/preevents/grid_obs/monthly_sig/reports_'+str(rgiYears[0]+yy)+str("%02d" %  iMonths[mo])+'.nc'))
            ncid.close()
    rgrSCSdata[rgrSCSdata<0] = np.nan
    
    
    print( '    Read the ERA-Interim data')
    # We read in ERA-Interim data for the largest region and cut it to fit smaller regions
    DomDelta=np.max(DomDegreeAdd)
    Wlon=ctr[:, 0].min()
    Elon=ctr[:, 0].max()
    Nlat=ctr[:, 1].max()
    Slat=ctr[:, 1].min()
    DomainWT=np.array([[Elon+DomDelta, Slat-DomDelta],
                       [Wlon-DomDelta, Slat-DomDelta],
                       [Wlon-DomDelta, Nlat+DomDelta],
                       [Elon+DomDelta, Nlat+DomDelta],
                       [Elon+DomDelta, Slat-DomDelta]])
    grWTregion=mplPath.Path(DomainWT)
    
    # ###################################################
    #         Read the ERA-Interim grid and data
    from Functions_Extreme_WTs import ReadERAI
    DailyVarsLargeDom=ReadERAI(grWTregion,      # shapefile with WTing region
                       rgdTime,         # time period for WTing
                       iMonths,         # list of months that should be considered
                       rgsWTfolders,    # directories containing WT files
                       rgsWTvars)       # netcdf variable names of WT variables
    
    # ###################################################
    print( '    Read the ERA-Interim data specific for the region')
    
    for re in range(len(WT_Domains)):
        print( '    ------')
        print(( '    Domain '+WT_Domains[re]))
        DeltaX=np.max(DomDegreeAdd)-DomDegreeAdd[re]
        if DeltaX != 0:
            DomainWT=np.array([[Elon+DomDegreeAdd[re], Slat-DomDegreeAdd[re]],
                       [Wlon-DomDegreeAdd[re], Slat-DomDegreeAdd[re]],
                       [Wlon-DomDegreeAdd[re], Nlat+DomDegreeAdd[re]],
                       [Elon+DomDegreeAdd[re], Nlat+DomDegreeAdd[re]],
                       [Elon+DomDegreeAdd[re], Slat-DomDegreeAdd[re]]])
    
            grWTregion=mplPath.Path(DomainWT)
            rgrGridCells=[(DailyVarsLargeDom[1].ravel()[ii], DailyVarsLargeDom[2].ravel()[ii]) for ii in range(len(DailyVarsLargeDom[1].ravel()))]
            rgrSRact=np.array(grWTregion.contains_points(rgrGridCells)); rgrSRact=np.reshape(rgrSRact, (DailyVarsLargeDom[1].shape[0], DailyVarsLargeDom[1].shape[1]))
            rgiSrWT=np.array(np.where(rgrSRact == True))
            iLatMax=rgiSrWT[0,:].max()
            iLatMin=rgiSrWT[0,:].min()
            iLonMax=rgiSrWT[1,:].max()
            iLonMin=rgiSrWT[1,:].min()
            DailyVars=DailyVarsLargeDom[0][:, iLatMin:iLatMax, iLonMin:iLonMax,:]
        else:
            DailyVars=DailyVarsLargeDom[0]
    
        # perform split sample statistic
        for ss in range(2):
            print(( '    Split Sample Nr. '+str(ss+1)))
            if ss == 0:
                DailyVarsTrain=DailyVars[:SPLIT,:]
                DailyVarsEval=DailyVars[-SPLIT:,:]
                Ptrain=rgrSCSdata[:SPLIT]
                Peval=rgrSCSdata[-SPLIT:]
                TimeTrain=rgdTime[:SPLIT]
                TimeEval=rgdTime[-SPLIT:]
            else:
                DailyVarsTrain=DailyVars[-SPLIT:,:]
                DailyVarsEval=DailyVars[:SPLIT,:]
                Ptrain=rgrSCSdata[-SPLIT:]
                Peval=rgrSCSdata[:SPLIT]
                TimeTrain=rgdTime[-SPLIT:]
                TimeEval=rgdTime[:SPLIT]
    
    #KRF
            rgiSRgridcells=rgrSRactP[iLatMinP:iLatMaxP, iLonMinP:iLonMaxP].astype('int')
           #TotRecordsDays=np.sum(rgrSCSdata[:, (rgiSRgridcells == 1)], axis=(1))  #comment out if using default 6,10,30
           #NoDaysExtremes=len(TotRecordsDays[TotRecordsDays>10]) # HARD CODED 6 for sig reports, 20 for all reports  #comment out if using default 6,10,30
           #print(("NoDaysExtremes: ", NoDaysExtremes))  #comment out if using default 6,10,30
           #rgrNrOfExtremes = [NoDaysExtremes] #comment out if using default 6,10,30,etc top extreme days. Else this gives no reports/day, hardcoded above 
           #print(rgrNrOfExtremes) #comment out if using default 6,10,30
    
            rgrPRrecords=np.sum(Ptrain[:, (rgiSRgridcells==1)], axis=(1))
            rgrPReval=np.sum(Peval[:, (rgiSRgridcells == 1)], axis=(1))
    #endKRF
    
            for ne in range(len(rgrNrOfExtremes)):
                DailyVarsAct=np.copy(DailyVarsTrain)
                print(( '        '+str(rgrNrOfExtremes[ne])+' EXTREMES'))
                iNrOfExtremes=rgrNrOfExtremes[ne]   # we consider the N highest rainfall extremes
    
               # KRF Move next three lines above out of loop, use to determine records>##
               #rgiSRgridcells=rgrSRactP[iLatMinP:iLatMaxP,iLonMinP:iLonMaxP].astype('int')
               #rgrPRrecords=np.sum(Ptrain[:,(rgiSRgridcells==1)], axis=(1))
               #rgrPReval=np.sum(Peval[:,(rgiSRgridcells == 1)], axis=(1))
    
    
                # Test effect of spatial smoothing
                for sm in range(len(SpatialSmoothing)):
                    # annual cycle treatment
                    for ac in range(len(Annual_Cycle)):
                        print( '            Loop over variable permutations')
                        for va1 in range(len(Combinations)): 
                            XWT_output=XWT(DailyVarsTrain[:,:,:, Combinations[va1]],
                                           DailyVarsEval[:,:,:, Combinations[va1]],
                                           rgrPRrecords,
                                           rgrPReval,
                                           TimeTrain,
                                           TimeEval,
                                           rgrNrOfExtremes[ne],
                                           SpatialSmoothing[sm])
    
                            SkillScores_All[va1, ne, re, ac, sm, ss, Metrics.index('PSS')]=XWT_output['grPSS'] # Perkins Skill Score
                            SkillScores_All[va1, ne, re, ac, sm, ss, Metrics.index('MRD')]=XWT_output['grMRD'] # Mean relative difference
                            SkillScores_All[va1, ne, re, ac, sm, ss, Metrics.index('MRR')]=XWT_output['grMRR'] # Mean Rank Ratio
                            SkillScores_All[va1, ne, re, ac, sm, ss, Metrics.index('APR')]=np.abs(XWT_output['APR']-1) # Average precision-recall score
                            SkillScores_All[va1, ne, re, ac, sm, ss, Metrics.index('PEX')]=np.abs(XWT_output['PEX']/100.-1) # Percent of points excluded for ED larger than the 75 percentile
                print(' ')
    
    np.savez(sDataDir+Region+'-'+sMonths+'.npz',
             SkillScores_All=SkillScores_All, 
             Combinations=Combinations, 
             rgsWTvars=VarsFullName,
             rgrNrOfExtremes=rgrNrOfExtremes,
             WT_Domains=WT_Domains,
             Annual_Cycle=Annual_Cycle,
             SpatialSmoothing=SpatialSmoothing,
             Metrics=Metrics,
             Dimensions=Dimensions)

else:
    print('    Load: '+SaveStats)
    DATA=np.load(SaveStats)
    SkillScores_All=DATA['SkillScores_All']
    Combinations=DATA['Combinations']
    VarsFullName=DATA['rgsWTvars']
    rgrNrOfExtremes=DATA['rgrNrOfExtremes']
    WT_Domains=DATA['WT_Domains']
    Annual_Cycle=DATA['Annual_Cycle']
    SpatialSmoothing=DATA['SpatialSmoothing']
    Metrics=DATA['Metrics']
    Dimensions=DATA['Dimensions']

# Find optimum and print best setting
Metrics=list(Metrics)
Scores=[Metrics.index('APR'), Metrics.index('PEX')]
Mean_SS=np.mean(SkillScores_All[:,:,:,:,:,:, Scores], axis=(5, 6))
iOpt=np.where(Mean_SS.min() == Mean_SS)

print(' ')
print('====================================')
print('======    OPTIMAL SETTINGS    ======')
print(('Region: '+Region))
print(('Months: '+sMonths))
print('VARIABLES')
for va in range(len(Combinations[iOpt[0][0]])):
    print(('    '+VarsFullName[int(Combinations[iOpt[0][0]][va])]))
print(('Extreme Nr     : '+str(rgrNrOfExtremes[iOpt[1][0]])))
print(('Domain Size    : '+str(WT_Domains[iOpt[2][0]])))
print(('Annual Cy. Rem.: '+str(Annual_Cycle[iOpt[3][0]])))
print(('Smoothing      : '+str(SpatialSmoothing[iOpt[4][0]])))
print(('Average Score  : '+str(np.round(Mean_SS.min(), 2))))
print('====================================')








# In[80]:



# PlotFile=sRegion+'_XWT_Search-Optimum_'+YYYY_stamp+'_'+sMonths+'.pdf'
# from Functions_Extreme_WTs import SearchOptimum_XWT
# SearchOptimum_XWT(PlotFile,
#                  sPlotDir,
#                  SkillScores_All,
#                  GlobalMinimum1,
#                  GlobalMinimum2,
#                  Optimum,
#                  VariableIndices,
#                  Dimensions,
#                  Metrics,
#                  VarsFullName,
#                  ss,
#                  rgrNrOfExtremes,
#                  WT_Domains,
#                  Annual_Cycle,
#                  SpatialSmoothing)


#def PlotOptimum_XWT(PlotFile,
#                    sPlotDir,
#                    SkillScores_All,
#                    GlobalMinimum1,
#                    GlobalMinimum2,
#                    Optimum,
#                    VariableIndices,
#                    Dimensions,
#                    Metrics,
#                    VarsFullName,
#                    ss,
#                    rgrNrOfExtremes,
#                    WT_Domains,
#                    Annual_Cycle,
#                    SpatialSmoothing)
