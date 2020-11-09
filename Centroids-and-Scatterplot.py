#!/usr/bin/env python
'''File name: Centroids-and-Scatterplot.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 16.04.2018
    Date last modified: 16.04.2018

    ############################################################## 
    Purpos:

    Apply the optimal settings for the region of interest that was
    determined with:
    ~/papers/Extreme-WTs-US/programs/Extreme-WTs/SearchOptimum_ExtremeEvent-WeatherTyping.py

    1) Read the PRISM and ERA-Interim data
    2) Calculate the WT centroids
    3) Plot the Centroids
    3) Calculate Eucledian Distances for each day
    4) Plot ED vs. PR accumulation

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
import matplotlib
matplotlib.use('Agg')
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
import seaborn as sns
# import metpy.calc as mpcalc
import shapefile as shp
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
from calendar import monthrange
import math

from Functions_Extreme_WTs import XWT
from Functions_Extreme_WTs import MRR, MRD, perkins_skill
from Functions_Extreme_WTs import PreprocessWTdata
from Functions_Extreme_WTs import Centroids_to_NetCDF

def read_shapefile(sf):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' 
    column holding the geometry information. This uses the pyshp
    package
    """
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df

# ###################################################
# Get the setup for the XWTing

from SCS_XWT_apply_settings import rgdTime, iMonths, sPlotDir, sDataDir, Region, sSubregionSCS, rgsWTvars, VarsFullName, rgsWTfolders, rgrNrOfExtremes, WT_Domains, DomDegreeAdd, Annual_Cycle, SpatialSmoothing, Metrics, Dimensions, ClusterMeth, ClusterBreakup

MinDistDD=7 # two extremes should be at least X days appart
#MinDistDD=0 # KRF: try zero, two extremes should be at least X days appart

ss = "-"
Season=ss.join([str(iMonths[ii]) for ii in range(len(iMonths))])
YYYY_stamp=str(rgdTime.year[0])+'-'+str(rgdTime.year[-1])
ss='-'
VarsJoint=ss.join(VarsFullName)
sMonths=ss.join([str(iMonths[ii]) for ii in range(len(iMonths))])
rgiSeasonWT=np.isin(rgdTime.month, iMonths)
rgiYears=np.unique(rgdTime.year)
rgdTime=rgdTime[rgiSeasonWT]
SPLIT=np.where(rgdTime.year <= rgiYears[int(len(rgiYears)/2)])[0][-1]


sPlotDir=sPlotDir+Region+'/'
if not os.path.exists(sPlotDir):
    os.makedirs(sPlotDir)

## KRF comment out:  print str(rgrNrOfExtremes)+' EXTREMES'
iNrOfExtremes=rgrNrOfExtremes   # we consider the N highest rainfall extremes


# ###################################################
#       READ IN SHAPEFILE
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
ctr = np.copy(Subregion)
grSCSregion=mplPath.Path(Subregion)
TMP=np.array(grSCSregion.contains_points(rgrGridCells))
rgrSRactP[TMP == 1]=1
rgrSRactP=np.reshape(rgrSRactP, (rgrLatSCS.shape[0], rgrLatSCS.shape[1]))

DomainSizes=['S', 'M', 'L', 'XXL']
DomDegreeAdd=[2, 5, 10, 20]
DomDelta=DomDegreeAdd[DomainSizes.index(WT_Domains)]

rgiSrSCS=np.array(np.where(rgrSRactP == True))
Wlon=Subregion[:, 0].min()
Elon=Subregion[:, 0].max()
Nlat=Subregion[:, 1].max()
Slat=Subregion[:, 1].min()
DomainWT=np.array([[Elon+DomDelta, Slat-DomDelta],
                   [Wlon-DomDelta, Slat-DomDelta],
                   [Wlon-DomDelta, Nlat+DomDelta],
                   [Elon+DomDelta, Nlat+DomDelta],
                   [Elon+DomDelta, Slat-DomDelta]])
grWTregion=mplPath.Path(DomainWT)

# ###################################################
#         Read the SCS grid and data
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

from Functions_Extreme_WTs import ExtremeDays
rgiSRgridcells=rgrSRactP[iLatMinP:iLatMaxP, iLonMinP:iLonMaxP].astype('int')
rgrPRrecords=np.sum(rgrSCSdata[:, (rgiSRgridcells==1)], axis=(1))

#KRF
# Comment out following 4 lines if want to use default 6,10,30,etc top extreme days.
# Else this block overrides the defaul to set no of extreme days to be no reports/day, hard coded for each case.
#iNrOfExtremes=len(rgrPRrecords[rgrPRrecords>10]) # HARD CODED 6 for sig reports, 20 for all reports
#print(("iNrOfExtremes for Records: " +str(iNrOfExtremes)))
#print(("Length of Records: "+str(len(rgrPRrecords))))
#rgrNrOfExtremes=iNrOfExtremes
#endKRF

rgiExtrTrain=ExtremeDays(rgrPRrecords, iNrOfExtremes, MinDistDD)
ExtrTrainDays=rgdTime[rgiExtrTrain]
rgrPReval=np.sum(rgrSCSdata[:, (rgiSRgridcells == 1)], axis=(1))
rgiExtrEval=ExtremeDays(rgrPRrecords, iNrOfExtremes, MinDistDD)
rgiExtremeDays=rgdTime[rgiExtrEval]

# ###################################################
# ###################################################
#             PERFORM CLUSTER ANALYSIS
sClusterSave=sDataDir+'Clusters'+str(iNrOfExtremes)+'_'+Region+'_'+YYYY_stamp+'_'+VarsJoint+'_'+Season
if os.path.isfile(sClusterSave)  == 0:
    # ###################################################
    #         Read the ERA-Interim grid and data

    from Functions_Extreme_WTs import ReadERAI
    DailyVarsOrig, LonWT, LatWT=ReadERAI(grWTregion,        # shapefile with WTing region
                                         rgdTime,           # time period for WTing
                                         iMonths,           # list of months that should be considered
                                         rgsWTfolders,      # directories containing WT files
                                         rgsWTvars)         # netcdf variable names of WT variables
    # run split sample thest and derive centroids of full dataset
    Samples=['SS1', "SS2", "Full"]
    grClustersFin={}
    grEucledianDist={}
    grCorrelation={}
    for ss in range(len(Samples)):
        print(('    Split Sample '+Samples[ss]))
        if ss == 0:
            DailyVarsTrain=DailyVarsOrig[:SPLIT,:]
            DailyVarsEval=DailyVarsOrig[-SPLIT:,:]
            Ptrain=rgrPRrecords[:SPLIT]
            Peval=rgrPRrecords[-SPLIT:]
            TimeTrain=rgdTime[:SPLIT]
            TimeEval=rgdTime[-SPLIT:]
        elif ss == 1:
            DailyVarsTrain=DailyVarsOrig[-SPLIT:,:]
            DailyVarsEval=DailyVarsOrig[:SPLIT,:]
            Ptrain=rgrPRrecords[-SPLIT:]
            Peval=rgrPRrecords[:SPLIT]
            TimeTrain=rgdTime[-SPLIT:]
            TimeEval=rgdTime[:SPLIT]
        elif ss == 2:
            DailyVarsTrain=DailyVarsOrig
            DailyVarsEval=DailyVarsOrig
            Ptrain=rgrPRrecords
            Peval=rgrPRrecords
            TimeTrain=rgdTime
            TimeEval=rgdTime

        XWT_output=XWT(DailyVarsTrain,
            DailyVarsEval,
            Ptrain,
            Peval,
            TimeTrain,
            TimeEval,
            iNrOfExtremes,
            SpatialSmoothing,
            ClusterMeth=ClusterMeth,
            ClusterBreakup=ClusterBreakup)

        # ################################################
        # ######       EUCLEDIAN DISTANCES
        EucledianDist=XWT_output['EucledianDistAllWTs']
        Correlation =XWT_output['grCorrelatioAllWTs']
        rgrClustersFin=XWT_output['grClustersFin']

        NetCDFname=sDataDir+Region+'_XWT-centroids_train-'+str(TimeTrain.year[0])+'-'+str(TimeTrain.year[-1])+            '_eval-'+str(TimeEval.year[0])+'-'+str(TimeEval.year[-1])+'_E'+str("%03d" % iNrOfExtremes)+            '_XWTs'+str(rgrClustersFin[1].max()+1)+'_Vars-'+VarsJoint+'_M-'+sMonths+'.nc'
        print('    save: '+NetCDFname)
        Centroids_to_NetCDF(NetCDFname,
                        XWT_output,
                        LonWT,
                        LatWT,
                        DailyVarsTrain,
                        rgdTime,
                        TimeEval,
                        VarsJoint,
                        rgiExtremeDays)

        from Functions_Extreme_WTs import Scatter_ED_PR
        MinDistance=np.min(EucledianDist, axis=1)
        ClosestWT=np.argmin(EucledianDist, axis=1)
        MaxCorr=np.max(Correlation, axis=1)
        Scatter_ED_PR(MinDistance,
                      ClosestWT,
                      Peval,
                      rgrNrOfExtremes,
                      PlotLoc=sPlotDir,
                      PlotName='Scatter_'+Region+'_NrExt-'+str(rgrNrOfExtremes)+'_Smooth-'+str(SpatialSmoothing)+'_AnnCy-'+Annual_Cycle+'_'+VarsJoint+'_'+sMonths+'_'+Samples[ss]+'.pdf')
        # save the data
        grClustersFin[Samples[ss]]=rgrClustersFin
        grEucledianDist[Samples[ss]]=EucledianDist
        grCorrelation[Samples[ss]]=Correlation

    DATAcollection={'grClustersFin':grClustersFin, 
                    'grEucledianDist':grEucledianDist, 
                    'grCorrelatio':grCorrelation, 
                    'LonWT':LonWT,
                    'LatWT':LatWT}
    with open(sClusterSave, 'wb') as handle:
        pickle.dump(DATAcollection, handle)
    with open(sClusterSave, 'rb') as handle:
        npzfile = pickle.load(handle)
else:
    print('    Restore: '+sClusterSave)
    with open(sClusterSave, 'rb') as handle:
        npzfile = pickle.load(handle)

    grClustersFin=npzfile['grClustersFin']
    grEucledianDist=npzfile['grEucledianDist']
    grCorrelation=npzfile['grCorrelatio']

rgrClustersFin=grClustersFin['Full']
EucledianDist=grEucledianDist['Full']
Correlation=grCorrelation['Full']

# sort the clusters according to the number of days to ensure that the
# same pattern is at the same location for different event sample sizes
rgiClustSize=[sum(rgrClustersFin[1] == cc) for cc in range(rgrClustersFin[1].max()+1)]
rgiSorted=np.argsort(rgiClustSize)
rgiSorted=list(range(len(rgiSorted)))

#KRF
print(rgrClustersFin[1].max()+1)




# #############################################################################
# ######             PLOT THE CENTROIDS
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pylab import *
import string
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import shapefile
from Functions_Extreme_WTs import add_subplot_axes

rgsWTvarsA=['z', 'u', 'v', 'cape']
rgsWTfolders=['/glade/scratch/prein/ERA-Interim/Z500/Z500_daymean_',\
              '/glade/scratch/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
              '/glade/scratch/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
              '/glade/scratch/prein/ERA-Interim/CAPE_ECMWF/fin_CAPE-ECMWF-sfc_ERA-Interim_12-9_']
# start reading in the precipitation from PRISM for a larger region
iRegionPlus=30 # grid cell added around shape rectangle
ncid=Dataset('/glade/scratch/prein/ERA-Interim/PSL/fin_PSL-sfc_ERA-Interim_12-0_2014.nc', mode='r') # open the netcdf file
rgrLatWT1D=np.squeeze(ncid.variables['lat'][:])
rgrLonWT1D=np.squeeze(ncid.variables['lon'][:])
ncid.close()
rgrLonWT=np.asarray(([rgrLonWT1D,]*rgrLatWT1D.shape[0]))
rgrLonWT[rgrLonWT > 180]=rgrLonWT[rgrLonWT > 180]-360
rgrLatWT=np.asarray(([rgrLatWT1D,]*rgrLonWT1D.shape[0])).transpose()
rgrGridCells=[(rgrLatWT.ravel()[ii], rgrLonWT.ravel()[ii]) for ii in range(len(rgrLonWT.ravel()))]
rgrSRact=np.array(grWTregion.contains_points(rgrGridCells)); rgrSRact=np.reshape(rgrSRact, (rgrLatWT.shape[0], rgrLatWT.shape[1]))
rgiSrWT=np.array(np.where(rgrSRact == True))
iLatMax=rgiSrWT[0,:].max()+iRegionPlus
iLatMin=rgiSrWT[0,:].min()-iRegionPlus
iLonMax=rgiSrWT[1,:].max()+iRegionPlus
iLonMin=rgiSrWT[1,:].min()-iRegionPlus
rgrLatWT_SR=rgrLatWT[iLatMin:iLatMax, iLonMin:iLonMax]
rgrLonWT_SR=rgrLonWT[iLatMin:iLatMax, iLonMin:iLonMax]
rgrWTdata=np.zeros((len(rgiExtremeDays), iLatMax-iLatMin, iLonMax-iLonMin, len(rgsWTvarsA))); rgrWTdata[:]=np.nan
for dd in range(len(rgiExtremeDays)):
    rgdTimeYY = pd.date_range(datetime.datetime(rgiExtremeDays[dd].year, 0o1, 0o1, 0), end=datetime.datetime(rgiExtremeDays[dd].year, 12, 31, 23), freq='d')
    rgiDD=np.where(((rgdTimeYY.year == rgiExtremeDays[dd].year) & (rgdTimeYY.month ==rgiExtremeDays[dd].month ) & (rgdTimeYY.day == rgiExtremeDays[dd].day)))[0]
    for va in range(len(rgsWTvarsA)):
        ncid=Dataset(rgsWTfolders[va]+str(rgiExtremeDays[dd].year)+'.nc', mode='r')
        rgrWTdata[dd,:,:, va]=np.squeeze(ncid.variables[rgsWTvarsA[va]][rgiDD[0], iLatMin:iLatMax, iLonMin:iLonMax])
        ncid.close()
# average over the WTs
rgrWTcentroids=np.zeros((rgrClustersFin[1].max()+1, rgrWTdata.shape[1], rgrWTdata.shape[2], rgrWTdata.shape[3])); rgrWTcentroids[:]=np.nan
for cc in range(rgrClustersFin[1].max()+1):
    rgiClAct=(rgrClustersFin[1] == (cc))
    rgrWTcentroids[cc,:]=np.mean(rgrWTdata[rgiClAct,:,:,:], axis=0)
plt.rcParams.update({'font.size': 18})
rgsLableABC=list(string.ascii_lowercase)

# Calculate the geometry of the plot dependent on how many panels there are
PanWidth=4 # cm
PanHeight=2.5 # cm
xdist=0.5
ydist=0.5
Panels=rgrClustersFin[1].max()+1
iColums=np.min([4,Panels])
iRows=np.max([1,int((math.ceil(Panels/4.)))])
iXX=[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
iYY=[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]

X_Fig=PanWidth*iColums+xdist*(iColums+1)
Y_Fig=PanHeight*iRows+(iRows+1)


plt.rcParams.update({'font.size': 18})
rgsLableABC=list(string.ascii_lowercase)
fig = plt.figure(figsize=(X_Fig,Y_Fig*1.5))
gs1 = gridspec.GridSpec(iRows,iColums)
gs1.update(left=0.01, right=0.99,
           bottom=0.47, top=0.94,
           wspace=0.05, hspace=0.20)
dLat=((52-25)/2.)*1.5
dLon=(130-75)/2.
rllcrnrlat=np.mean((rgrLatWT_SR.min(),rgrLatWT_SR.max()))-dLat #      25 #rllcrnrlat-(urcrnrlat-rllcrnrlat)*0.03
urcrnrlat=np.mean((rgrLatWT_SR.min(),rgrLatWT_SR.max()))+dLat      #52 #urcrnrlat+(urcrnrlat-rllcrnrlat)*0.01
llcrnrlon=np.mean((rgrLonWT_SR.min(),rgrLonWT_SR.max()))-dLon #    -130 #llcrnrlon-abs(llcrnrlon-urcrnrlon)*0.01
urcrnrlon=np.mean((rgrLonWT_SR.min(),rgrLonWT_SR.max()))+dLon #     -75 #urcrnrlon+abs(llcrnrlon-urcrnrlon)*0.01
# # rllcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon=np.min(rgrLatPR_SR),np.max(rgrLatPR_SR),np.min(rgrLatPR_SR),np.max(rgrLatPR_SR)
# rllcrnrlat=25 #rllcrnrlat-(urcrnrlat-rllcrnrlat)*0.03
# urcrnrlat=52 #urcrnrlat+(urcrnrlat-rllcrnrlat)*0.01
# llcrnrlon=-130 #llcrnrlon-abs(llcrnrlon-urcrnrlon)*0.01
# urcrnrlon=-75 #urcrnrlon+abs(llcrnrlon-urcrnrlon)*0.01
for pa in range(Panels):
    # define margins of subplot (see http://matplotlib.org/users/gridspec.html)
    iSampleSize=np.sum(rgrClustersFin[1] == pa)
    ax = plt.subplot(gs1[iYY[pa],iXX[pa]])
    try:
        m = Basemap(projection='cea',                    llcrnrlat= rllcrnrlat ,urcrnrlat=urcrnrlat,                    llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,resolution='l', fix_aspect=False)
    except:
        stop()
    plt.axis('off')
    xi, yi = m(rgrLonWT_SR, rgrLatWT_SR)
    #Load ColorMap
    rgrColorTable=['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffff', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695'][::-1]
    # plot CAPE
    clevs=np.arange(0, 4400, 400); clevs[0]=50
    rgrDataAct=rgrWTcentroids[rgiSorted[pa],:,:, 3]
    rgrDataAct[rgrDataAct < 400] = np.nan
    # biasContDist=4
    # iContNr=len(rgrColorTable)+1
    # clevs=np.arange(0, iContNr*biasContDist,biasContDist)+994
    # rgrDataAct=rgrWTcentroids[rgiSorted[yy*4+xx], :,:,0]/100.
    cs = m.contourf(xi, yi, rgrDataAct, clevs, cmap='YlOrRd', extend='both')
    # plot wind field
    iLev=1; iDist=3
    Q = plt.quiver(xi[::iDist, ::iDist], yi[::iDist, ::iDist], rgrWTcentroids[rgiSorted[pa], ::iDist, ::iDist, 1], rgrWTcentroids[rgiSorted[pa], ::iDist, ::iDist, 2], units='width', zorder = 2, pivot='middle', width=0.004, scale=150)
    if pa == 0:
        qk = ax.quiverkey(Q, 0.95, 1.05, 10, r'$10 \frac{m}{s}$', labelpos='E') #,coordinates='figure')
    # plot ZG500
    CS=m.contour(xi[:,:], yi[:,:], rgrWTcentroids[rgiSorted[pa],:,:, 0]/9.81, levels=np.arange(4000, 6000, 100).astype('int'), linewidths=2, colors='k') #['#d9f0a3','#addd8e','#78c679','#41ab5d','#238443','#005a32'])
    plt.clabel(CS, inline=1, fontsize=12, fmt='%d')
    # plot cluster region
    iLatMax=rgrLatWT1D[rgiSrWT[0,:].max()]
    iLatMin=rgrLatWT1D[rgiSrWT[0,:].min()]
    iLonMax=rgrLonWT1D[rgiSrWT[1,:].max()]
    iLonMin=rgrLonWT1D[rgiSrWT[1,:].min()]
    lats = [iLatMin, iLatMax, iLatMax, iLatMin, iLatMin]
    lons = np.array([iLonMin, iLonMin, iLonMax, iLonMax, iLonMin])-360
    x, y = m(lons, lats)
    m.plot(x, y, lw=3, ls='--', c='r')

    # Lable the map
    plt.title(rgsLableABC[pa]+') WT'+str(pa+1)+', '+str(iSampleSize)+' days', fontsize=16)

    m.drawcoastlines(color='#525252')
    m.drawcountries(color='#525252')
    m.drawstates(color='#525252')
    # add shapefile of catchment
    XX, YY=m(Subregion[:, 1], Subregion[:, 0])
    m.plot(XX, YY, c='r', lw=2)

    # Add Histogram in the lower left corner
    rect = [0.1, 0.1, 0.4, 0.25]
    ax1 = add_subplot_axes(ax, rect)
    # plt.axis('off')
    rgiMonths=np.array([rgiExtremeDays[rgrClustersFin[1] == (pa)][nn].month for nn in range(len(rgiExtremeDays[rgrClustersFin[1] == (pa)]))])
   #ax1.hist(rgiMonths, np.array(list(range(1, 13, 1)))+0.5, normed=1, facecolor='k', alpha=0.75, rwidth=0.6)
    ax1.hist(rgiMonths, np.array(list(range(1, 13, 1)))+0.5, density=True, facecolor='k', alpha=0.75, rwidth=0.6) # KRF edit after phy3 convert
    ax1.set_xticks(np.array(list(range(1, 13, 1))))
    ax1.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xlim([0.5, 12.5])
    # ax2.yaxis.set_ticks_position('left')
    ax1.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        right='off',         # ticks along the top edge are off
        labelleft='off') # labels along the bottom edge are off
    ax1.patch.set_alpha(0.0)
            
    # ==========================================================================================
    # ==========================================================================================
    # plot the actual centroids for each WT
    LonWT=npzfile['LonWT']
    LatWT=npzfile['LatWT']
    pos1 = np.array(ax.get_position())
    gs2 = gridspec.GridSpec(1,len(rgsWTvars))
    gs2.update(left=pos1[0][0], right=pos1[1][0],
           bottom=0.10, top=0.30,
           wspace=0.05, hspace=0.20)
    for va in range(len(rgsWTvars)):
        ax = plt.subplot(gs2[0,va])
        CentroidsAct=rgrClustersFin[0][pa]
        CentroidsAct=np.reshape(CentroidsAct,(LonWT.shape[0],LonWT.shape[1],len(rgsWTvars)))
        m = Basemap(projection='cea',                    llcrnrlat= np.min(LatWT) ,urcrnrlat=np.max(LatWT),                    llcrnrlon=np.min(LonWT),urcrnrlon=np.max(LonWT),resolution='l', fix_aspect=False)
        xi, yi = m(LonWT, LatWT)
        cs1= plt.contourf(xi, yi, CentroidsAct[:,:,va], cmap='coolwarm',levels=np.linspace(-2,2,11),extend='both')
        plt.axis('off')
        # add shapefile of catchment
        XX,YY=m(ctr[:,0], ctr[:,1])
        m.plot(XX,YY, c='r', lw=2)
        m.drawcoastlines(color='k')
        m.drawcountries(color='k')
        m.drawstates(color='k')
        plt.title(rgsLableABC[pa]+str(int(va+1))+') '+VarsFullName[va], fontsize=14)

# add colorbar for centroids
CbarAx = axes([0.025, 0.04, 0.95, 0.02])
cb = colorbar(cs1, cax = CbarAx, orientation='horizontal', extend='both') #, ticks=np.arange(300,1600,200))
cb.ax.tick_params(labelsize=14)
cb.ax.set_title('normalized centroids []', fontsize=14)



            # # add individual days beneath the plot
            # POSSITION=np.array(ax.get_position().bounds)
            # iSamples=np.sum((rgrClustersFin[1] == (yy*4+xx)))
            # gs2 = gridspec.GridSpec(1, iSamples)
            # gs2.update(left=POSSITION[0], right=POSSITION[0]+POSSITION[2],
            #            bottom=POSSITION[1]-0.15, top=POSSITION[1]-0.015,
            #            wspace=0.05, hspace=0.20)
            # for cc in range(iSamples):
            #     DATA=rgrWTdata[(rgrClustersFin[1] == (yy*4+xx)),:,:,:][cc,:,:,:]
            #     axs=plt.subplot(gs2[0, cc])
            #     m = Basemap(projection='cea',\
            #                 llcrnrlat= rllcrnrlat, urcrnrlat=urcrnrlat,\
            #                 llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, resolution='l', fix_aspect=False)
            #     plt.axis('off')
            #     xi, yi = m(rgrLonWT_SR, rgrLatWT_SR)
            #     # plot CAPE
            #     clevs=np.arange(0, 4400, 400); clevs[0]=50
            #     rgrDataAct=DATA[:,:, 3]
            #     rgrDataAct[rgrDataAct < 400] = np.nan
            #     cs = m.contourf(xi, yi, rgrDataAct, clevs, cmap='YlOrRd', extend='both')
            #     iLev=1; iDist=3
            #     Q = plt.quiver(xi[::iDist, ::iDist], yi[::iDist, ::iDist], rgrWTcentroids[rgiSorted[yy*4+xx], ::iDist, ::iDist, 1], rgrWTcentroids[rgiSorted[yy*4+xx], ::iDist, ::iDist, 2], units='width', zorder = 2, pivot='middle', width=0.004, scale=150)
            #     # plot geopotential heigth
            #     CS=m.contour(xi[:,:], yi[:,:], DATA[:,:, 0]/9.81, levels=np.arange(4000, 6000, 100).astype('int'), linewidths=2, colors='k')
            #     plt.clabel(CS, inline=1, fontsize=12, fmt='%d')
            #     # plot cluster region
            #     iLatMax=rgrLatWT1D[rgiSrWT[0,:].max()]
            #     iLatMin=rgrLatWT1D[rgiSrWT[0,:].min()]
            #     iLonMax=rgrLonWT1D[rgiSrWT[1,:].max()]
            #     iLonMin=rgrLonWT1D[rgiSrWT[1,:].min()]
            #     lats = [iLatMin, iLatMax, iLatMax, iLatMin, iLatMin]
            #     lons = np.array([iLonMin, iLonMin, iLonMax, iLonMax, iLonMin])-360
            #     x, y = m(lons, lats)
            #     m.plot(x, y, lw=3, ls='--', c='r')
            #     m.drawcoastlines(color='#525252')
            #     m.drawcountries(color='#525252')
            #     m.drawstates(color='#525252')
            #     # add shapefile of catchment
            #     XX, YY=m(Subregion[:, 1], Subregion[:, 0])
            #     m.plot(XX, YY, c='r', lw=2)


# add colorbar
CbarAx = axes([0.025, 0.38, 0.95, 0.02])
cb = colorbar(cs, cax = CbarAx, orientation='horizontal', extend='both', ticks=clevs)
cb.ax.set_title('CAPE [J kg$^{-1}$]')

# Save the plot
# plt.show()
sPlotFile=sPlotDir
# sPlotName= 'BottomUp-'+str(rgrClustersFin[1].max()+1)+'WT_Centroids.pdf'
sPlotName= 'BottomUp_'+str("%03d" % iNrOfExtremes)+'_Events_'+str(rgrClustersFin[1].max()+1)+'WT_Centroids_'+YYYY_stamp+'_'+VarsJoint+'_'+sMonths+'.pdf'
if os.path.isdir(sPlotFile) != 1:
    subprocess.call(["mkdir", "-p", sPlotFile])
print('        Plot map to: '+sPlotFile+sPlotName)
fig.savefig(sPlotFile+sPlotName)








# # #############################################################################
# # ######             PLOT SCS average in WTs

# start reading in the precipitation from PRISM for a larger region
iRegionPlus=2 # grid cell added around shape rectangle
ncid=Dataset('/glade/work/fossell/preevents/grid_obs/monthly_sig/reports_201411.nc', mode='r') # open the netcdf file
rgrLatPR=np.squeeze(ncid.variables['Lats'][0,:])
rgrLonPR=np.squeeze(ncid.variables['Lons'][0,:])
ncid.close()
rgrGridCells=[(rgrLatPR.ravel()[ii],rgrLonPR.ravel()[ii]) for ii in range(len(rgrLonPR.ravel()))]
rgrSRact=np.array(grSCSregion.contains_points(rgrGridCells)); rgrSRact=np.reshape(rgrSRact, (rgrLatPR.shape[0], rgrLatPR.shape[1]))
rgiSrPR=np.array(np.where(rgrSRact == True))
iLatMax=np.min([rgiSrPR[0,:].max()+100, rgrLatPR.shape[0]-1])
iLatMin=np.max([rgiSrPR[0,:].min()-100, 0])
iLonMax=np.min([rgiSrPR[1,:].max()+100, rgrLatPR.shape[1]-1])
iLonMin=np.max([rgiSrPR[1,:].min()-100, 0])
rgrLatPR_SR=rgrLatPR[iLatMin:iLatMax,iLonMin:iLonMax]
rgrLonPR_SR=rgrLonPR[iLatMin:iLatMax,iLonMin:iLonMax]
rgrPRdata=np.zeros((rgrNrOfExtremes,iLatMax-iLatMin,iLonMax-iLonMin)); rgrPRdata[:]=np.nan
jj=0
for dd in range(len(rgiExtremeDays)):
    MonDays = monthrange(rgiExtremeDays[dd].year, rgiExtremeDays[dd].month)[1]
    rgdTimeYY = pd.date_range(datetime.datetime(rgiExtremeDays[dd].year, rgiExtremeDays[dd].month, 1,0), end=datetime.datetime(rgiExtremeDays[dd].year, rgiExtremeDays[dd].month, MonDays,23), freq='d')
    rgiDD=np.where(((rgdTimeYY.year == rgiExtremeDays[dd].year) & (rgdTimeYY.month ==rgiExtremeDays[dd].month ) & (rgdTimeYY.day == rgiExtremeDays[dd].day)))[0]
    ncid=Dataset('/glade/work/fossell/preevents/grid_obs/monthly_sig/reports_'+str(rgiExtremeDays[dd].year)+str(rgiExtremeDays[dd].month).zfill(2)+'.nc', mode='r')
    try:
        rgrPRdata[dd,:,:]=np.squeeze(ncid.variables['OSR'][rgiDD[0],iLatMin:iLatMax,iLonMin:iLonMax])
    except:
        stop()
    ncid.close()
    jj=jj+len(rgiDD)
rgrPRdata[rgrPRdata < 0]=np.nan

# precipitation in WTs
rgrPR_WTs=np.zeros((rgrClustersFin[1].max()+1,rgrPRdata.shape[1],rgrPRdata.shape[2])); rgrPR_WTs[:]=np.nan
for cc in range(rgrClustersFin[1].max()+1):
    rgiClAct=(rgrClustersFin[1] == (cc))
    try:
        rgrPR_WTs[cc,:]=np.nanmean(rgrPRdata[rgiClAct,:], axis=0)
    except:
        stop()

plt.rcParams.update({'font.size': 18})
rgsLableABC=list(string.ascii_lowercase)


PanWidth=4 # cm
PanHeight=2.5 # cm
xdist=0.5
ydist=0.5
Panels=rgrClustersFin[1].max()+1
iColums=np.min([4,Panels])
iRows=np.max([1,int((math.ceil(Panels/4.)))])
iXX=[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
iYY=[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]

X_Fig=PanWidth*iColums+xdist*(iColums+1)
Y_Fig=PanHeight*iRows+(iRows+1)


plt.rcParams.update({'font.size': 18})
rgsLableABC=list(string.ascii_lowercase)
fig = plt.figure(figsize=(X_Fig,Y_Fig*1.1))
gs1 = gridspec.GridSpec(iRows,iColums)
gs1.update(left=0.01, right=0.99,
           bottom=0.20, top=0.90,
           wspace=0.05, hspace=0.20)
dLat=((52-25)/2.)*1.1
dLon=(130-75)/2.
rllcrnrlat=np.mean((rgrLatWT_SR.min(),rgrLatWT_SR.max()))-dLat #      25 #rllcrnrlat-(urcrnrlat-rllcrnrlat)*0.03
urcrnrlat=np.mean((rgrLatWT_SR.min(),rgrLatWT_SR.max()))+dLat      #52 #urcrnrlat+(urcrnrlat-rllcrnrlat)*0.01
llcrnrlon=np.mean((rgrLonWT_SR.min(),rgrLonWT_SR.max()))-dLon #    -130 #llcrnrlon-abs(llcrnrlon-urcrnrlon)*0.01
urcrnrlon=np.mean((rgrLonWT_SR.min(),rgrLonWT_SR.max()))+dLon #     -75 #urcrnrlon+abs(llcrnrlon-urcrnrlon)*0.01
for pa in range(Panels):
    iSampleSize=np.sum(rgrClustersFin[1] == ([pa]))
    # rAvPR=np.mean(rgrPRrecords[rgiExtremePR][rgrClustersFin[1] == (rgiSorted[pa])])
    ax = plt.subplot(gs1[iYY[pa],iXX[pa]])
    m = Basemap(projection='cea',\
                    llcrnrlat= rllcrnrlat ,urcrnrlat=urcrnrlat,\
                    llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,resolution='i', fix_aspect=False)
    plt.axis('off')
    xi, yi = m(rgrLonPR_SR, rgrLatPR_SR)
    #Load ColorMap
    rgrColorTable=['#ffffe5','#fff7bc','#fee391','#fec44f','#fe9929','#ec7014','#cc4c02','#993404','#662506']
    rgrColorTable=np.append(['#ffffff'],rgrColorTable)
    clevs=np.array(np.arange(0,1,0.05))[:len(rgrColorTable)]
    rgrDataAct=rgrPR_WTs[rgiSorted[pa], :,:]
    # rgrDataAct[rgrDataAct < 0.1] = np.nan
    cs = m.contourf(xi,yi,rgrDataAct,clevs,colors=rgrColorTable, extend='max')
    # Lable the map
    plt.title(rgsLableABC[pa]+') WT'+str(pa+1)+', '+str(iSampleSize)+' days, ') #+str(int(rAvPR))+' mm d$^{-1}$') #, fontsize=16)


    # plot cluster region
    iLatMax=rgrLatWT1D[rgiSrWT[0,:].max()]
    iLatMin=rgrLatWT1D[rgiSrWT[0,:].min()]
    iLonMax=rgrLonWT1D[rgiSrWT[1,:].max()]
    iLonMin=rgrLonWT1D[rgiSrWT[1,:].min()]
    lats = [iLatMin, iLatMax, iLatMax, iLatMin, iLatMin]
    lons = np.array([iLonMin, iLonMin, iLonMax, iLonMax, iLonMin])-360
    x, y = m(lons, lats)
    m.plot(x, y, lw=3, ls='--', c='r')
    m.drawcoastlines(color='#525252')
    m.drawcountries(color='#525252')
    m.drawstates(color='#525252')
    # add shapefile of catchment
    XX, YY=m(Subregion[:, 1], Subregion[:, 0])
    m.plot(XX, YY, c='r', lw=2)

    # add shapefile of catchment
    XX,YY=m(ctr[:,0], ctr[:,1])
    m.plot(XX,YY, c='k', lw=2)

# add colorbar
CbarAx = axes([0.10, 0.07, 0.80, 0.02])
cb = colorbar(cs, cax = CbarAx, orientation='horizontal', extend='both', ticks=clevs)
cb.ax.set_title('average SCS reports [-]')


# Save the plot
# plt.show()
sPlotFile=sPlotDir
# sPlotName= 'BottomUp-'+str(rgrClustersFin[1].max()+1)+'WT_precipitation.pdf'
sPlotName= 'BottomUp_'+str("%03d" % iNrOfExtremes)+'_Events'+'_'+str(rgrClustersFin[1].max()+1)+'WT_SCS-events'+'_'+YYYY_stamp+'_'+VarsJoint+'_'+sMonths+'.pdf'
if os.path.isdir(sPlotFile) != 1:
    subprocess.call(["mkdir","-p",sPlotFile])
print( '        Plot map to: '+sPlotFile+sPlotName)
fig.savefig(sPlotFile+sPlotName)





stop()

