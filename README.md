# Extreme Weather Typing (XWT) for Severe Convective Storms (SCS)
XWT program was developed and adapted to SCS by Andy Prein. 

## Dependencies
* Python 3
* HDBscan


## Components
**1. Search for optimum** 
  * SCS_search_settings.py
  * SearchOptimum_XWT-Combination.py
  
**2. Apply settings** 
  * SCS_XWT_apply_settings.py
  * Centroids-and-Scatterplot.py
  
## Running the Programs
Example for running on Casper using cloned NCAR Package Library.

**1. Load python module and your NPL that has HDBscan installed.**<br>
   Environment exmaple as located in *environment_hdbscan*

`module load python/3.7.5`

`ncar_pylib python3_20200417`

**2. Run the Search Program**

Edit the *SCS_search_settings.py*

`sbatch runSearch_hdbscan.csh` <br>
This calls *SearchOptimum_XWT-Combination.py* <br>
Output includes optimum settings.

**3. Apply the settings to the clustering algorithms**

Edit the *SCS_XWT_apply_settings.py* with settings from step 2 (or other desired settings)

`./Centroids-and-Scatterplot.py`<br>
Output located in data/ and plot/ directories as set in SCS* settings files.  
