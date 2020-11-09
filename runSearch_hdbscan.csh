#!/bin/csh
#SBATCH --job-name=search_optimum
#SBATCH --account=NMMM0021
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --partition=dav
#SBATCH --output=optimum.out.%j

setenv TMPDIR /glade/scratch/fossell/temp
mkdir -p $TMPDIR

module load ncarenv
module load python  #Load python3 for HDBscan
ncar_pylib /glade/work/fossell/python3_20200417


### Run program
#srun ./SearchOptimum_XWT.py
srun ./SearchOptimum_XWT-Combination.py
