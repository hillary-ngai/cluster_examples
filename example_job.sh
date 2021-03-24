#/bin/bash

# set up environment
. ~/med-dialogue-system/env/bin/activate

# symlink checkpoint directory to run directory
#ln -s /checkpoint/$USER/$SLURM_JOB_ID /output

python3 train.py

hostname
echo "Today is `date`"
