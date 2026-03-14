#!/bin/sh -v
#PBS -e /mnt/home/valtersve/matiss_sem_seg/logs
#PBS -o /mnt/home/valtersve/matiss_sem_seg/logs
#PBS -q batch
#PBS -l nodes=1:ppn=4:gpus=1,feature=l40s
#PBS -l mem=80gb
#PBS -l walltime=96:00:00
#PBS -N utonia_linear_probe
#PBS -W x=HOSTLIST:wn74

source /mnt/home/valtersve/anaconda3/etc/profile.d/conda.sh
conda activate matiss_utonia
export LD_LIBRARY_PATH=/mnt/home/valtersve/anaconda3/envs/matiss_utonia/lib:$LD_LIBRARY_PATH

echo Working Directory: $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)

export TMPDIR=/mnt/beegfs2/beegfs_scratch/$USER/$PBS_JOBID

beegfs-ctl --getquota --uid "$(id -u valtersve)"

mkdir -m 700 -p /mnt/beegfs2/beegfs_scratch/$USER/$PBS_JOBID

cp $PBS_O_WORKDIR/Utonia/demo/linear_probe.py $TMPDIR

# cp -a "$PBS_O_WORKDIR/data/LiDAR_Net" "$TMPDIR/"

python $TMPDIR/linear_probe.py --data_path $PBS_O_WORKDIR/data/LiDAR_Net --output_path $TMPDIR

cp $TMPDIR/utonia_lidarnet.pt $PBS_O_WORKDIR

rm -rf $TMPDIR

exit
