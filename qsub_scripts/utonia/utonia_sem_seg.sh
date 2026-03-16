#!/bin/sh -v
#PBS -e /mnt/home/valtersve/matiss_sem_seg/logs
#PBS -o /mnt/home/valtersve/matiss_sem_seg/logs
#PBS -q batch
#PBS -l nodes=1:ppn=4:gpus=1,feature=l40s
#PBS -l mem=80gb
#PBS -l walltime=96:00:00
#PBS -N utonia_sem_seg
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

cp $PBS_O_WORKDIR/Utonia/demo/utonia_sem_seg.py $TMPDIR
cp $PBS_O_WORKDIR/data/scan_20_opt_denoised.ply $TMPDIR
cp $PBS_O_WORKDIR/utonia_lidarnet.pt $TMPDIR

python $TMPDIR/utonia_sem_seg.py --ckpt_path $TMPDIR/utonia_lidarnet.pt --input_path $TMPDIR/scan_20_opt_denoised.ply --output_path $TMPDIR/utonia_res.ply

cp $TMPDIR/utonia_res.ply $PBS_O_WORKDIR

rm -rf $TMPDIR

exit
