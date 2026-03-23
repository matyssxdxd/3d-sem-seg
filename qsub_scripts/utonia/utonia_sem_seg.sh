#!/bin/sh -v
#PBS -e /mnt/home/valtersve/matiss_sem_seg/logs
#PBS -o /mnt/home/valtersve/matiss_sem_seg/logs
#PBS -q batch
#PBS -l nodes=1:ppn=4:gpus=1,feature=l40s
#PBS -l mem=80gb
#PBS -l walltime=96:00:00
#PBS -N utonia_sem_seg
#PBS -W x=HOSTLIST:wn74

export MAMBA_EXE='/mnt/beegfs2/beegfs_scratch/valtersve/.mamba/bin/micromamba';
export MAMBA_ROOT_PREFIX='/mnt/beegfs2/beegfs_scratch/valtersve/.mamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from micromamba activate
fi
unset __mamba_setup

micromamba activate matiss_utonia

echo Working Directory: $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)

export TMPDIR=/mnt/beegfs2/beegfs_scratch/$USER/$PBS_JOBID

beegfs-ctl --getquota --uid "$(id -u valtersve)"

mkdir -m 700 -p /mnt/beegfs2/beegfs_scratch/$USER/$PBS_JOBID

cp $PBS_O_WORKDIR/Utonia/demo/utonia_sem_seg.py $TMPDIR
cp $PBS_O_WORKDIR/data/scan_20_opt_denoised.ply $TMPDIR
cp $PBS_O_WORKDIR/utonia_lidarnet.pt $TMPDIR

python $TMPDIR/utonia_sem_seg.py --input_path $TMPDIR/scan_20_opt_denoised.ply --output_path $TMPDIR/utonia_res.ply --ckpt_path $TMPDIR/utonia_lidarnet.pt

cp $TMPDIR/utonia_res.ply $PBS_O_WORKDIR

rm -rf $TMPDIR

exit
