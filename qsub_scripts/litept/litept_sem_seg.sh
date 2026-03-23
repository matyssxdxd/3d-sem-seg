#!/bin/sh -v
#PBS -e /mnt/home/valtersve/matiss_sem_seg/logs
#PBS -o /mnt/home/valtersve/matiss_sem_seg/logs
#PBS -q batch
#PBS -l nodes=1:ppn=4:gpus=1,feature=l40s
#PBS -l mem=80gb
#PBS -l walltime=96:00:00
#PBS -N litept_sem_seg
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

micromamba activate matiss_litept

module load gcc/11.2.0
export LD_LIBRARY_PATH=/mnt/opt/exp_soft/spack/opt/spack/linux-centos7-x86_64/gcc-8.3.0/gcc-11.2.0-hbazyfn6aayq2jlpog7uck7w5xa5nmm4/lib64:$LD_LIBRARY_PATH

echo Working Directory: $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)

export TMPDIR=/mnt/beegfs2/beegfs_scratch/$USER/$PBS_JOBID

mkdir -m 700 -p $TMPDIR

echo "Copying LitePT repo..."
cp -r $PBS_O_WORKDIR/LitePT $TMPDIR/

echo "Copying script and data..."
cp $PBS_O_WORKDIR/litept_sem_seg.py $TMPDIR/LitePT/
cp $PBS_O_WORKDIR/data/scan_20_opt_denoised.ply $TMPDIR/LitePT/

cd $TMPDIR/LitePT

echo "Running inference..."
python litept_sem_seg.py --input_path scan_20_opt_denoised.ply --output_file $TMPDIR/litept_res.ply

cp $TMPDIR/litept_res.ply $PBS_O_WORKDIR

echo "Cleaning up..."
rm -rf $TMPDIR

exit
