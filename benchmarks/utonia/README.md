```bash
micromamba create -n matiss_utonia -c conda-forge python=3.10 pip -y
micromamba activate matiss_utonia

python -m pip install --upgrade pip wheel
python -m pip uninstall -y setuptools
python -m pip install "setuptools<82"

python -m pip install \
  torch==2.5.0 \
  torchvision==0.20.0 \
  torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install spconv-cu118
python -m pip install ninja packaging
python -m pip install flash-attn --no-build-isolation
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
python -m pip install huggingface_hub timm trimesh
python -m pip install addict scipy plyfile

cd /path/to/Utonia
python -m pip install -e . --no-build-isolation
```
