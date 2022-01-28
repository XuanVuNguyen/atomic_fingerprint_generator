# Extract atomic fingerprints from molecules using pretrained GROVER
Using pretrained GROVER to extract the atomic fingerprints from molecule. The fingerprints can be used for further tasks.

GROVER is short for Graph Representation frOm self-superVised mEssage passing tRansformer which is a Transformer-based self-supervised message-passing neural network by Rong and colleagues as in the paper: [Self-Supervised Graph Transformer on Large-Scale Molecular Data](https://arxiv.org/abs/2007.02835).

# Intalling requirements
1. Create and activate a conda environment:
```
conda create --name grover python=3.6.8
conda activate grover
```
2. Install requirements from `requirements.txt` file:
```
conda install -c conda-forge -c pytorch -c acellera -c RMG --file=requirements.txt
```

# Download the pretrained model
There are two pretrained models provided by the original authors. Download, extract and save the `.pt` file in `models_pretrained/`. 
* [GROVER<sub>base</sub>](https://ai.tencent.com/ailab/ml/ml-data/grover-models/pretrain/grover_base.tar.gz)
* [GROVER<sub>large</sub>](https://ai.tencent.com/ailab/ml/ml-data/grover-models/pretrain/grover_large.tar.gz)

# Inference figerprints
Run the `main.py` file:
```
python main.py
```
Details about the arguments can be viewed in the `setup_parser()` function found in the `main.py`, or by running:
```
python main.py -h
```
If no arguments are specified, then the default arguments are used.

By default, the outputs are saved in `extracted_fingerprint`. The outputs include 3 files:
* `atom_fp.npy`: contains the atomic fingerprints.
* `distance.npy`: contains the shortest relative distance matrices between nodes of the molecular graphs.
* `smiles.txt`: contains the SMILES strings of the molecules.

In order to read the `.npy` files, please refer to: [this part in the `numpy.save` documentation](https://numpy.org/doc/stable/reference/generated/numpy.save.html#:~:text=with%20open(%27test.npy%27%2C%20%27wb%27)%20as%20f%3A%0A...%20%20%20%20%20np.save(f%2C%20np.array(%5B1%2C%202%5D))%0A...%20%20%20%20%20np.save(f%2C%20np.array(%5B1%2C%203%5D))%0A%3E%3E%3E%20with%20open(%27test.npy%27%2C%20%27rb%27)%20as%20f%3A%0A...%20%20%20%20%20a%20%3D%20np.load(f)%0A...%20%20%20%20%20b%20%3D%20np.load(f)%0A%3E%3E%3E%20print(a%2C%20b)%0A%23%20%5B1%202%5D%20%5B1%203%5D)