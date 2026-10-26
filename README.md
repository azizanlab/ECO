# ECO: Energy-Constrained Operator Learning for Chaotic Dynamics with Boundedness Guarantees

Code for **ECO: Energy-Constrained Operator Learning for Chaotic Dynamics with Boundedness Guarantees**.

ECO augments a neural operator (DeepONet or FNO) with a projection layer based on a learned energy
$V(w) = (w - x_0)^\top Q (w - x_0)$. Each prediction is projected so that $V$ cannot grow outside the
ellipsoid $V \le c^2$, and states inside the ellipsoid stay inside it. This makes the ellipsoid an absorbing
set and keeps long autoregressive rollouts bounded. $Q$ and $x_0$ are parameterized as functions of the
spatial coordinate, so the constraint is discretization invariant. The repository contains experiments on
three systems:

| Folder | System | Backbones |
|---|---|---|
| `L63/` | Lorenz-63 | MLP |
| `KS/` | Kuramoto–Sivashinsky (1D) | DeepONet, FNO |
| `KF/` | Kolmogorov flow (2D Navier–Stokes), Re = 40 and Re = 500 | DeepONet, FNO |

## Installation

Python 3.9 or newer is required.

```bash
git clone https://github.com/azizanlab/ECO.git
cd ECO
pip install torch numpy scipy matplotlib tqdm scikit-learn seaborn gstools
```

A CUDA GPU is recommended for training and evaluating the KS and KF models. Everything also runs on CPU.

## Data and trained models

The KS and KF datasets and all trained models are available on Zenodo:
**https://doi.org/10.5281/zenodo.23672714**

| File | Contents | Size |
|---|---|---|
| `KS_data.zip` | KS training data, and test trajectories on 512- and 1024-point grids | 145 MB |
| `KF_data.zip` | KF training data (220 trajectories each at Re = 40 and Re = 500), and long test trajectories | 13 GB |
| `Trained_Models.zip` | Trained checkpoints for every model in the paper, with saved rollouts | 3.4 GB |

The L63 data is small and is included in this repository under `L63/Data/`.

Download the zip files into the repository root, then run:

```bash
unzip -q KS_data.zip -x '__MACOSX/*' '*.DS_Store' && mv KS_data KS/Data
unzip -q KF_data.zip -x '__MACOSX/*' '*.DS_Store' && mv KF_data KF/data
unzip -q Trained_Models.zip -x '__MACOSX/*' '*.DS_Store'
```

The code expects this layout (all paths are relative to the repository root):

```
KS/Data/
    KS_data_batched_l100.53_grid512_M8_T500.0_dt0.01_amp5.0/data.npz                        # training
    KS_data_test_l100.53_grid512_M1_T2000.0_dt0.005_dt_sample0.2_amp20.0.npz/data.npz       # test, 512 grid
    KS_data_test_l100.53_grid1024_M1_T2000.0_dt0.005_dt_sample0.2_amp20.0/data.npz          # test, 1024 grid
KF/data/
    KF_Re40_M128_tsave1_dt0.0005_T500_n220/data.pt                                          # training, Re = 40
    KF_Re40_M128_tsave1_dt0.0005_T2000_n1/data.pt                                           # test, Re = 40
    KF_Re500_M128_tsave1_dt0.0001_T500_n220/data.pt                                         # training, Re = 500
    KF_Re500_M128_tsave1_dt0.0001_T2000_n1/data.pt                                          # test, Re = 500
Trained_Models/
    KS/{DeepONet,FNO}_{constrained,unconstrained}/
    KF/Re40_{FNO,deeponet}_{constrained,unconstrained}/
    KF/Re500_FNO_{constrained,unconstrained}/
```

Note that the 512-grid KS test folder name really does end in `.npz`.

Each folder in `Trained_Models/` contains `model_epoch_best.pt` (the checkpoint) and `model_params.npz`
(the architecture settings used to rebuild the model). "Constrained" models use the ECO projection
layer, and "unconstrained" models are the same backbone without it.
<!--
## Evaluating the trained models

Run each command from inside the system's folder.

**KS.** Evaluates on both the 512- and 1024-point test grids:

```bash
cd KS
python eval.py --model_dir ../Trained_Models/KS/DeepONet_constrained
```

Results are written to `eval_results_grid512/` and `eval_results_grid1024/` inside the model folder: one-step
predictions, the test rollout, a rollout from a random initial condition, the energy $V$ over time, PCA
projections with the learned ellipsoid, state and PCA distributions, and the Fourier spectrum. Time steps
where an unconstrained rollout blows up are left out of the distribution and PCA plots, and the number
dropped is printed.

**KF.** The second argument is the Reynolds number (`40` or `500`):

```bash
cd KF
python eval.py ../Trained_Models/KF/Re40_FNO_constrained/model_params.npz 40
```

Results are written to `eval_results_ICscale{30,1}_M64/` inside the model folder: one-step and rollout
animations, PCA modes, spatial correlation, PCA projections, distributions, the 2D Fourier spectrum and
energy over time.

**L63.**

```bash
cd L63
python eval.py --model-dir Trained_Models/<date>/<run_name>
```

## Training

Each `train.py` saves its run to `Trained_Models/<MMDD_HH>/<run_name>/` inside the system's folder. The
commands below rebuild the architectures of the models in `Trained_Models/`.
<!-- TODO: add the training hyperparameters used in the paper (--epochs, --lr, --bsize, --lam_reg_vol, and for KF --n_traj and --sched); these are not stored in model_params.npz. -->
<!--
**KS** (from `KS/`):

```bash
# DeepONet (note --trunk_scale 0.05; see below)
python train.py --backbone deeponet --trunk_scale 0.05 --output_dim 256 --branch_conv_channels 32 64 128 \
    --branch_fc_dims 256 --trunk_hidden_dims 256 256 256 --dt 1.0 \
    --project --diag_Q --discrete_proj --c_init 175.0                    # constrained
python train.py --backbone deeponet --trunk_scale 0.05 --output_dim 256 --branch_conv_channels 32 64 128 \
    --branch_fc_dims 256 --trunk_hidden_dims 256 256 256 --dt 1.0 \
    --no-nn_Q --no-nn_x0                                                 # unconstrained

# FNO
python train.py --backbone fno --dt 1.0 --project --diag_Q --discrete_proj --c_init 80.0   # constrained
python train.py --backbone fno --dt 1.0 --no-nn_Q --no-nn_x0                               # unconstrained
```

`--trunk_scale` multiplies the spatial grid before it is passed to the network (the DeepONet trunk, the FNO
grid channel, and the $Q$ and $x_0$ networks). The KS DeepONet models were trained with `0.05` and the FNO
models with the default `1.0`. It is saved in `model_params.npz`, and `eval.py` reads it from there.

**KF** (from `KF/`):

```bash
# FNO, Re = 40 (use --Re 500 for Re = 500, with --c_init 400.0 for the constrained model)
python train.py --Re 40 --backbone fno --project --c_init 300.0          # constrained
python train.py --Re 40 --backbone fno --no-nn_Q --no-nn_x0              # unconstrained

# DeepONet, Re = 40
python train.py --Re 40 --backbone deeponet --output_dim 1024 --branch_conv_channels 64 128 256 512 \
    --branch_fc_dims 1024 --trunk_hidden_dims 1024 1024 1024 1024 --project --diag_Q --c_init 400.0   # constrained
python train.py --Re 40 --backbone deeponet --output_dim 1024 --branch_conv_channels 64 128 256 512 \
    --branch_fc_dims 1024 --trunk_hidden_dims 1024 1024 1024 1024 --no-nn_Q --no-nn_x0                 # unconstrained
```

The training data is stored on a 128×128 grid and downsampled to 64×64 for training.

**L63** (from `L63/`):

```bash
python train.py --discrete-proj
```

Run `python train.py --help` in any folder to see all options.

## Generating data

- **L63:** `L63/gen_L63_data.py`.
- **KF:** `KF/KF_data_gen.py`. The Reynolds number, time step, length and number of trajectories are set at
  the top of the file, and it requires a CUDA GPU.
- **KS:** use the provided data on Zenodo.

## Citation
-->

<!-- TODO: BibTeX for the paper and the Zenodo record -->
