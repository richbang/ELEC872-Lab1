# ELEC 872 Week 4 Workshop: robust LBP face recognition

Daeho Bang, Yujong Cho

The baseline LBP face recogniser (LBP<sup>u2</sup><sub>16,2</sub>, 30x37 windows, chi-square,
nearest neighbour; Ahonen et al.) is kept unchanged. Two steps are added so that accuracy
stays high under uneven lighting and in-plane rotation:

1. **Brightness-plane subtraction** (Sung & Poggio, 1998): fit `a*x + b*y + c` to the image by
   least squares (clipped 0/255 pixels left out) and subtract it.
2. **Rotation search**: store every gallery image rotated by -24..24 deg (6 deg steps) and use
   the smallest chi-square over the copies.

## Results (ORL, images 1-5 gallery / 6-10 probe, rank-1)

| method | normal | bright | light 0.2 | 0.4 | 0.6 | 0.8 | rot 5 | rot 10 | rot 20 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.975 | 0.975 | 0.955 | 0.940 | 0.870 | 0.705 | 0.965 | 0.890 | 0.630 |
| ours | 0.995 | 0.980 | 0.985 | 0.990 | 0.985 | 0.985 | 0.995 | 0.980 | 0.940 |

Normal photos, paper protocol (100 random 5+5 splits): baseline 0.983, ours 0.983.
Full tables, including conditions never used for tuning: [results/workshop.md](results/workshop.md).

## Code

- [`lbpface/robust.py`](lbpface/robust.py): the two added steps (`RobustLBP`)
- [`experiments/workshop.py`](experiments/workshop.py): the evaluation (split and test conditions)
- [`tests/test_robust.py`](tests/test_robust.py), [`tests/test_workshop.py`](tests/test_workshop.py)

Everything else is the course's baseline code, unchanged.

## Data

The ORL database (AT&T Laboratories Cambridge) is not included. Put it at
`data/ORL/s1/1.pgm ... data/ORL/s40/10.pgm`, e.g.

```
git clone --depth 1 https://github.com/saeid436/Face-Recognition-MLP.git orl_src
cp -r orl_src/ORL data/ORL
```

## Run

```
pip install -r requirements.txt
python -m experiments.workshop --methods baseline        # baseline only (~20 s)
python -m experiments.workshop --extra --n-perm 100      # all methods and extra conditions (~10 min)
python -m pytest -q
```
