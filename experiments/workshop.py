"""Workshop (Oct. 6): keep the ORL LBP recogniser working under uneven lighting and rotation.

Protocol
    Fixed split: images 1-5 of every subject form the gallery, images 6-10 the probes
    (200 + 200).  Only the probes are changed:

    normal        unchanged
    bright        0.5 * I + 20
    light s       I + 255 * s * (x - 0.5), x = column / (width - 1); s = 0.2, 0.4, 0.6, 0.8
    rot a         bilinear rotation by a degrees about the centre, black corners; a = 5, 10, 20

    Every changed image is rounded and clipped to 0..255.  These definitions were
    reconstructed from the workshop description; they reproduce its quoted baseline
    numbers (0.705 for light 0.8, 0.630 for rot 20).

Methods (see ``lbpface/robust.py``): baseline, +plane, +rotation, +plane+rotation.
``--extra`` adds conditions that none of the settings were chosen on.  ``--n-perm``
also runs the paper's random-split protocol (5 + 5 images, unchanged photos).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

import numpy as np

from lbpface import repro
from lbpface.classify import recognition_rate
from lbpface.datasets import load_orl
from lbpface.robust import RobustLBP, rotate
from lbpface.stats import permutation_rates, summarise

HANDOUT = {"light 0.8": 0.705, "rot 20": 0.630}  # baseline numbers quoted in the workshop text

METHODS = {
    "baseline": RobustLBP(plane=False, angles=(0.0,)),
    "+plane": RobustLBP(plane=True, angles=(0.0,)),
    "+rotation": RobustLBP(plane=False),
    "+plane+rotation": RobustLBP(plane=True),
}


def u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def rot(images: np.ndarray, angle: float) -> np.ndarray:
    return np.stack([u8(rotate(im, angle, fill="constant")) for im in images])


def make_probes(P: np.ndarray, extra: bool = False) -> dict:
    """Condition name -> changed probe images (uint8, ``(n, H, W)``)."""
    h, w = P.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w]
    x, y = xx / (w - 1), yy / (h - 1)
    I = P.astype(np.float64)
    probes = {"normal": P, "bright": u8(I * 0.5 + 20)}
    probes.update({f"light {s}": u8(I + 255 * s * (x - 0.5)) for s in (0.2, 0.4, 0.6, 0.8)})
    probes.update({f"rot {a}": rot(P, a) for a in (5, 10, 20)})
    if extra:
        probes.update({f"rot {a}": rot(P, a) for a in (-20, -10, 15, 25, 30)})
        probes["light 0.8 reversed"] = u8(I - 255 * 0.8 * (x - 0.5))
        probes["light 0.8 vertical"] = u8(I + 255 * 0.8 * (y - 0.5))
        probes["light 0.8 multiplicative"] = u8(I * (0.2 + 1.6 * x))
        probes["spotlight"] = u8(I - 40 + 160 * np.exp(-((x - 0.15) ** 2 + (y - 0.35) ** 2) / (2 * 0.25 ** 2)))
        probes["light 0.6 + rot 10"] = rot(u8(I + 255 * 0.6 * (x - 0.5)), 10)
    return probes


def run(data: str = "data/ORL", methods=tuple(METHODS), extra: bool = False, n_perm: int = 0,
        seed: int = 0, verbose: bool = True):
    """Returns ``(config, dataset_fingerprint, results)``."""
    fs = load_orl(data)
    gal = np.array([int(os.path.splitext(os.path.basename(n))[0]) <= 5 for n in fs.names])
    G, gl, P, pl = fs.images[gal], fs.subjects[gal], fs.images[~gal], fs.subjects[~gal]
    probes = make_probes(P, extra)
    results = {"rank1": {}}
    for m in methods:
        model = METHODS[m]
        gf = model.gallery_features(G)
        row = {c: recognition_rate(model.distances(model.features(imgs), gf), gl, pl) for c, imgs in probes.items()}
        results["rank1"][m] = row
        if verbose:
            print(f"{m:16s} " + " ".join(f"{c}={r:.3f}" for c, r in row.items()), flush=True)
    if n_perm:
        results["permutation_normal"] = {}
        for m in methods:
            model = METHODS[m]
            d = model.distances(model.features(fs.images), model.gallery_features(fs.images))
            s = summarise(permutation_rates({m: d}, fs.subjects, n_perm, seed, 5, 5)[m])
            results["permutation_normal"][m] = s
            if verbose:
                print(f"{m:16s} random 5+5 splits: mean {s['mean']:.4f} std {s['std']:.4f}", flush=True)
    config = {"data": "ORL", "seed": seed, "split": "images 1-5 gallery, 6-10 probe",
              "conditions": list(probes), "methods": {m: asdict(METHODS[m]) for m in methods},
              "n_perm": n_perm}
    return config, fs.fingerprint, results


def markdown(results) -> str:
    rows = results["rank1"]
    conds = list(next(iter(rows.values())))
    lines = ["| method | " + " | ".join(conds) + " |", "|---" * (len(conds) + 1) + "|"]
    lines += [f"| {m} | " + " | ".join(f"{r[c]:.3f}" for c in conds) + " |" for m, r in rows.items()]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/ORL")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--extra", action="store_true", help="also run conditions nothing was tuned on")
    ap.add_argument("--n-perm", type=int, default=0, help="random-split permutations on normal photos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/workshop.json")
    a = ap.parse_args(argv)
    config, fp, results = run(a.data, tuple(a.methods), a.extra, a.n_perm, a.seed)
    for c, v in HANDOUT.items():
        if "baseline" in results["rank1"]:
            print(f"handout baseline {c}: {v:.3f}   reproduced: {results['rank1']['baseline'][c]:.3f}")
    doc = repro.write_results(a.out, "workshop", config, fp, results)
    md = os.path.splitext(a.out)[0] + ".md"
    with open(md, "w") as f:
        f.write(markdown(results) + "\n")
    print(f"wrote {a.out} and {md}   digest {doc['result_digest'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
