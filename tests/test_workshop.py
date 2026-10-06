"""Workshop protocol on the real ORL data: the reconstructed conditions match the handout's
baseline numbers, and the robust recogniser beats them without losing normal photos."""
from experiments import workshop

CONDS = ("normal", "light 0.8", "rot 20")


def _rates(orl_dir, method):
    *_, res = workshop.run(orl_dir, methods=(method,), verbose=False)
    return res["rank1"][method]


def test_baseline_reproduces_the_handout(orl_dir):
    r = _rates(orl_dir, "baseline")
    assert r["light 0.8"] == workshop.HANDOUT["light 0.8"]
    assert r["rot 20"] == workshop.HANDOUT["rot 20"]
    assert r["normal"] >= 0.97


def test_robust_recogniser_keeps_normal_and_improves_the_hard_conditions(orl_dir):
    base = _rates(orl_dir, "baseline")
    rob = _rates(orl_dir, "+plane+rotation")
    assert rob["normal"] >= base["normal"]
    assert rob["light 0.8"] >= base["light 0.8"] + 0.2
    assert rob["rot 20"] >= base["rot 20"] + 0.2
