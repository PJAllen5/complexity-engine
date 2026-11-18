"""Synthetic known-answer validation of the collective_behavior profile, roadmap item A8
(docs/design/validation-and-new-domains.md), on the Vicsek et al. (1995) model.

Each particle moves at constant speed and takes the mean heading of its neighbours within a
radius, plus uniform noise in [-eta/2, eta/2]. Polarization, |mean heading vector|, is the
order parameter: near 1 at low noise, near 0 at high noise, with a transition between.

Setup: 100 particles in a periodic 7 x 7 box (density ~2), speed 0.03, radius 1; noise
0.5, 1.5, 2.5, 3.5, 5.0; 2 replicate runs per noise level, 6,000 steps each after 1,000 burn-in.

Criteria, fixed before running:
1. Mean polarization above 0.9 at eta = 0.5 and below 0.25 at eta = 5.0 (100 random headings
   give ~0.09).
2. Mean polarization falls as noise rises (each step down by more than -0.02).
3. Near the transition, fluctuations are slowest: the mean DFA exponent of polarization (through
   the pipeline, collective_behavior profile) is highest at an intermediate noise (1.5-3.5),
   not at either end.
"""

import numpy as np
import pandas as pd
import pytest

from complexity_engine.analysis import analyze_dataset

NOISES = (0.5, 1.5, 2.5, 3.5, 5.0)
N, BOX, SPEED, RADIUS = 100, 7.0, 0.03, 1.0
BURN_IN, STEPS = 1000, 6000


def vicsek_polarization(noise, seed, n=N, box=BOX, steps=STEPS, burn_in=BURN_IN):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, box, (n, 2))
    theta = rng.uniform(-np.pi, np.pi, n)
    out = np.empty(steps)
    for t in range(burn_in + steps):
        delta = pos[:, None, :] - pos[None, :, :]
        delta -= box * np.round(delta / box)
        neighbours = (delta ** 2).sum(-1) <= RADIUS ** 2
        mean_heading = np.arctan2(neighbours @ np.sin(theta), neighbours @ np.cos(theta))
        theta = mean_heading + rng.uniform(-noise / 2, noise / 2, n)
        pos = (pos + SPEED * np.column_stack([np.cos(theta), np.sin(theta)])) % box
        if t >= burn_in:
            out[t - burn_in] = np.hypot(np.cos(theta).mean(), np.sin(theta).mean())
    return out


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    frames = [
        pd.DataFrame({"condition": f"eta_{noise}", "session": f"run{rep}", "frame": np.arange(STEPS),
                      "polarization": vicsek_polarization(noise, seed=10 * i + rep)})
        for i, noise in enumerate(NOISES) for rep in range(2)
    ]
    path = tmp_path_factory.mktemp("vicsek") / "frames.csv"
    data = pd.concat(frames)
    data.to_csv(path, index=False)
    result = analyze_dataset(path, signal_cols=["polarization"], unit_cols=["session"],
                             domain="collective_behavior", output_dir=path.parent / "out")
    units = result["complexity_by_unit"]
    return {
        "polarization": data.groupby("condition")["polarization"].mean(),
        "dfa": units.groupby("condition")["dfa_exponent"].mean(),
    }


def _by_noise(series):
    return np.array([series[f"eta_{noise}"] for noise in NOISES])


def test_order_at_low_noise_and_disorder_at_high_noise(sweep):
    polarization = _by_noise(sweep["polarization"])
    assert polarization[0] > 0.9 and polarization[-1] < 0.25


def test_polarization_falls_as_noise_rises(sweep):
    assert np.all(np.diff(_by_noise(sweep["polarization"])) < 0.02)


def test_fluctuations_are_slowest_near_the_transition(sweep):
    dfa = _by_noise(sweep["dfa"])
    assert int(np.argmax(dfa)) in (1, 2, 3)
