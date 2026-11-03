"""Predeclared, model-independent scoring and paired uncertainty calculations."""
from decimal import Decimal
import re

import numpy as np

NUMBER = r"[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
FINAL = re.compile(rf"####\s*({NUMBER})\s*\.?\s*")


def canonical_number(text):
    if not re.fullmatch(NUMBER, text):
        raise ValueError("Unsupported numeric answer")
    value = Decimal(text.replace(",", ""))
    if value == 0:
        return "0"
    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def final_answer(text):
    """Require exactly one #### answer line, at the end; no last-number guess."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    markers = [i for i, line in enumerate(lines) if line.startswith("####")]
    if not lines or markers != [len(lines) - 1]:
        return None
    match = FINAL.fullmatch(lines[-1])
    return canonical_number(match[1]) if match else None


def gsm_score(response, gold):
    answer = final_answer(response)
    return {"parsed_answer": answer, "format_valid": answer is not None,
            "correct": answer is not None and answer == canonical_number(gold)}


def _paired_arrays(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim == 1:
        left = left[None, :]
    if right.ndim == 1:
        right = right[None, :]
    if left.shape != right.shape or left.ndim != 2 or min(left.shape) == 0:
        raise ValueError("Paired nonempty seed-by-prompt matrices required")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Non-finite paired values")
    return left, right


def paired_family_interval(left, right, families, draws=10000, seed=4713,
                           resample_training_seeds=False):
    """Mean of paired differences; resample whole families with replacement.

    Each sampled family contributes all its prompts. The estimand weights
    prompts equally, not families equally. No completion is an independent
    prompt. Training seeds stay fixed unless explicitly resampled as a separate
    sensitivity analysis; three seeds give limited seed-population information.
    """
    left, right = _paired_arrays(left, right)
    if len(families) != left.shape[1] or draws < 1000:
        raise ValueError("Family count mismatch or fewer than 1000 bootstrap draws")
    if any(not isinstance(f, str) or not f for f in families):
        raise ValueError("Every prompt needs a nonempty family")
    groups = sorted(set(families))
    assignment = np.asarray([[f == group for group in groups] for f in families], dtype=float)
    family_sizes = assignment.sum(axis=0)
    difference = left - right
    family_sums = difference @ assignment
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=draws)
    if resample_training_seeds:
        selected = rng.integers(0, left.shape[0], size=(draws, left.shape[0]))
        sums = family_sums[selected].mean(axis=1)
    else:
        sums = family_sums.mean(axis=0)
    samples = (counts * sums).sum(axis=1) / (counts @ family_sizes)
    interval = np.quantile(samples, [0.025, 0.975])
    return {"difference": float(difference.mean()), "per_seed_differences": difference.mean(axis=1).tolist(),
            "paired_95_percent_interval": interval.tolist(), "prompt_count": left.shape[1],
            "family_count": len(groups), "training_seed_count": left.shape[0],
            "bootstrap_draws": draws, "bootstrap_seed": seed,
            "training_seeds_resampled": resample_training_seeds,
            "conditional_on_observed_completions": True,
            "degenerate_bootstrap": bool(np.ptp(samples) < 1e-12),
            "estimand": "Equal-weight mean over prompts and observed paired training seeds",
            "resampling_unit": "Whole prompt family, paired across models",
            "caution": "Few manually assigned prompt families; interval is conditional on this benchmark/split and observed sampled completions. A degenerate interval does not bound rare or unobserved differences and must not be presented as population equivalence."}


def paired_problem_interval(reference, intervention, draws=10000, seed=4714):
    """Positive loss means the intervention reduced exact-match accuracy."""
    reference, intervention = _paired_arrays(reference, intervention)
    if not np.isin(reference, [0, 1]).all() or not np.isin(intervention, [0, 1]).all():
        raise ValueError("Capability scores must be binary exact matches")
    n = reference.shape[1]
    rng = np.random.default_rng(seed)
    difference = (reference - intervention).mean(axis=0)
    selected = rng.integers(0, n, size=(draws, n))
    samples = difference[selected].mean(axis=1)
    interval = np.quantile(samples, [0.025, 0.975])
    return {"reference_accuracy": float(reference.mean()), "intervention_accuracy": float(intervention.mean()),
            "accuracy_loss": float(difference.mean()), "paired_95_percent_interval": interval.tolist(),
            "problem_count": n, "training_seed_count": reference.shape[0],
            "bootstrap_draws": draws, "bootstrap_seed": seed,
            "capability_preservation_within_5_points": bool(n >= 200 and interval[1] <= 0.05),
            "interpretation": "Paired problem bootstrap conditional on observed training seeds. A 200-problem damage screen, not a general capability guarantee."}
