"""Frozen deep-oracle value telemetry for V4 P2 training.

The labels are fixed engine proxies, not solved-game truth.  The useful signal
is therefore paired drift from the iteration-11 reference on the same boards.
"""

import hashlib
import json

import numpy as np


SCHEMA_VERSION = 1
STAGES = ("early", "middle", "late")
WINDOW_BANDS = ("windows_1_4", "windows_5_8", "windows_9_11", "windows_12_14")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(payload, name):
    value = np.asarray(payload[name])
    if value.size != 1:
        raise ValueError("{} must contain one scalar.".format(name))
    return value.reshape(-1)[0].item()


def _validate_suite(suite):
    required = {
        "schema_version", "boards", "oracle_values", "reference_values",
        "stages", "bands", "position_hashes", "metadata",
    }
    missing = sorted(required - set(suite))
    if missing:
        raise ValueError("Deep-value telemetry suite is missing: {}".format(
            ", ".join(missing)
        ))
    if int(_scalar(suite, "schema_version")) != SCHEMA_VERSION:
        raise ValueError("Unsupported deep-value telemetry schema.")
    boards = np.asarray(suite["boards"])
    positions = len(boards)
    if boards.shape != (positions, 2, 5, 5) or positions < 12:
        raise ValueError("Deep-value boards must have shape (N, 2, 5, 5).")
    for name in (
        "oracle_values", "reference_values", "stages", "bands",
        "position_hashes",
    ):
        if np.asarray(suite[name]).shape != (positions,):
            raise ValueError("{} must contain one entry per board.".format(name))
    if not np.all(np.isfinite(suite["oracle_values"])):
        raise ValueError("Deep-value oracle labels must be finite.")
    if not np.all(np.isfinite(suite["reference_values"])):
        raise ValueError("Deep-value reference predictions must be finite.")
    stages = set(map(str, suite["stages"]))
    bands = set(map(str, suite["bands"]))
    if stages != set(STAGES):
        raise ValueError("Deep-value telemetry has unexpected stage strata.")
    if bands != set(WINDOW_BANDS):
        raise ValueError("Deep-value telemetry has unexpected window strata.")
    if len(set(map(str, suite["position_hashes"]))) != positions:
        raise ValueError("Deep-value telemetry position hashes are not unique.")
    metadata = suite["metadata"]
    if isinstance(metadata, np.ndarray):
        metadata = metadata.item()
    metadata = json.loads(str(metadata))
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Deep-value metadata has the wrong schema.")
    if metadata.get("reference_iteration") != 11:
        raise ValueError("Deep-value reference must be P2 iteration 11.")
    return metadata


def load_deep_value_telemetry_suite(path):
    with np.load(path, allow_pickle=False) as payload:
        suite = {name: payload[name].copy() for name in payload.files}
    suite["metadata_parsed"] = _validate_suite(suite)
    suite["fingerprint"] = file_sha256(path)
    return suite


def _predict_values(nnet, boards, batch_size):
    values = []
    for start in range(0, len(boards), batch_size):
        _, batch_values = nnet.predict_batch(boards[start:start + batch_size])
        values.append(np.asarray(batch_values, dtype=np.float64).reshape(-1))
    result = np.concatenate(values)
    if result.shape != (len(boards),) or not np.all(np.isfinite(result)):
        raise ValueError("Deep-value predictor returned invalid values.")
    return result


def _pearson(predictions, labels):
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(predictions) < 2 or predictions.std() == 0 or labels.std() == 0:
        return 0.0
    return float(np.corrcoef(predictions, labels)[0, 1])


def _summary(predictions, labels):
    residual = np.asarray(predictions, dtype=np.float64) - labels
    return {
        "pearson": _pearson(predictions, labels),
        "mse": float(np.mean(residual ** 2)),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def _paired_bootstrap(predictions, reference, labels, strata, samples, seed):
    """Stratified paired CIs for current-minus-reference metric changes."""
    rng = np.random.RandomState(int(seed))
    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    pearson_deltas = np.empty(samples, dtype=np.float64)
    mse_deltas = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        indices = np.concatenate([
            group[rng.randint(len(group), size=len(group))] for group in groups
        ])
        current_summary = _summary(predictions[indices], labels[indices])
        reference_summary = _summary(reference[indices], labels[indices])
        pearson_deltas[sample] = (
            current_summary["pearson"] - reference_summary["pearson"]
        )
        mse_deltas[sample] = current_summary["mse"] - reference_summary["mse"]
    return {
        "pearson_delta_95_low": float(np.quantile(pearson_deltas, 0.025)),
        "pearson_delta_95_high": float(np.quantile(pearson_deltas, 0.975)),
        "mse_delta_95_low": float(np.quantile(mse_deltas, 0.025)),
        "mse_delta_95_high": float(np.quantile(mse_deltas, 0.975)),
    }


def evaluate_deep_value_telemetry(
    nnet,
    suite,
    batch_size=256,
    bootstrap_samples=2000,
    seed=20260814,
    overall_pearson_drop=0.02,
    overall_mse_rise=0.015,
    recent_pearson_drop=0.04,
    recent_mse_rise=0.02,
):
    """Evaluate paired value drift from the frozen iteration-11 reference."""
    metadata = _validate_suite(suite)
    batch_size = int(batch_size)
    bootstrap_samples = int(bootstrap_samples)
    if batch_size < 1 or bootstrap_samples < 1:
        raise ValueError("Deep-value batch/bootstrap sizes must be positive.")
    thresholds = (
        overall_pearson_drop, overall_mse_rise,
        recent_pearson_drop, recent_mse_rise,
    )
    if any(float(value) < 0 for value in thresholds):
        raise ValueError("Deep-value warning thresholds cannot be negative.")

    boards = np.asarray(suite["boards"])
    labels = np.asarray(suite["oracle_values"], dtype=np.float64)
    reference = np.asarray(suite["reference_values"], dtype=np.float64)
    stages = np.asarray(suite["stages"]).astype(str)
    bands = np.asarray(suite["bands"]).astype(str)
    current = _predict_values(nnet, boards, batch_size)
    strata = np.char.add(np.char.add(bands, "/"), stages)

    metrics = {
        "v4_deep_value_telemetry_due": True,
        "v4_deep_value_suite_positions": int(len(boards)),
        "v4_deep_value_suite_fingerprint": suite.get("fingerprint", "in_memory"),
        "v4_deep_value_reference_iteration": int(metadata["reference_iteration"]),
        "v4_deep_value_reference_checkpoint_sha256": metadata.get(
            "reference_checkpoint_sha256"
        ),
        "v4_deep_value_overall_pearson_drop_threshold": float(overall_pearson_drop),
        "v4_deep_value_overall_mse_rise_threshold": float(overall_mse_rise),
        "v4_deep_value_recent_pearson_drop_threshold": float(recent_pearson_drop),
        "v4_deep_value_recent_mse_rise_threshold": float(recent_mse_rise),
    }

    def add_group(prefix, mask):
        current_summary = _summary(current[mask], labels[mask])
        reference_summary = _summary(reference[mask], labels[mask])
        metrics["{}_count".format(prefix)] = int(np.sum(mask))
        for name in ("pearson", "mse", "mae", "bias"):
            metrics["{}_{}".format(prefix, name)] = current_summary[name]
            metrics["{}_reference_{}".format(prefix, name)] = reference_summary[name]
            metrics["{}_{}_delta".format(prefix, name)] = (
                current_summary[name] - reference_summary[name]
            )

    all_positions = np.ones(len(boards), dtype=bool)
    add_group("v4_deep_value_overall", all_positions)
    for stage in STAGES:
        add_group("v4_deep_value_stage_{}".format(stage), stages == stage)
    for band in WINDOW_BANDS:
        add_group("v4_deep_value_{}".format(band), bands == band)

    metrics.update({
        "v4_deep_value_overall_pearson_warning": bool(
            metrics["v4_deep_value_overall_pearson_delta"]
            <= -float(overall_pearson_drop)
        ),
        "v4_deep_value_overall_mse_warning": bool(
            metrics["v4_deep_value_overall_mse_delta"]
            >= float(overall_mse_rise)
        ),
        "v4_deep_value_recent_pearson_warning": bool(
            metrics["v4_deep_value_windows_9_11_pearson_delta"]
            <= -float(recent_pearson_drop)
        ),
        "v4_deep_value_recent_mse_warning": bool(
            metrics["v4_deep_value_windows_9_11_mse_delta"]
            >= float(recent_mse_rise)
        ),
    })
    metrics["v4_deep_value_warning"] = bool(any(
        metrics[name] for name in (
            "v4_deep_value_overall_pearson_warning",
            "v4_deep_value_overall_mse_warning",
            "v4_deep_value_recent_pearson_warning",
            "v4_deep_value_recent_mse_warning",
        )
    ))
    metrics.update({
        "v4_deep_value_overall_{}".format(name): value
        for name, value in _paired_bootstrap(
            current, reference, labels, strata, bootstrap_samples, seed
        ).items()
    })
    recent = bands == "windows_9_11"
    metrics.update({
        "v4_deep_value_recent_{}".format(name): value
        for name, value in _paired_bootstrap(
            current[recent], reference[recent], labels[recent],
            stages[recent], bootstrap_samples, int(seed) ^ 0x911,
        ).items()
    })
    return metrics
