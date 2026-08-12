"""Frozen canonical-seam regression telemetry for V4 self-play.

The suite deliberately retains the P1c supervised targets and its per-position
losses.  P2 can therefore change its training targets without changing the
meaning of this diagnostic.
"""

import hashlib
import json

import numpy as np

from santorini.D4Canonical import canonicalize_board
from santorini.SantoriniOracle import anonymous_board_key


IDENTITY_TRANSFORM = (0, False)
SCHEMA_VERSION = 1
POLICY_WEIGHT = 0.25
METRIC_NAMES = ("objective", "policy_loss", "value_loss", "top1")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_selection_boards(engine_path, run13_path, plan_path):
    """Load the raw rules boards addressed by a selection-only frozen plan."""
    with np.load(plan_path, allow_pickle=False) as plan, np.load(
        engine_path, allow_pickle=False
    ) as engine, np.load(run13_path, allow_pickle=False) as run13:
        if np.any(plan["split_ids"] != 1):
            raise ValueError("Seam diagnostic requires a selection-only plan.")
        boards = []
        for corpus_id, position_index in zip(
            plan["corpus_ids"], plan["position_indices"]
        ):
            payload = engine if int(corpus_id) == 0 else run13
            position_index = int(position_index)
            if int(payload["split_ids"][position_index]) != 1:
                raise ValueError("Selection plan points outside the selection split.")
            boards.append(payload["boards"][position_index].astype(np.int8))
    return boards


def seam_profile(game, board):
    """Return unique-successor frame-switch exposure for one canonical board."""
    _, current_transforms, _ = canonicalize_board(board)
    if IDENTITY_TRANSFORM not in current_transforms:
        raise ValueError("Seam input board is not in its D4 canonical frame.")
    successors = {}
    valid_actions = np.flatnonzero(game.getValidMoves(board, 1))
    for action in valid_actions:
        next_board, next_player = game.getNextState(board, 1, int(action))
        next_canonical = game.getCanonicalForm(next_board, next_player)
        successor_key = anonymous_board_key(next_canonical)
        if successor_key in successors:
            continue
        _, transforms, _ = canonicalize_board(next_canonical)
        successors[successor_key] = IDENTITY_TRANSFORM not in transforms
    if not successors:
        raise ValueError("Selection seam input has no legal successors.")
    switches = int(sum(successors.values()))
    return {
        "legal_actions": int(len(valid_actions)),
        "unique_successors": int(len(successors)),
        "frame_switch_successors": switches,
        "frame_switch_exposure": switches / float(len(successors)),
        "current_stabilizer_size": int(len(current_transforms)),
    }


def quartile_buckets(exposures):
    """Assign stable equal-count quartiles without data-driven cutoffs."""
    exposures = np.asarray(exposures, dtype=np.float64)
    if exposures.ndim != 1 or not len(exposures):
        raise ValueError("Exposure values must be a nonempty vector.")
    order = np.argsort(exposures, kind="stable")
    buckets = np.empty(len(exposures), dtype=np.int8)
    buckets[order] = np.minimum(3, 4 * np.arange(len(exposures)) // len(exposures))
    return buckets


def policies_to_csr(policies):
    policies = np.asarray(policies, dtype=np.float32)
    if policies.ndim != 2:
        raise ValueError("Policy targets must be a matrix.")
    rows, columns = np.nonzero(policies)
    values = policies[rows, columns]
    indptr = np.zeros(len(policies) + 1, dtype=np.int64)
    np.add.at(indptr, rows + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, columns.astype(np.int32), values.astype(np.float32)


def _validate_suite(suite):
    required = {
        "boards", "policy_indptr", "policy_indices", "policy_values",
        "value_targets", "exposures", "exposure_quartiles",
        "baseline_policy_loss", "baseline_value_loss", "baseline_objective",
        "baseline_top1", "metadata",
    }
    missing = sorted(required - set(suite))
    if missing:
        raise ValueError("Seam telemetry suite is missing: {}".format(", ".join(missing)))
    boards = np.asarray(suite["boards"])
    positions = len(boards)
    if boards.shape != (positions, 2, 5, 5) or positions < 4:
        raise ValueError("Seam telemetry boards must have shape (N, 2, 5, 5), N >= 4.")
    indptr = np.asarray(suite["policy_indptr"])
    indices = np.asarray(suite["policy_indices"])
    values = np.asarray(suite["policy_values"])
    if indptr.shape != (positions + 1,) or indptr[0] != 0 or indptr[-1] != len(indices):
        raise ValueError("Invalid sparse policy offsets in seam telemetry suite.")
    if len(indices) != len(values) or np.any(np.diff(indptr) <= 0):
        raise ValueError("Every seam telemetry position must have a nonempty policy target.")
    if np.any(indices < 0):
        raise ValueError("Seam telemetry policy indices cannot be negative.")
    for name in (
        "value_targets", "exposures", "exposure_quartiles",
        "baseline_policy_loss", "baseline_value_loss", "baseline_objective",
        "baseline_top1",
    ):
        if np.asarray(suite[name]).shape != (positions,):
            raise ValueError("{} must contain one value per position.".format(name))
    buckets = np.asarray(suite["exposure_quartiles"], dtype=np.int8)
    if np.any((buckets < 0) | (buckets > 3)) or np.any(np.bincount(buckets, minlength=4) == 0):
        raise ValueError("Seam telemetry requires four nonempty exposure quartiles.")
    metadata = suite["metadata"]
    if isinstance(metadata, np.ndarray):
        metadata = metadata.item()
    metadata = json.loads(str(metadata))
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported seam telemetry suite schema.")
    return metadata


def load_seam_telemetry_suite(path):
    with np.load(path, allow_pickle=False) as payload:
        suite = {name: payload[name].copy() for name in payload.files}
    suite["metadata_parsed"] = _validate_suite(suite)
    suite["fingerprint"] = file_sha256(path)
    return suite


def _loss_vectors_from_predictions(policies, values, suite):
    policies = np.asarray(policies, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    positions = len(suite["boards"])
    if policies.ndim != 2 or policies.shape[0] != positions or len(values) != positions:
        raise ValueError("Seam telemetry predictor returned the wrong batch shape.")
    indptr = np.asarray(suite["policy_indptr"], dtype=np.int64)
    indices = np.asarray(suite["policy_indices"], dtype=np.int64)
    targets = np.asarray(suite["policy_values"], dtype=np.float64)
    if len(indices) and int(np.max(indices)) >= policies.shape[1]:
        raise ValueError("Seam telemetry policy target exceeds predictor action size.")
    policy_loss = np.empty(positions, dtype=np.float64)
    target_top1 = np.empty(positions, dtype=np.int64)
    for row in range(positions):
        start, end = int(indptr[row]), int(indptr[row + 1])
        action_indices = indices[start:end]
        action_targets = targets[start:end]
        policy_loss[row] = -np.dot(
            action_targets,
            np.log(np.maximum(policies[row, action_indices], 1e-12)),
        )
        target_top1[row] = action_indices[int(np.argmax(action_targets))]
    value_loss = (
        values - np.asarray(suite["value_targets"], dtype=np.float64)
    ) ** 2
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "objective": POLICY_WEIGHT * policy_loss + value_loss,
        "top1": (np.argmax(policies, axis=1) == target_top1).astype(np.float64),
    }


def evaluate_loss_vectors(nnet, suite, batch_size=256):
    """Evaluate a predict_batch-compatible network on the frozen suite."""
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("Seam telemetry batch size must be positive.")
    policies = []
    values = []
    boards = suite["boards"]
    for start in range(0, len(boards), batch_size):
        batch_policies, batch_values = nnet.predict_batch(boards[start:start + batch_size])
        policies.append(np.asarray(batch_policies))
        values.append(np.asarray(batch_values).reshape(-1))
    return _loss_vectors_from_predictions(
        np.concatenate(policies), np.concatenate(values), suite
    )


def summarize_loss_vectors(metrics, buckets):
    buckets = np.asarray(buckets, dtype=np.int8)
    return {
        "overall": {
            name: float(np.mean(metrics[name])) for name in METRIC_NAMES
        },
        "by_quartile": {
            str(bucket + 1): {
                name: float(np.mean(metrics[name][buckets == bucket]))
                for name in METRIC_NAMES
            }
            for bucket in range(4)
        },
    }


def _contrast_interval(excess_objective, buckets, samples, seed):
    low = np.asarray(excess_objective, dtype=np.float64)[buckets == 0]
    high = np.asarray(excess_objective, dtype=np.float64)[buckets == 3]
    rng = np.random.RandomState(int(seed))
    contrasts = []
    remaining = int(samples)
    while remaining:
        count = min(500, remaining)
        low_means = low[rng.randint(len(low), size=(count, len(low)))].mean(axis=1)
        high_means = high[rng.randint(len(high), size=(count, len(high)))].mean(axis=1)
        contrasts.append(high_means - low_means)
        remaining -= count
    return tuple(map(float, np.quantile(np.concatenate(contrasts), (0.025, 0.975))))


def evaluate_seam_telemetry(
    nnet,
    suite,
    batch_size=256,
    bootstrap_samples=2000,
    seed=20260818,
    alert_delta=0.02,
):
    """Return flat scalar telemetry against the frozen P1c seam baseline."""
    metadata = _validate_suite(suite)
    if int(bootstrap_samples) < 1:
        raise ValueError("Seam telemetry bootstrap sample count must be positive.")
    if float(alert_delta) < 0:
        raise ValueError("Seam telemetry alert delta cannot be negative.")
    current = evaluate_loss_vectors(nnet, suite, batch_size=batch_size)
    buckets = np.asarray(suite["exposure_quartiles"], dtype=np.int8)
    baseline = {
        name: np.asarray(suite["baseline_{}".format(name)], dtype=np.float64)
        for name in METRIC_NAMES
    }
    current_summary = summarize_loss_vectors(current, buckets)
    baseline_summary = summarize_loss_vectors(baseline, buckets)
    current_contrast = (
        current_summary["by_quartile"]["4"]["objective"]
        - current_summary["by_quartile"]["1"]["objective"]
    )
    baseline_contrast = (
        baseline_summary["by_quartile"]["4"]["objective"]
        - baseline_summary["by_quartile"]["1"]["objective"]
    )
    contrast_delta = current_contrast - baseline_contrast
    excess_objective = current["objective"] - baseline["objective"]
    interval = _contrast_interval(
        excess_objective, buckets, bootstrap_samples, seed
    )
    metrics = {
        "v4_seam_telemetry_due": True,
        "v4_seam_suite_positions": int(len(buckets)),
        "v4_seam_suite_fingerprint": suite.get("fingerprint", "in_memory"),
        "v4_seam_baseline_checkpoint_sha256": metadata.get(
            "baseline_checkpoint_sha256"
        ),
        "v4_seam_objective": current_summary["overall"]["objective"],
        "v4_seam_policy_loss": current_summary["overall"]["policy_loss"],
        "v4_seam_value_loss": current_summary["overall"]["value_loss"],
        "v4_seam_top1": current_summary["overall"]["top1"],
        "v4_seam_objective_delta_from_baseline": float(np.mean(excess_objective)),
        "v4_seam_high_minus_low_objective_contrast": float(current_contrast),
        "v4_seam_baseline_high_minus_low_objective_contrast": float(baseline_contrast),
        "v4_seam_contrast_delta_from_baseline": float(contrast_delta),
        "v4_seam_contrast_delta_bootstrap_95_low": interval[0],
        "v4_seam_contrast_delta_bootstrap_95_high": interval[1],
        "v4_seam_alert_delta": float(alert_delta),
        "v4_seam_warning": bool(contrast_delta > float(alert_delta)),
        "v4_seam_confirmed_warning": bool(
            contrast_delta > float(alert_delta) and interval[0] > 0.0
        ),
    }
    for bucket in range(4):
        label = "q{}".format(bucket + 1)
        for name in METRIC_NAMES:
            metrics["v4_seam_{}_{}".format(label, name)] = (
                current_summary["by_quartile"][str(bucket + 1)][name]
            )
        mask = buckets == bucket
        metrics["v4_seam_{}_objective_delta_from_baseline".format(label)] = float(
            np.mean(excess_objective[mask])
        )
    return metrics
