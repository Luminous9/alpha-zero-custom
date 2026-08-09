"""Target construction and matched supervised screening utilities for V4."""

from dataclasses import dataclass
import math

import numpy as np

from .SantoriniGame import SantoriniGame
from .V4BootstrapCorpus import V4BootstrapDataset
from .V4Encoder import encode_v4_boards


GLOBAL_SCORE_TEMPERATURE = 261.8
DEFAULT_ALPHA_BOOT = 0.50
DEFAULT_STAGE_RELIABILITY = (0.25, 0.75, 1.0)


def score_to_value(score, temperature=GLOBAL_SCORE_TEMPERATURE):
    score = float(score)
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Score temperature must be finite and positive.")
    if score >= 9_000:
        return 1.0
    if score <= -9_000:
        return -1.0
    scaled = float(np.clip(score / temperature, -50.0, 50.0))
    return float(2.0 / (1.0 + math.exp(-scaled)) - 1.0)


def blended_value_target(
    winner_value,
    score,
    stage_id,
    alpha_boot=DEFAULT_ALPHA_BOOT,
    stage_reliability=DEFAULT_STAGE_RELIABILITY,
    stage_aware=True,
    temperature=GLOBAL_SCORE_TEMPERATURE,
):
    alpha = float(alpha_boot)
    if not 0 <= alpha <= 1:
        raise ValueError("alpha_boot must be in [0, 1].")
    if len(stage_reliability) != 3 or any(
        not np.isfinite(value) or value < 0 or value > 1
        for value in stage_reliability
    ):
        raise ValueError("Stage reliability must contain three values in [0, 1].")
    if int(stage_id) not in range(3):
        raise ValueError("stage_id must be 0, 1, or 2.")
    if stage_aware:
        alpha *= float(stage_reliability[int(stage_id)])
    return float(
        (1.0 - alpha) * float(winner_value)
        + alpha * score_to_value(score, temperature)
    )


def smooth_engine_policy(game, board, teacher_policy, epsilon):
    teacher_policy = np.asarray(teacher_policy, dtype=np.float32)
    epsilon = float(epsilon)
    if not 0 <= epsilon < 1:
        raise ValueError("Policy smoothing epsilon must be in [0, 1).")
    valids = game.getValidMoves(board, 1).astype(bool)
    if np.any(teacher_policy[~valids] > 1e-7):
        raise ValueError("Teacher policy assigns mass to an illegal action.")
    total = float(teacher_policy.sum())
    if not np.isclose(total, 1.0, atol=1e-5):
        raise ValueError("Teacher policy does not sum to one.")
    teacher_policy = teacher_policy / total
    support = teacher_policy > 0
    alternatives = valids & ~support
    if not epsilon or not np.any(alternatives):
        return teacher_policy.astype(np.float32)
    smoothed = teacher_policy * (1.0 - epsilon)
    smoothed[alternatives] = epsilon / int(np.sum(alternatives))
    return smoothed.astype(np.float32)


@dataclass
class PreparedV4Corpus:
    encoded_boards: np.ndarray
    policies: np.ndarray
    winner_values: np.ndarray
    score_values: np.ndarray
    stage_blended_values: np.ndarray
    global_blended_values: np.ndarray
    stage_ids: np.ndarray
    source_ids: np.ndarray
    split_ids: np.ndarray

    def value_targets(self, target):
        if target == "winner":
            return self.winner_values
        if target == "stage_blend":
            return self.stage_blended_values
        if target == "global_blend":
            return self.global_blended_values
        raise ValueError("Unknown V4 value target: {}".format(target))


def prepare_corpus(
    engine_path,
    run13_path,
    plan_path,
    expected_split,
    policy_epsilon=0.05,
    alpha_boot=DEFAULT_ALPHA_BOOT,
    stage_reliability=DEFAULT_STAGE_RELIABILITY,
    temperature=GLOBAL_SCORE_TEMPERATURE,
):
    game = SantoriniGame(5, sequential_placement=True)
    boards = []
    policies = []
    winner_values = []
    score_values = []
    stage_blended_values = []
    global_blended_values = []
    stage_ids = []
    source_ids = []
    split_ids = []
    with V4BootstrapDataset(engine_path, run13_path, plan_path) as dataset:
        for index in range(len(dataset)):
            example = dataset[index]
            if int(example.split_id) != int(expected_split):
                raise ValueError("Sampling plan contains an example from the wrong split.")
            policy = example.policy
            if example.source_id < 2:
                policy = smooth_engine_policy(
                    game, example.board, policy, policy_epsilon
                )
            else:
                valids = game.getValidMoves(example.board, 1).astype(bool)
                if np.any(policy[~valids] > 1e-7):
                    raise ValueError("Run13 policy assigns mass to an illegal action.")
                total = float(policy.sum())
                if not np.isfinite(total) or total <= 0:
                    raise ValueError("Run13 policy must have positive finite mass.")
                policy = (policy / total).astype(np.float32)
            score_value = score_to_value(example.score, temperature)
            boards.append(example.board)
            policies.append(policy)
            winner_values.append(example.winner_value)
            score_values.append(score_value)
            stage_blended_values.append(blended_value_target(
                example.winner_value,
                example.score,
                example.stage_id,
                alpha_boot,
                stage_reliability,
                True,
                temperature,
            ))
            global_blended_values.append(blended_value_target(
                example.winner_value,
                example.score,
                example.stage_id,
                alpha_boot,
                stage_reliability,
                False,
                temperature,
            ))
            stage_ids.append(example.stage_id)
            source_ids.append(example.source_id)
            split_ids.append(example.split_id)
    return PreparedV4Corpus(
        encoded_boards=encode_v4_boards(boards),
        policies=np.asarray(policies, dtype=np.float32),
        winner_values=np.asarray(winner_values, dtype=np.float32),
        score_values=np.asarray(score_values, dtype=np.float32),
        stage_blended_values=np.asarray(stage_blended_values, dtype=np.float32),
        global_blended_values=np.asarray(global_blended_values, dtype=np.float32),
        stage_ids=np.asarray(stage_ids, dtype=np.int8),
        source_ids=np.asarray(source_ids, dtype=np.int8),
        split_ids=np.asarray(split_ids, dtype=np.int8),
    )


def apply_d4_augmentation(boards, policies, symmetry_ids, game):
    boards = np.asarray(boards)
    policies = np.asarray(policies)
    symmetry_ids = np.asarray(symmetry_ids)
    if len(boards) != len(policies) or len(boards) != len(symmetry_ids):
        raise ValueError("Boards, policies, and symmetry IDs must have equal length.")
    if np.any((symmetry_ids < 0) | (symmetry_ids >= 8)):
        raise ValueError("D4 symmetry IDs must be in [0, 7].")
    transformed_boards = np.empty_like(boards)
    transformed_policies = np.empty_like(policies)
    for symmetry_id in range(8):
        selected = np.flatnonzero(symmetry_ids == symmetry_id)
        if not len(selected):
            continue
        rotations, flip = divmod(symmetry_id, 2)
        spatial = np.rot90(boards[selected], rotations, axes=(-2, -1))
        if flip:
            spatial = np.flip(spatial, axis=-1)
        transformed_boards[selected] = spatial
        old_indices, new_indices = game.getPolicySymmetryPermutation(rotations, bool(flip))
        transformed_policies[np.ix_(selected, new_indices)] = policies[np.ix_(selected, old_indices)]
    return transformed_boards, transformed_policies
