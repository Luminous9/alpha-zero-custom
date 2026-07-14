import os

import numpy as np


def resolve_reference_suite_path(path):
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if os.path.isfile(path):
        return path
    if not os.path.exists(path):
        raise FileNotFoundError('Reference suite does not exist: {}'.format(path))
    if not os.path.isdir(path):
        raise ValueError('Reference suite path is neither a file nor a directory: {}'.format(path))

    candidates = []
    for root, _, filenames in os.walk(path):
        candidates.extend(
            os.path.join(root, filename)
            for filename in filenames
            if filename.lower().endswith('.npz')
        )
    candidates.sort()
    if not candidates:
        raise FileNotFoundError('No .npz reference suite found under directory: {}'.format(path))
    if len(candidates) > 1:
        preferred = [candidate for candidate in candidates if os.path.basename(candidate) == 'v2_reference_500.npz']
        if len(preferred) == 1:
            return preferred[0]
        raise ValueError(
            'Multiple .npz files found under reference-suite directory; pass the exact file path: {}'.format(
                ', '.join(candidates)
            )
        )
    return candidates[0]


class ReferenceSuite:
    def __init__(self, path):
        self.path = resolve_reference_suite_path(path)
        with np.load(self.path, allow_pickle=False) as payload:
            self.boards = payload['boards'].astype(int)
            self.policies = payload['policies'].astype(np.float32)
            self.values = payload['values'].astype(np.float32)
            self.stages = payload['stages'].astype(np.int8)

    def evaluate(self, game, nnet, batch_size=256):
        predicted_policies = []
        predicted_values = []
        for start in range(0, len(self.boards), batch_size):
            policies, values = nnet.predict_batch(self.boards[start:start + batch_size])
            predicted_policies.append(policies)
            predicted_values.append(values)
        predicted_policies = np.concatenate(predicted_policies)
        predicted_values = np.concatenate(predicted_values)

        if predicted_policies.shape[1] == 1625 and self.policies.shape[1] == 1600:
            comparable = predicted_policies.reshape(-1, 25, 65)[:, :, :64].reshape(-1, 1600)
        elif predicted_policies.shape[1] == self.policies.shape[1]:
            comparable = predicted_policies
        else:
            raise ValueError(
                'Reference policy size {} is incompatible with network policy size {}.'.format(
                    self.policies.shape[1],
                    predicted_policies.shape[1],
                )
            )

        epsilon = 1e-12
        cross_entropy = -np.mean(np.sum(self.policies * np.log(comparable + epsilon), axis=1))
        target_log = np.where(self.policies > 0, np.log(self.policies + epsilon), 0.0)
        kl = np.mean(np.sum(self.policies * (target_log - np.log(comparable + epsilon)), axis=1))
        top1 = np.mean(np.argmax(self.policies, axis=1) == np.argmax(comparable, axis=1))
        value_mse = np.mean((self.values - predicted_values) ** 2)

        legal_masses = []
        normalized_entropies = []
        for board, policy in zip(self.boards, predicted_policies):
            valids = game.getValidMoves(board, 1).astype(bool)
            legal = policy[valids]
            mass = float(legal.sum())
            legal_masses.append(mass)
            if mass > 0 and len(legal) > 1:
                normalized = legal / mass
                entropy = -float(np.sum(normalized * np.log(normalized + epsilon))) / np.log(len(legal))
                normalized_entropies.append(entropy)

        return {
            'reference_policy_cross_entropy': float(cross_entropy),
            'reference_policy_kl': float(kl),
            'reference_top1_accuracy': float(top1),
            'reference_value_mse': float(value_mse),
            'reference_legal_policy_mass': float(np.mean(legal_masses)),
            'reference_normalized_legal_entropy': float(np.mean(normalized_entropies)),
        }
