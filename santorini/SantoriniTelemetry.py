import numpy as np


class ReferenceSuite:
    def __init__(self, path):
        self.path = path
        with np.load(path, allow_pickle=False) as payload:
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
