import logging
import os
import random
import sys

import numpy as np
from tqdm import tqdm

sys.path.append('../..')
from NeuralNet import NeuralNet
from utils import AverageMeter, dotdict

import torch
import torch.optim as optim

from .SantoriniNNet import SantoriniNNet

log = logging.getLogger(__name__)


args = dotdict({
    'lr': 0.001,
    'lr_schedule': [],
    'optimizer': 'adam',
    'weight_decay': 0.0,
    'dropout': 0.2,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'input_channels': 6,
    'num_channels': 64,
    'num_residual_blocks': 5,
    'value_hidden_size': 128,
    'max_train_steps': None,
    'replay_reuse': None,
    'on_the_fly_symmetry': False,
    'quiet': False,
})


ARCHITECTURES = {
    'v2': {
        'num_channels': 64,
        'num_residual_blocks': 5,
        'policy_channels': 64,
        'action_encoding': 'spatial64',
    },
    'v3': {
        'num_channels': 96,
        'num_residual_blocks': 8,
        'policy_channels': 65,
        'action_encoding': 'spatial65-placement',
    },
}


class NNetWrapper(NeuralNet):
    architecture = 'v2'

    def __init__(self, game):
        self.game = game
        self.net_args = self._make_net_args()
        self.nnet = SantoriniNNet(game, self.net_args)
        _, self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.native_action_size = self.board_x * self.board_y * self.net_args.policy_channels

        if self.net_args.cuda:
            self.nnet.cuda()
        optimizer_class = optim.AdamW if self.net_args.optimizer == 'adamw' else optim.Adam
        self.optimizer = optimizer_class(
            self.nnet.parameters(),
            lr=self.net_args.lr,
            weight_decay=self.net_args.weight_decay,
        )

    def _make_net_args(self):
        # Runtime configuration historically uses attribute assignment on dotdict.
        # Copy through getattr so values set before construction are not lost when
        # the backing dictionary still contains its module default.
        net_args = dotdict({key: getattr(args, key) for key in args})
        architecture = ARCHITECTURES[self.architecture]
        net_args.num_channels = architecture['num_channels']
        net_args.num_residual_blocks = architecture['num_residual_blocks']
        net_args.policy_channels = architecture['policy_channels']
        net_args.action_encoding = architecture['action_encoding']
        return net_args

    def _sync_runtime_args(self):
        self.net_args.lr = args.lr
        self.net_args.lr_schedule = list(args.lr_schedule)
        self.net_args.optimizer = args.optimizer
        self.net_args.weight_decay = args.weight_decay
        self.net_args.dropout = args.dropout
        self.net_args.epochs = args.epochs
        self.net_args.batch_size = args.batch_size
        self.net_args.max_train_steps = args.max_train_steps
        self.net_args.replay_reuse = args.replay_reuse
        self.net_args.on_the_fly_symmetry = args.on_the_fly_symmetry
        self.net_args.quiet = args.quiet
        return self.net_args

    @staticmethod
    def encode_board(board):
        pieces = board[0]
        heights = board[1]
        encoded = np.zeros((args.input_channels, pieces.shape[0], pieces.shape[1]), dtype=np.float32)

        encoded[0] = pieces > 0
        encoded[1] = pieces < 0
        encoded[2] = heights == 1
        encoded[3] = heights == 2
        encoded[4] = heights == 3
        encoded[5] = heights >= 4

        return encoded

    @classmethod
    def encode_boards(cls, boards):
        return np.array([cls.encode_board(board) for board in boards], dtype=np.float32)

    def train(self, examples, new_example_count=None, validation_examples=None, iteration=None):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        runtime_args = self._sync_runtime_args()
        if not examples:
            raise ValueError('Cannot train without training examples.')
        learning_rate = self._learning_rate_for_iteration(iteration, runtime_args)
        for parameter_group in self.optimizer.param_groups:
            parameter_group['lr'] = learning_rate
            parameter_group['weight_decay'] = runtime_args.weight_decay
        encoded_boards, target_pis, target_vs = self._encode_training_examples(examples)

        symmetry_multiplier = 8 if runtime_args.on_the_fly_symmetry else 1
        virtual_example_count = len(examples) * symmetry_multiplier
        if runtime_args.replay_reuse is not None:
            if new_example_count is None or int(new_example_count) < 1:
                raise ValueError(
                    'Replay-reuse scheduling requires at least one newly generated training example.'
                )
            new_example_count = int(new_example_count)
            uncapped_training_steps = max(
                1,
                int(np.ceil(new_example_count * runtime_args.replay_reuse / runtime_args.batch_size)),
            )
            training_schedule = 'fresh-data-reuse'
        else:
            epoch_batch_count = max(1, int(np.ceil(virtual_example_count / runtime_args.batch_size)))
            uncapped_training_steps = runtime_args.epochs * epoch_batch_count
            training_schedule = 'legacy-epochs'
        training_steps = uncapped_training_steps
        if runtime_args.max_train_steps is not None:
            training_steps = min(training_steps, int(runtime_args.max_train_steps))
        epoch_batch_count = max(1, int(np.ceil(training_steps / runtime_args.epochs)))
        if runtime_args.quiet:
            log.info(
                'Optimizer schedule: mode=%s fresh=%s replay=%s steps=%s/%s lr=%g weight_decay=%g',
                training_schedule,
                new_example_count,
                len(examples),
                training_steps,
                uncapped_training_steps,
                learning_rate,
                runtime_args.weight_decay,
            )

        metrics = {}
        completed_steps = 0
        iteration_pi_losses = AverageMeter()
        iteration_v_losses = AverageMeter()
        for epoch in range(runtime_args.epochs):
            steps_this_epoch = min(epoch_batch_count, training_steps - completed_steps)
            if steps_this_epoch <= 0:
                break
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            if runtime_args.quiet:
                log.info(
                    'Training segment %s/%s (%s/%s batches; %s/%s total steps)',
                    epoch + 1,
                    runtime_args.epochs,
                    steps_this_epoch,
                    epoch_batch_count,
                    completed_steps + steps_this_epoch,
                    training_steps,
                )
            else:
                print('TRAINING SEGMENT ::: ' + str(epoch + 1))

            t = tqdm(range(steps_this_epoch), desc='Training Net', disable=runtime_args.quiet)
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=runtime_args.batch_size)
                batch_boards = encoded_boards[sample_ids]
                batch_pis = target_pis[sample_ids]
                if runtime_args.on_the_fly_symmetry:
                    symmetry_ids = np.random.randint(8, size=runtime_args.batch_size)
                    batch_boards, batch_pis = self._apply_symmetries(
                        batch_boards,
                        batch_pis,
                        symmetry_ids,
                    )
                boards = torch.from_numpy(batch_boards)
                batch_target_pis = torch.from_numpy(batch_pis)
                batch_target_vs = torch.from_numpy(target_vs[sample_ids])

                if self.net_args.cuda:
                    boards = boards.contiguous().cuda()
                    batch_target_pis = batch_target_pis.contiguous().cuda()
                    batch_target_vs = batch_target_vs.contiguous().cuda()

                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(batch_target_pis, out_pi)
                l_v = self.loss_v(batch_target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                iteration_pi_losses.update(l_pi.item(), boards.size(0))
                iteration_v_losses.update(l_v.item(), boards.size(0))
                if not runtime_args.quiet:
                    t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            completed_steps += steps_this_epoch

            if runtime_args.quiet:
                log.info(
                    'Finished training segment %s/%s: pi_loss=%.4f v_loss=%.4f',
                    epoch + 1,
                    runtime_args.epochs,
                    pi_losses.avg,
                    v_losses.avg,
                )
            metrics = {
                'policy_loss': float(pi_losses.avg),
                'value_loss': float(v_losses.avg),
                'total_loss': float(pi_losses.avg + v_losses.avg),
                'final_segment_policy_loss': float(pi_losses.avg),
                'final_segment_value_loss': float(v_losses.avg),
                'final_segment_total_loss': float(pi_losses.avg + v_losses.avg),
                'iteration_policy_loss': float(iteration_pi_losses.avg),
                'iteration_value_loss': float(iteration_v_losses.avg),
                'iteration_total_loss': float(
                    iteration_pi_losses.avg + iteration_v_losses.avg
                ),
                'epoch': epoch + 1,
                'training_segments_completed': epoch + 1,
                'training_steps': int(completed_steps),
                'uncapped_training_steps': int(uncapped_training_steps),
                'virtual_replay_examples': int(virtual_example_count),
                'training_examples': int(len(examples)),
                'symmetry_augmentation_multiplier': int(symmetry_multiplier),
                'effective_replay_epochs': float(
                    completed_steps * runtime_args.batch_size /
                    (len(examples) if runtime_args.replay_reuse is not None else virtual_example_count)
                ),
                'average_draws_per_stored_position': float(
                    completed_steps * runtime_args.batch_size / len(examples)
                ),
                'base_replay_epochs': float(
                    completed_steps * runtime_args.batch_size / len(examples)
                ),
                'fresh_training_examples': (
                    int(new_example_count) if new_example_count is not None else None
                ),
                'target_replay_reuse': (
                    float(runtime_args.replay_reuse)
                    if runtime_args.replay_reuse is not None else None
                ),
                'actual_replay_reuse': (
                    float(completed_steps * runtime_args.batch_size / new_example_count)
                    if new_example_count else None
                ),
                'training_schedule': training_schedule,
                'learning_rate': float(learning_rate),
                'weight_decay': float(runtime_args.weight_decay),
                'optimizer': str(runtime_args.optimizer),
            }
        if runtime_args.quiet:
            log.info(
                'Finished optimizer iteration: steps=%s pi_loss=%.4f v_loss=%.4f total_loss=%.4f',
                completed_steps,
                iteration_pi_losses.avg,
                iteration_v_losses.avg,
                iteration_pi_losses.avg + iteration_v_losses.avg,
            )
        validation_metrics = self._validation_metrics(
            validation_examples if validation_examples is not None else []
        )
        metrics.update(validation_metrics)
        if runtime_args.quiet and validation_metrics.get('validation_examples'):
            log.info(
                'Held-out validation (%s positions): placement policy_kl=%s value_loss=%s; '
                'standard policy_kl=%s value_loss=%s',
                validation_metrics['validation_examples'],
                self._format_metric(validation_metrics.get('placement_validation_policy_kl')),
                self._format_metric(validation_metrics.get('placement_validation_value_loss')),
                self._format_metric(validation_metrics.get('standard_validation_policy_kl')),
                self._format_metric(validation_metrics.get('standard_validation_value_loss')),
            )
        return metrics

    @staticmethod
    def _format_metric(value):
        return 'n/a' if value is None else '{:.4f}'.format(value)

    @staticmethod
    def _learning_rate_for_iteration(iteration, runtime_args):
        learning_rate = float(runtime_args.lr)
        if iteration is None:
            return learning_rate
        for start_iteration, scheduled_rate in sorted(runtime_args.lr_schedule):
            if int(iteration) >= int(start_iteration):
                learning_rate = float(scheduled_rate)
        return learning_rate

    def _validation_metrics(self, examples):
        metrics = {'validation_examples': int(len(examples))}
        if not examples:
            return metrics

        accumulators = {
            phase: {
                'count': 0,
                'policy_loss': 0.0,
                'target_entropy': 0.0,
                'value_loss': 0.0,
                'policy_top1': 0,
                'value_sign': 0,
            }
            for phase in ('placement', 'standard')
        }
        batch_size = max(1, int(self.net_args.batch_size))
        for offset in range(0, len(examples), batch_size):
            batch = examples[offset:offset + batch_size]
            boards, target_pis, target_vs = list(zip(*batch))
            predicted_pis, predicted_vs = self.predict_batch(boards)
            for board, target_pi, target_v, predicted_pi, predicted_v in zip(
                boards,
                target_pis,
                target_vs,
                predicted_pis,
                predicted_vs,
            ):
                phase = 'placement' if (
                    hasattr(self.game, 'isPlacementPhase') and self.game.isPlacementPhase(board)
                ) else 'standard'
                bucket = accumulators[phase]
                target_pi = np.asarray(target_pi, dtype=np.float64)
                predicted_pi = np.asarray(predicted_pi, dtype=np.float64)
                positive = target_pi > 0
                policy_loss = -float(
                    np.sum(target_pi[positive] * np.log(np.maximum(predicted_pi[positive], 1e-12)))
                )
                target_entropy = -float(np.sum(target_pi[positive] * np.log(target_pi[positive])))
                valids = np.asarray(self.game.getValidMoves(board, 1), dtype=bool)
                legal_prediction = np.where(valids, predicted_pi, -np.inf)

                bucket['count'] += 1
                bucket['policy_loss'] += policy_loss
                bucket['target_entropy'] += target_entropy
                bucket['value_loss'] += float((float(target_v) - float(predicted_v)) ** 2)
                predicted_action = int(np.argmax(legal_prediction))
                bucket['policy_top1'] += int(
                    target_pi[predicted_action] >= np.max(target_pi) - 1e-12
                )
                bucket['value_sign'] += int((float(target_v) >= 0) == (float(predicted_v) >= 0))

        total_count = 0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        for phase, bucket in accumulators.items():
            count = bucket['count']
            metrics['{}_validation_examples'.format(phase)] = int(count)
            if not count:
                continue
            prefix = '{}_validation_'.format(phase)
            metrics[prefix + 'policy_loss'] = bucket['policy_loss'] / count
            metrics[prefix + 'policy_kl'] = (
                bucket['policy_loss'] - bucket['target_entropy']
            ) / count
            metrics[prefix + 'value_loss'] = bucket['value_loss'] / count
            metrics[prefix + 'policy_top1_accuracy'] = bucket['policy_top1'] / count
            metrics[prefix + 'value_sign_accuracy'] = bucket['value_sign'] / count
            total_count += count
            total_policy_loss += bucket['policy_loss']
            total_value_loss += bucket['value_loss']
        metrics['validation_policy_loss'] = total_policy_loss / total_count
        metrics['validation_value_loss'] = total_value_loss / total_count
        metrics['validation_total_loss'] = (
            metrics['validation_policy_loss'] + metrics['validation_value_loss']
        )
        return metrics

    def _apply_symmetries(self, encoded_boards, target_pis, symmetry_ids):
        """Apply selected D4 symmetries to an encoded board/policy batch."""
        encoded_boards = np.asarray(encoded_boards)
        target_pis = np.asarray(target_pis)
        symmetry_ids = np.asarray(symmetry_ids, dtype=np.int8)
        if len(encoded_boards) != len(target_pis) or len(encoded_boards) != len(symmetry_ids):
            raise ValueError('Boards, policies, and symmetry ids must have matching lengths.')

        transformed_boards = np.empty_like(encoded_boards)
        transformed_pis = np.empty_like(target_pis)
        for symmetry_id in range(8):
            batch_indices = np.flatnonzero(symmetry_ids == symmetry_id)
            if len(batch_indices) == 0:
                continue
            rotations = symmetry_id // 2
            flip = bool(symmetry_id % 2)
            boards = np.rot90(
                encoded_boards[batch_indices],
                rotations,
                axes=(-2, -1),
            )
            if flip:
                boards = np.flip(boards, axis=-1)
            transformed_boards[batch_indices] = boards

            old_indices, new_indices = self.game.getPolicySymmetryPermutation(rotations, flip)
            transformed_pis[
                batch_indices[:, None],
                new_indices[None, :],
            ] = target_pis[
                batch_indices[:, None],
                old_indices[None, :],
            ]
        return transformed_boards, transformed_pis

    def _encode_training_examples(self, examples):
        boards, pis, vs = list(zip(*examples))
        return (
            self.encode_boards(boards),
            np.array(pis, dtype=np.float32),
            np.array(vs, dtype=np.float32),
        )

    def predict(self, board):
        """
        board: canonical Santorini rules board with shape (2, n, n)
        """
        pis, vs = self.predict_batch([board])
        return pis[0], float(vs[0])

    def predict_batch(self, boards):
        """
        boards: iterable of canonical Santorini rules boards with shape (2, n, n)
        """
        encoded = torch.FloatTensor(self.encode_boards(boards))
        if self.net_args.cuda:
            encoded = encoded.contiguous().cuda()

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(encoded)

        policies = torch.exp(pi).data.cpu().numpy()
        if policies.shape[1] != self.action_size:
            policies = self._adapt_native_policies(policies)
        return policies, v.view(-1).data.cpu().numpy()

    def _adapt_native_policies(self, policies):
        game_local_actions = getattr(self, 'action_size', 0) // (self.board_x * self.board_y)
        native_local_actions = self.net_args.policy_channels
        if native_local_actions != 64 or game_local_actions != 65:
            raise ValueError(
                'Cannot adapt {}-channel {} policy to game action size {}.'.format(
                    native_local_actions,
                    self.architecture,
                    self.action_size,
                )
            )
        adapted = np.zeros((len(policies), self.action_size), dtype=policies.dtype)
        adapted_view = adapted.reshape(len(policies), self.board_x * self.board_y, game_local_actions)
        adapted_view[:, :, :64] = policies.reshape(len(policies), self.board_x * self.board_y, 64)
        return adapted

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', include_optimizer=False, metadata=None):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            if not self.net_args.quiet:
                print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.makedirs(folder)
        else:
            if not self.net_args.quiet:
                print("Checkpoint Directory exists! ")
        payload = {
            'state_dict': self.nnet.state_dict(),
            'architecture': self.architecture,
            'num_channels': self.net_args.num_channels,
            'num_residual_blocks': self.net_args.num_residual_blocks,
            'policy_channels': self.net_args.policy_channels,
            'action_encoding': self.net_args.action_encoding,
            'optimizer': self.net_args.optimizer,
        }
        if include_optimizer:
            payload['optimizer_state_dict'] = self.optimizer.state_dict()
            payload['numpy_rng_state'] = np.random.get_state()
            payload['python_rng_state'] = random.getstate()
            payload['torch_rng_state'] = torch.get_rng_state()
            if torch.cuda.is_available():
                payload['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        if metadata:
            payload['training_metadata'] = dict(metadata)
        torch.save(payload, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', load_optimizer=False):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("No model in path {}".format(filepath))
        map_location = None if self.net_args.cuda else 'cpu'
        try:
            checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filepath, map_location=map_location)
        checkpoint_architecture = checkpoint.get('architecture')
        if checkpoint_architecture is not None and checkpoint_architecture != self.architecture:
            raise ValueError(
                'Checkpoint architecture "{}" cannot be loaded into "{}".'.format(
                    checkpoint_architecture,
                    self.architecture,
                )
            )
        checkpoint_encoding = checkpoint.get('action_encoding')
        if checkpoint_encoding is not None and checkpoint_encoding != self.net_args.action_encoding:
            raise ValueError(
                'Checkpoint action encoding "{}" cannot be loaded into "{}".'.format(
                    checkpoint_encoding,
                    self.net_args.action_encoding,
                )
            )
        try:
            self.nnet.load_state_dict(checkpoint['state_dict'])
        except RuntimeError as exc:
            raise RuntimeError(
                'Checkpoint at {} does not match Santorini architecture "{}" '
                '({} blocks, {} channels).'.format(
                    filepath,
                    self.architecture,
                    self.net_args.num_residual_blocks,
                    self.net_args.num_channels,
                )
            ) from exc
        if load_optimizer:
            optimizer_state = checkpoint.get('optimizer_state_dict')
            if optimizer_state is None:
                log.warning(
                    'Checkpoint %s has no optimizer state; %s will resume fresh.',
                    filepath,
                    self.net_args.optimizer,
                )
            else:
                checkpoint_optimizer = checkpoint.get('optimizer', 'adam')
                if checkpoint_optimizer != self.net_args.optimizer:
                    log.info(
                        'Migrating compatible %s optimizer state into %s.',
                        checkpoint_optimizer,
                        self.net_args.optimizer,
                    )
                self.optimizer.load_state_dict(optimizer_state)
                for parameter_group in self.optimizer.param_groups:
                    for key, value in self.optimizer.defaults.items():
                        parameter_group.setdefault(key, value)
            if 'numpy_rng_state' in checkpoint:
                np.random.set_state(checkpoint['numpy_rng_state'])
            if 'python_rng_state' in checkpoint:
                random.setstate(checkpoint['python_rng_state'])
            if 'torch_rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['torch_rng_state'])
            if self.net_args.cuda and 'cuda_rng_state_all' in checkpoint:
                self._restore_cuda_rng_states(checkpoint['cuda_rng_state_all'])
        return checkpoint.get('training_metadata', {})

    @staticmethod
    def _restore_cuda_rng_states(saved_states):
        saved_states = list(saved_states)
        available_devices = torch.cuda.device_count()
        if len(saved_states) != available_devices:
            log.warning(
                'Checkpoint contains CUDA RNG state for %s device(s), but this runtime exposes %s; '
                'restoring the first %s compatible state(s).',
                len(saved_states),
                available_devices,
                min(len(saved_states), available_devices),
            )
        for device_index, state in enumerate(saved_states[:available_devices]):
            torch.cuda.set_rng_state(state, device=device_index)


class V3NNetWrapper(NNetWrapper):
    architecture = 'v3'


def build_nnet(game, architecture='v2'):
    if architecture == 'v2':
        return NNetWrapper(game)
    if architecture == 'v3':
        return V3NNetWrapper(game)
    raise ValueError("Unknown Santorini neural architecture: {}".format(architecture))
