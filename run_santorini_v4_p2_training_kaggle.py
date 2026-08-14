"""Run a configurable guarded V4 P2 continuation from a resumable checkpoint."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


RUNTIME_CONTRACT = 'santorini_v4_p2_runtime'
RESULT_CONTRACT = 'santorini_v4_p2_training_result'
ITERATION1_CHECKPOINT_NAME = 'p2-iteration1-training.pth.tar'
VALUE_ANCHOR_NAME = 'p1c-value-anchor.pth.tar'
SEAM_SUITE_NAME = 'v4-seam-telemetry-suite.npz'
ORACLE_RELATIVE_PATH = Path('oracle-build/santorini-oracle-linux-x86_64')

TRUNK_LR = 1e-4
POLICY_LR = 1e-4
VALUE_LR = 1e-4
REPLAY_REUSE = 2.0
BRIDGE_START_ITERATION = 2
BRIDGE_END_ITERATION = 11
BRIDGE_START_BETA = 0.25
BRIDGE_END_BETA = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-manifest', required=True)
    parser.add_argument(
        '--source-root',
        help=(
            'Project checkout to execute. Defaults to the manifest-validated '
            'bundled source; a Kaggle git checkout may be supplied explicitly.'
        ),
    )
    parser.add_argument('--expected-source-commit')
    parser.add_argument('--resume-checkpoint', required=True)
    parser.add_argument('--resume-replay', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-iterations', type=int, required=True)
    parser.add_argument('--snapshot-interval', type=int, default=1)
    parser.add_argument('--arena-games', type=int, default=40)
    parser.add_argument('--no-end-arenas', action='store_true')
    parser.add_argument('--allow-non-p100', action='store_true')
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left, right, tolerance=1e-8):
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _bridge_beta(iteration):
    progress = min(1.0, max(0.0, (
        iteration - BRIDGE_START_ITERATION
    ) / (BRIDGE_END_ITERATION - BRIDGE_START_ITERATION)))
    return BRIDGE_START_BETA + progress * (
        BRIDGE_END_BETA - BRIDGE_START_BETA
    )


def _telemetry_rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line]
        if path.is_file() else []
    )


def _load_resume_state(checkpoint_path, replay_path, seam_hash, value_anchor_hash):
    import torch
    from santorini.ReplayBuffer import load_compact_replay

    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    metadata = dict(payload.get('training_metadata', {}))
    iteration = int(metadata.get('iteration', -1))
    if iteration < 1:
        raise ValueError('Resume checkpoint has no valid P2 iteration.')
    if metadata.get('training_mode') != 'latest':
        raise ValueError('Resume checkpoint is not latest-mode training state.')
    if 'optimizer_state_dict' not in payload:
        raise ValueError('Resume checkpoint lacks optimizer state.')
    for state_name in ('python_rng_state', 'numpy_rng_state', 'torch_rng_state'):
        if state_name not in payload:
            raise ValueError('Resume checkpoint lacks {}.'.format(state_name))
    if metadata.get('v4_seam_suite_fingerprint') != seam_hash:
        raise ValueError('Resume checkpoint uses a different frozen seam suite.')
    if metadata.get('v4_teacher_objective_current') is None:
        raise ValueError('Resume checkpoint lacks the current teacher objective.')
    if metadata.get('v4_teacher_objective_reference') is None:
        raise ValueError('Resume checkpoint lacks the cumulative teacher reference.')
    if int(metadata.get('oracle_sparring_nodes', -1)) != 5000:
        raise ValueError('Resume checkpoint does not use 5k oracle sparring.')
    if int(metadata.get('oracle_sparring_ladder_version', -1)) != 2:
        raise ValueError('Resume checkpoint does not use ladder version 2.')
    if not _close(metadata.get('replay_reuse'), REPLAY_REUSE):
        raise ValueError('Resume checkpoint does not use fixed 2x replay reuse.')
    if int(metadata.get('replay_reuse_warmup_iters', -1)) != 0:
        raise ValueError('Resume checkpoint unexpectedly uses a replay warm-up.')
    rates = metadata.get('optimizer_group_learning_rates', {})
    for group, expected in (
        ('trunk', TRUNK_LR),
        ('policy_head', POLICY_LR),
        ('value_head', VALUE_LR),
    ):
        if not _close(rates.get(group), expected):
            raise ValueError('Resume checkpoint has the wrong {} LR.'.format(group))
    if iteration < BRIDGE_END_ITERATION:
        if metadata.get('value_target_mode') != 'p1c_anchor_to_outcome_z':
            raise ValueError('Pre-iteration-11 resume state is not on Arm D.')
        if metadata.get('value_target_anchor_checkpoint_sha256') != value_anchor_hash:
            raise ValueError('Resume checkpoint uses the wrong value anchor.')
        if not _close(metadata.get('value_target_beta'), _bridge_beta(iteration)):
            raise ValueError('Resume checkpoint has the wrong bridge beta.')
    elif metadata.get('value_target_mode') == 'p1c_anchor_to_outcome_z':
        if not _close(metadata.get('value_target_beta'), 1.0):
            raise ValueError('Post-bridge resume state has a non-unit beta.')
    elif metadata.get('value_target_mode') != 'outcome_z':
        raise ValueError('Resume checkpoint has an unsupported value-target mode.')

    windows = load_compact_replay(replay_path)
    expected_windows = min(iteration, 20)
    if len(windows) != expected_windows or any(not window for window in windows):
        raise ValueError(
            'Resume replay has {} windows; expected {} for iteration {}.'.format(
                len(windows), expected_windows, iteration
            )
        )
    return metadata, windows


def _validate_runtime(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('contract') != RUNTIME_CONTRACT:
        raise ValueError('Wrong V4 runtime manifest contract.')
    root = manifest_path.parent
    for relative, expected in manifest.get('sources', {}).items():
        path = root / relative
        if not path.is_file() or _sha256(path) != expected['sha256']:
            raise ValueError('Runtime source digest changed: {}'.format(path))
    inputs = {}
    for name, expected in manifest['inputs'].items():
        path = root / 'inputs' / name
        if not path.is_file() or _sha256(path) != expected['sha256']:
            raise ValueError('Runtime input digest changed: {}'.format(path))
        inputs[name] = path
    oracle = root / ORACLE_RELATIVE_PATH
    if not oracle.is_file() or _sha256(oracle) != manifest['oracle_build']['linux_binary_sha256']:
        raise ValueError('Runtime oracle digest changed.')
    return manifest, root, inputs, oracle


def _validate_row(row, manifest, expected_iteration, previous_objective):
    reference = float(manifest['lineage']['teacher_objective_reference'])
    if int(row.get('iteration', -1)) != expected_iteration:
        raise RuntimeError('Trainer wrote the wrong iteration.')
    if int(row.get('games', -1)) != 240:
        raise RuntimeError('Iteration did not complete 240 games.')
    if int(row.get('oracle_sparring_games', -1)) != 24:
        raise RuntimeError('Iteration did not complete 24 sparring games.')
    if not _close(row.get('target_replay_reuse'), REPLAY_REUSE):
        raise RuntimeError('Iteration changed fixed replay reuse.')
    if int(row.get('replay_reuse_warmup_iters', -1)) != 0:
        raise RuntimeError('Iteration enabled replay warm-up.')
    if int(row.get('standard_prior_target_kl_count', 0)) < 256:
        raise RuntimeError('Iteration lacks sufficient prior-to-target KL telemetry.')
    if float(row.get('standard_prior_target_kl_mean', -1.0)) < 0.0:
        raise RuntimeError('Iteration reported invalid prior-to-target KL telemetry.')
    if not _close(row.get('prior_target_kl_warning_threshold'), 0.15):
        raise RuntimeError('Iteration changed the prior-to-target KL watch threshold.')
    if int(row.get('prior_target_kl_warning_iterations', -1)) != 3:
        raise RuntimeError('Iteration changed the prior-to-target KL watch persistence.')
    if row.get('v4_seam_suite_fingerprint') != manifest['inputs'][SEAM_SUITE_NAME]['sha256']:
        raise RuntimeError('Iteration used the wrong seam suite.')
    if not _close(row.get('v4_teacher_objective_previous'), previous_objective):
        raise RuntimeError('Iteration broke the teacher-objective chain.')
    if not _close(row.get('v4_teacher_objective_reference'), reference):
        raise RuntimeError('Iteration changed the cumulative teacher reference.')
    if not _close(row.get('v4_teacher_objective_step_threshold'), 0.05):
        raise RuntimeError('Iteration changed the teacher step gate.')
    if not _close(row.get('v4_teacher_objective_cumulative_threshold'), 0.10):
        raise RuntimeError('Iteration changed the cumulative teacher gate.')
    for key, expected in (
        ('trunk_learning_rate', TRUNK_LR),
        ('policy_head_learning_rate', POLICY_LR),
        ('value_head_learning_rate', VALUE_LR),
    ):
        if not _close(row.get(key), expected):
            raise RuntimeError('Iteration changed {}.'.format(key))
    if expected_iteration <= BRIDGE_END_ITERATION:
        if row.get('value_target_mode') != 'p1c_anchor_to_outcome_z':
            raise RuntimeError('Iteration left the bridge early.')
        if row.get('value_target_anchor_checkpoint_sha256') != manifest['inputs'][VALUE_ANCHOR_NAME]['sha256']:
            raise RuntimeError('Iteration used the wrong value anchor.')
        if not _close(row.get('value_target_beta'), _bridge_beta(expected_iteration)):
            raise RuntimeError('Iteration used the wrong bridge beta.')
    else:
        if row.get('value_target_mode') != 'outcome_z':
            raise RuntimeError('Post-bridge iteration did not use pure outcome z.')
        if not _close(row.get('value_target_beta'), 1.0):
            raise RuntimeError('Pure-z iteration reported non-unit beta.')
    current = float(row['v4_teacher_objective_current'])
    if not _close(
        row.get('v4_teacher_objective_cumulative_delta'), current - reference
    ):
        raise RuntimeError('Iteration reported an inconsistent cumulative delta.')
    return current


def _snapshot(output_dir, iteration):
    names = {
        'latest-training.pth.tar': 'checkpoint_{}-training.pth.tar'.format(iteration),
        'latest.pth.tar': 'checkpoint_{}.pth.tar'.format(iteration),
        'latest.examples.npz': 'checkpoint_{}.examples.npz'.format(iteration),
    }
    destinations = []
    for source_name, destination_name in names.items():
        source = output_dir / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_dir / destination_name
        shutil.copy2(source, destination)
        destinations.append(destination)
    return destinations


def _source_identity(project_root):
    required = ('main_santorini.py', 'arena_santorini_v4_p2_arm.py', 'Coach.py', 'MCTS.py')
    for name in required:
        if not (project_root / name).is_file():
            raise FileNotFoundError(project_root / name)
    identity = {'root': str(project_root)}
    if (project_root / '.git').exists():
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
        identity.update({'kind': 'git', 'commit': commit, 'dirty': dirty})
    else:
        identity.update({'kind': 'runtime_bundle', 'commit': None, 'dirty': False})
    return identity


def _run_arena(
    arena_entry,
    project_root,
    output_dir,
    anchor,
    anchor_iteration,
    current_iteration,
    games,
):
    output = output_dir / 'vs-iteration{}.json'.format(anchor_iteration)
    command = [
        sys.executable, str(arena_entry),
        '--anchor', str(anchor),
        '--anchor-iteration', str(anchor_iteration),
        '--current', str(output_dir / 'latest.pth.tar'),
        '--current-iteration', str(current_iteration),
        '--games', str(games),
        '--simulations', '96',
        '--batch-size', '128',
        '--seed', '20260715',
        '--device', 'cuda',
        '--output', str(output),
    ]
    subprocess.run(command, cwd=project_root, check=True)
    payload = json.loads(output.read_text())
    return output, payload


def main():
    import torch

    args = parse_args()
    if args.num_iterations < 1:
        raise ValueError('--num-iterations must be positive.')
    if args.snapshot_interval < 1:
        raise ValueError('--snapshot-interval must be positive.')
    if args.arena_games < 0 or args.arena_games % 2:
        raise ValueError('--arena-games must be nonnegative and even.')
    if not torch.cuda.is_available():
        raise RuntimeError('V4 P2 training requires CUDA.')
    gpu_name = torch.cuda.get_device_name(0)
    if not args.allow_non_p100 and 'P100' not in gpu_name.upper():
        raise RuntimeError(
            'The validated reference path requires a P100, found {}. '
            'Pass --allow-non-p100 only for an intentional hardware experiment.'.format(
                gpu_name
            )
        )

    runtime, runtime_root, inputs, bundled_oracle = _validate_runtime(
        args.runtime_manifest
    )
    project_root = (
        Path(args.source_root).resolve() if args.source_root else runtime_root
    )
    source_identity = _source_identity(project_root)
    if args.expected_source_commit:
        if source_identity.get('commit') != args.expected_source_commit:
            raise ValueError(
                'Source checkout is {}, expected {}.'.format(
                    source_identity.get('commit'), args.expected_source_commit
                )
            )
        if source_identity.get('dirty'):
            raise ValueError('Expected-commit source checkout is dirty.')
    entry_point = project_root / 'main_santorini.py'
    arena_entry = project_root / 'arena_santorini_v4_p2_arm.py'
    resume_checkpoint = Path(args.resume_checkpoint).resolve()
    resume_replay = Path(args.resume_replay).resolve()
    if not resume_checkpoint.is_file() or not resume_replay.is_file():
        raise FileNotFoundError('Resume checkpoint or replay does not exist.')
    metadata, windows = _load_resume_state(
        resume_checkpoint,
        resume_replay,
        runtime['inputs'][SEAM_SUITE_NAME]['sha256'],
        runtime['inputs'][VALUE_ANCHOR_NAME]['sha256'],
    )
    if not _close(
        metadata['v4_teacher_objective_reference'],
        runtime['lineage']['teacher_objective_reference'],
    ):
        raise ValueError('Resume checkpoint is outside the runtime P2 lineage.')

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError('Refusing to overwrite nonempty output: {}'.format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_binary = output_dir / 'santorini-oracle-linux-x86_64'
    shutil.copy2(bundled_oracle, oracle_binary)
    oracle_binary.chmod(0o755)

    start_iteration = int(metadata['iteration'])
    requested_end_iteration = start_iteration + args.num_iterations
    previous_objective = float(metadata['v4_teacher_objective_current'])
    start_checkpoint = resume_checkpoint
    completed = []
    snapshots = []
    status = 'completed'
    started = time.perf_counter()

    for iteration in range(start_iteration + 1, requested_end_iteration + 1):
        command = [
            sys.executable, str(entry_point),
            '--architecture', 'v4',
            '--training-mode', 'latest',
            '--num-iters', '1',
            '--num-eps', '240',
            '--num-mcts-sims', '96',
            '--search-mode', 'gumbel',
            '--gumbel-max-considered-actions', '16',
            '--gumbel-scale', '1.0',
            '--gumbel-placement-scale', '1.5',
            '--evaluation-gumbel-scale', '0.0',
            '--evaluation-gumbel-placement-scale', '1.5',
            '--placement-scale-exploration-probability', '0.10',
            '--placement-exploration-gumbel-scale', '2.25',
            '--playout-cap-randomization',
            '--playout-cap-full-probability', '0.25',
            '--playout-cap-fast-sims', '32',
            '--self-play-batch-size', '128',
            '--batch-size', '256',
            '--replay-reuse', '2',
            '--replay-reuse-warmup-iters', '0',
            '--validation-fraction', '0.05',
            '--optimizer', 'adamw',
            '--learning-rate', str(TRUNK_LR),
            '--weight-decay', '0.0001',
            '--lr-schedule', 'none',
            '--history-iters', '20',
            '--maxlen-of-queue', '200000',
            '--load-model',
            '--load-folder', str(resume_checkpoint.parent),
            '--load-file', resume_checkpoint.name,
            '--load-examples',
            '--examples-file', str(resume_replay),
            '--keep-loaded-examples',
            '--checkpoint', str(output_dir),
            '--checkpoint-examples-to-keep', '0',
            '--oracle-sparring-probability', '0.10',
            '--oracle-sparring-nodes', '5000',
            '--oracle-sparring-workers', '4',
            '--oracle-sparring-opening-seed', '20260921',
            '--oracle-sparring-ladder-version', '2',
            '--oracle-sparring-ratchet-games', '80',
            '--oracle-sparring-ratchet-score', '0.55',
            '--oracle-binary', str(oracle_binary),
            '--v4-seam-telemetry-suite', str(inputs[SEAM_SUITE_NAME]),
            '--v4-seam-telemetry-interval', '1',
            '--v4-seam-telemetry-batch-size', '256',
            '--v4-teacher-objective-step-threshold', '0.05',
            '--v4-teacher-objective-cumulative-threshold', '0.10',
            '--prior-target-kl-warning-threshold', '0.15',
            '--prior-target-kl-warning-min-positions', '256',
            '--prior-target-kl-warning-iterations', '3',
            '--milestone-interval', '20',
            '--telemetry-match-games', '40',
            '--telemetry-match-batch-size', '128',
            '--telemetry-placement-games', '40',
            '--telemetry-placement-temperature', '1.0',
            '--telemetry-opening-seed', '20260715',
            '--seed', '20260930',
            '--quiet',
        ]
        if iteration <= BRIDGE_END_ITERATION:
            command += [
                '--value-target-anchor-checkpoint', str(inputs[VALUE_ANCHOR_NAME]),
                '--value-target-beta-start', str(BRIDGE_START_BETA),
                '--value-target-beta-end', str(BRIDGE_END_BETA),
                '--value-target-beta-start-iteration', str(BRIDGE_START_ITERATION),
                '--value-target-beta-end-iteration', str(BRIDGE_END_ITERATION),
                '--value-target-anchor-batch-size', '512',
            ]
        print('V4 P2 iteration {} command: {}'.format(
            iteration, ' '.join(command)
        ), flush=True)
        result = subprocess.run(command, cwd=project_root, check=False)
        rows = _telemetry_rows(output_dir / 'telemetry' / 'telemetry.jsonl')
        if len(rows) != len(completed) + 1:
            if result.returncode != 0:
                raise RuntimeError(
                    'Trainer exited {} before iteration {} wrote telemetry. '
                    'The trainer traceback immediately above is the root '
                    'failure.'.format(result.returncode, iteration)
                )
            raise RuntimeError('Expected one telemetry row from iteration {}.'.format(iteration))
        previous_objective = _validate_row(
            rows[-1], runtime, iteration, previous_objective
        )
        completed.append(iteration)
        paused = bool(
            rows[-1].get('v4_teacher_objective_gate_triggered')
            or rows[-1].get('v4_teacher_objective_cumulative_triggered')
            or rows[-1].get('oracle_sparring_ratchet_triggered')
        )
        should_snapshot = (
            iteration % args.snapshot_interval == 0
            or iteration == requested_end_iteration
            or paused
        )
        if should_snapshot:
            snapshots.extend(
                str(path.relative_to(output_dir))
                for path in _snapshot(output_dir, iteration)
            )
        if result.returncode != 0:
            if not paused:
                raise RuntimeError(
                    'Trainer exited {} without a declared safety pause.'.format(
                        result.returncode
                    )
                )
            status = 'paused'
            break
        if paused:
            raise RuntimeError('Trainer returned success despite a safety pause.')
        resume_checkpoint = output_dir / 'latest-training.pth.tar'
        resume_replay = output_dir / 'latest.examples.npz'

    last_iteration = completed[-1]
    arenas = {}
    arena_paths = []
    if not args.no_end_arenas and args.arena_games:
        anchors = [(start_iteration, start_checkpoint)]
        if start_iteration != 1:
            anchors.append((1, inputs[ITERATION1_CHECKPOINT_NAME]))
        for anchor_iteration, anchor in anchors:
            path, payload = _run_arena(
                arena_entry,
                project_root,
                output_dir,
                anchor,
                anchor_iteration,
                last_iteration,
                args.arena_games,
            )
            arena_paths.append(path)
            arenas[str(anchor_iteration)] = {
                'standard_current_score': payload['standard']['current_score'],
                'placement_current_score': payload['placement_inclusive']['current_score'],
            }

    required_outputs = [
        output_dir / 'latest-training.pth.tar',
        output_dir / 'latest.pth.tar',
        output_dir / 'latest.examples.npz',
        *arena_paths,
    ]
    contract = {
        'schema_version': 1,
        'contract': RESULT_CONTRACT,
        'status': status,
        'gpu': gpu_name,
        'start_iteration': start_iteration,
        'requested_iterations': args.num_iterations,
        'requested_end_iteration': requested_end_iteration,
        'completed_iterations': completed,
        'last_iteration': last_iteration,
        'snapshot_interval': args.snapshot_interval,
        'snapshots': snapshots,
        'source': source_identity,
        'elapsed_seconds': time.perf_counter() - started,
        'arena_summary_by_anchor_iteration': arenas,
        'inputs': {
            'resume_checkpoint': {'sha256': _sha256(start_checkpoint)},
            'resume_replay': {'sha256': _sha256(Path(args.resume_replay))},
            'runtime_manifest': {'sha256': _sha256(args.runtime_manifest)},
        },
        'outputs': {
            path.name: {'bytes': path.stat().st_size, 'sha256': _sha256(path)}
            for path in required_outputs
        },
        'training_contract': {
            'replay_reuse': REPLAY_REUSE,
            'learning_rate': TRUNK_LR,
            'bridge_end_iteration': BRIDGE_END_ITERATION,
            'post_bridge_value_target_mode': 'outcome_z',
            'history_iterations': 20,
            'prior_target_kl_warning_threshold': 0.15,
            'prior_target_kl_warning_iterations': 3,
        },
        'final_test_touched': False,
        'final_arena_seeds_touched': False,
    }
    contract_path = output_dir / 'p2-training-contract.json'
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + '\n')
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
