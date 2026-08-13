"""Continue the selected P2 Arm D lineage from iteration 4 through iteration 11."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


INPUT_ROOT = Path('/kaggle/input')
WORKING_ROOT = Path('/kaggle/working/santorini_v4_p2_d_continuation')
MANIFEST_NAME = 'p2-d-continuation-manifest.json'
RESUME_CHECKPOINT_NAME = 'p2-iteration4-d-training.pth.tar'
RESUME_REPLAY_NAME = 'p2-iteration4-d.examples.npz'
ITERATION1_CHECKPOINT_NAME = 'p2-iteration1-training.pth.tar'
VALUE_ANCHOR_NAME = 'p1c-value-anchor.pth.tar'
SEAM_SUITE_NAME = 'v4-seam-telemetry-suite.npz'


def _exactly_one(filename):
    matches = sorted(INPUT_ROOT.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(
            'Expected exactly one {} under {}, found: {}'.format(
                filename, INPUT_ROOT, [str(path) for path in matches]
            )
        )
    return matches[0]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left, right, tolerance=1e-8):
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _telemetry_rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line]
        if path.is_file() else []
    )


def _expected_beta(config, iteration):
    beta = config['value_beta']
    progress = min(1.0, max(0.0, (
        iteration - beta['start_iteration']
    ) / (beta['end_iteration'] - beta['start_iteration'])))
    return beta['start'] + progress * (beta['end'] - beta['start'])


def _validate_row(row, manifest, expected_iteration, previous_objective):
    protocol = manifest['protocol']
    config = manifest['configuration']
    lineage = manifest['lineage']
    if int(row.get('iteration', -1)) != expected_iteration:
        raise RuntimeError('Trainer wrote the wrong iteration.')
    if int(row.get('games', -1)) != protocol['games_per_iteration']:
        raise RuntimeError('Iteration did not complete the frozen game count.')
    if int(row.get('oracle_sparring_games', -1)) != 24:
        raise RuntimeError('Iteration did not complete 24 sparring games.')
    if not _close(row.get('target_replay_reuse', -1), protocol['replay_reuse']):
        raise RuntimeError('Iteration changed fixed replay reuse.')
    if int(row.get('replay_reuse_warmup_iters', -1)) != 0:
        raise RuntimeError('Continuation unexpectedly enabled reuse warm-up.')
    if row.get('v4_seam_suite_fingerprint') != manifest['inputs'][SEAM_SUITE_NAME]['sha256']:
        raise RuntimeError('Iteration used the wrong frozen seam suite.')
    if not _close(row.get('v4_teacher_objective_previous'), previous_objective):
        raise RuntimeError('Iteration broke the teacher-objective chain.')
    if not _close(
        row.get('v4_teacher_objective_reference'),
        lineage['teacher_objective_reference'],
    ):
        raise RuntimeError('Iteration changed the frozen teacher reference.')
    if not _close(row.get('v4_teacher_objective_step_threshold'), 0.05):
        raise RuntimeError('Iteration changed the teacher step gate.')
    if not _close(row.get('v4_teacher_objective_cumulative_threshold'), 0.10):
        raise RuntimeError('Iteration changed the cumulative teacher review.')
    for key in ('trunk_learning_rate', 'policy_head_learning_rate', 'value_head_learning_rate'):
        if not _close(row.get(key), config[key]):
            raise RuntimeError('{} does not match Arm D.'.format(key))
    if row.get('value_target_mode') != config['value_target_mode']:
        raise RuntimeError('Iteration used the wrong value-target mode.')
    if row.get('value_target_anchor_checkpoint_sha256') != manifest['inputs'][VALUE_ANCHOR_NAME]['sha256']:
        raise RuntimeError('Iteration used the wrong P1c value anchor.')
    if not _close(row.get('value_target_beta'), _expected_beta(config, expected_iteration)):
        raise RuntimeError('Iteration used the wrong absolute bridge beta.')
    current = float(row['v4_teacher_objective_current'])
    if not _close(
        row.get('v4_teacher_objective_cumulative_delta'),
        current - float(lineage['teacher_objective_reference']),
    ):
        raise RuntimeError('Iteration reported an inconsistent cumulative objective delta.')
    return current


def _snapshot(output_dir, iteration):
    names = (
        ('latest-training.pth.tar', 'checkpoint_{}-training.pth.tar'.format(iteration)),
        ('latest.pth.tar', 'checkpoint_{}.pth.tar'.format(iteration)),
        ('latest.examples.npz', 'checkpoint_{}.examples.npz'.format(iteration)),
    )
    for source_name, destination_name in names:
        source = output_dir / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_dir / destination_name)


def _run_arena(
    arena_entry,
    project_root,
    output_dir,
    anchor,
    anchor_iteration,
    current_iteration,
):
    output = output_dir / 'vs-iteration{}.json'.format(anchor_iteration)
    command = [
        sys.executable, str(arena_entry),
        '--anchor', str(anchor),
        '--anchor-iteration', str(anchor_iteration),
        '--current', str(output_dir / 'latest.pth.tar'),
        '--current-iteration', str(current_iteration),
        '--games', '40',
        '--simulations', '96',
        '--batch-size', '128',
        '--seed', '20260715',
        '--device', 'cuda',
        '--output', str(output),
    ]
    subprocess.run(command, cwd=project_root, check=True)
    payload = json.loads(output.read_text())
    if int(payload.get('anchor_iteration', -1)) != anchor_iteration:
        raise RuntimeError('Milestone arena recorded the wrong anchor iteration.')
    if int(payload.get('current_iteration', -1)) != current_iteration:
        raise RuntimeError('Milestone arena recorded the wrong current iteration.')
    return output, payload


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('The P2 continuation requires CUDA.')
    gpu_name = torch.cuda.get_device_name(0)
    if 'P100' not in gpu_name.upper():
        raise RuntimeError('The reference continuation requires a P100, found {}.'.format(gpu_name))

    manifest_path = _exactly_one(MANIFEST_NAME)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('contract') != 'santorini_v4_p2_d_continuation':
        raise RuntimeError('Wrong continuation manifest contract.')
    entry_point = _exactly_one('main_santorini.py')
    project_root = entry_point.parent
    arena_entry = project_root / 'arena_santorini_v4_p2_arm.py'
    if not arena_entry.is_file():
        raise FileNotFoundError(arena_entry)

    input_names = (
        RESUME_CHECKPOINT_NAME,
        RESUME_REPLAY_NAME,
        ITERATION1_CHECKPOINT_NAME,
        VALUE_ANCHOR_NAME,
        SEAM_SUITE_NAME,
    )
    inputs = {name: _exactly_one(name) for name in input_names}
    for name, path in inputs.items():
        if _sha256(path) != manifest['inputs'][name]['sha256']:
            raise RuntimeError('Input digest changed: {}'.format(path))
    bundled_oracle = manifest_path.parent / 'oracle-build' / 'santorini-oracle-linux-x86_64'
    if _sha256(bundled_oracle) != manifest['oracle_build']['linux_binary_sha256']:
        raise RuntimeError('Bundled oracle digest changed.')

    output_dir = WORKING_ROOT
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError('Refusing to overwrite nonempty continuation directory: {}'.format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_binary = output_dir / 'santorini-oracle-linux-x86_64'
    shutil.copy2(bundled_oracle, oracle_binary)
    oracle_binary.chmod(0o755)

    config = manifest['configuration']
    start_iteration = int(manifest['lineage']['start_iteration'])
    end_iteration = int(manifest['lineage']['end_iteration'])
    resume_checkpoint = inputs[RESUME_CHECKPOINT_NAME]
    resume_replay = inputs[RESUME_REPLAY_NAME]
    previous_objective = float(manifest['lineage']['resume_teacher_objective'])
    completed = []
    status = 'completed'
    started = time.perf_counter()

    for iteration in range(start_iteration + 1, end_iteration + 1):
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
            '--learning-rate', str(config['trunk_learning_rate']),
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
            '--milestone-interval', '20',
            '--telemetry-match-games', '40',
            '--telemetry-match-batch-size', '128',
            '--telemetry-placement-games', '40',
            '--telemetry-placement-temperature', '1.0',
            '--telemetry-opening-seed', '20260715',
            '--value-target-anchor-checkpoint', str(inputs[VALUE_ANCHOR_NAME]),
            '--value-target-beta-start', str(config['value_beta']['start']),
            '--value-target-beta-end', str(config['value_beta']['end']),
            '--value-target-beta-start-iteration', str(config['value_beta']['start_iteration']),
            '--value-target-beta-end-iteration', str(config['value_beta']['end_iteration']),
            '--value-target-anchor-batch-size', '512',
            '--seed', '20260930',
            '--quiet',
        ]
        print('Arm D continuation iteration {} command: {}'.format(
            iteration, ' '.join(command)
        ), flush=True)
        result = subprocess.run(command, cwd=project_root, check=False)
        rows = _telemetry_rows(output_dir / 'telemetry' / 'telemetry.jsonl')
        if len(rows) != len(completed) + 1:
            raise RuntimeError('Expected one new telemetry row from iteration {}.'.format(iteration))
        previous_objective = _validate_row(
            rows[-1], manifest, iteration, previous_objective
        )
        _snapshot(output_dir, iteration)
        completed.append(iteration)
        paused = bool(
            rows[-1].get('v4_teacher_objective_gate_triggered')
            or rows[-1].get('v4_teacher_objective_cumulative_triggered')
            or rows[-1].get('oracle_sparring_ratchet_triggered')
        )
        if result.returncode != 0:
            if not paused:
                raise RuntimeError(
                    'Trainer exited {} without a declared safety pause.'.format(result.returncode)
                )
            status = 'paused'
            break
        if paused:
            raise RuntimeError('Trainer returned success despite a safety pause.')
        resume_checkpoint = output_dir / 'latest-training.pth.tar'
        resume_replay = output_dir / 'latest.examples.npz'

    if not completed:
        raise RuntimeError('Continuation produced no completed iteration.')
    last_iteration = completed[-1]
    arenas = {}
    arena_paths = []
    for anchor_iteration, anchor in (
        (4, inputs[RESUME_CHECKPOINT_NAME]),
        (1, inputs[ITERATION1_CHECKPOINT_NAME]),
    ):
        path, payload = _run_arena(
            arena_entry,
            project_root,
            output_dir,
            anchor,
            anchor_iteration,
            last_iteration,
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
        'contract': 'santorini_v4_p2_d_continuation_result',
        'configuration': config,
        'status': status,
        'gpu': gpu_name,
        'start_iteration': start_iteration,
        'requested_end_iteration': end_iteration,
        'completed_iterations': completed,
        'last_iteration': last_iteration,
        'elapsed_seconds': time.perf_counter() - started,
        'arena_summary_by_anchor_iteration': arenas,
        'inputs': {
            name: {'sha256': _sha256(path)} for name, path in inputs.items()
        },
        'outputs': {
            path.name: {'bytes': path.stat().st_size, 'sha256': _sha256(path)}
            for path in required_outputs
        },
        'final_test_touched': False,
        'final_arena_seeds_touched': False,
    }
    contract_path = output_dir / 'p2-d-continuation-contract.json'
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + '\n')
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
