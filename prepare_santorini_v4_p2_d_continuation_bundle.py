"""Build the extracted Kaggle bundle for the selected Arm D iterations 5-11."""

import argparse
import json
import math
import os
from pathlib import Path
import zipfile

import torch

from santorini.OracleResearch import file_sha256
from santorini.ReplayBuffer import load_compact_replay


RESUME_CHECKPOINT_NAME = 'p2-iteration4-d-training.pth.tar'
RESUME_REPLAY_NAME = 'p2-iteration4-d.examples.npz'
ITERATION1_CHECKPOINT_NAME = 'p2-iteration1-training.pth.tar'
VALUE_ANCHOR_NAME = 'p1c-value-anchor.pth.tar'
SEAM_SUITE_NAME = 'v4-seam-telemetry-suite.npz'

CONFIGURATION = {
    'description': 'Selected Arm D: global fine-tuning LR plus P1c-to-z bridge',
    'trunk_learning_rate': 1e-4,
    'policy_head_learning_rate': 1e-4,
    'value_head_learning_rate': 1e-4,
    'value_target_mode': 'p1c_anchor_to_outcome_z',
    'value_beta': {
        'start': 0.25,
        'end': 1.0,
        'start_iteration': 2,
        'end_iteration': 11,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--resume-checkpoint',
        default=(
            'temp/santorini_v4_p2_diagnostic_arms/arm_D/'
            'latest-training.pth.zip'
        ),
    )
    parser.add_argument(
        '--resume-replay',
        default=(
            'temp/santorini_v4_p2_diagnostic_arms/arm_D/'
            'latest.examples.zip'
        ),
    )
    parser.add_argument(
        '--diagnostic-contract',
        default=(
            'temp/santorini_v4_p2_diagnostic_arms/arm_D/'
            'p2-diagnostic-contract.json'
        ),
    )
    parser.add_argument(
        '--iteration1-checkpoint',
        default=(
            'temp/santorini_v4_p2_smoke_results/transition/'
            'latest-training.pth.zip'
        ),
    )
    parser.add_argument(
        '--value-anchor',
        default='temp/santorini_v4_p2_handoff/p2-start.pth.tar',
    )
    parser.add_argument(
        '--seam-suite',
        default='temp/santorini_v4_p2_preparation/v4-seam-telemetry-suite.npz',
    )
    parser.add_argument(
        '--linux-oracle-binary',
        default=(
            'temp/santorini_v4_p2_linux_build/target/release/'
            'santorini-oracle'
        ),
    )
    parser.add_argument('--santorini-ai-license', default='../santorini-ai/LICENSE')
    parser.add_argument('--end-iteration', type=int, default=11)
    parser.add_argument(
        '--output', default='temp/santorini_v4_p2_d_continuation_5_11_bundle.zip'
    )
    return parser.parse_args()


def _close(left, right, tolerance=1e-8):
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _load_checkpoint(path, expected_iteration):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    metadata = dict(payload.get('training_metadata', {}))
    if int(metadata.get('iteration', -1)) != expected_iteration:
        raise ValueError(
            'Expected iteration {}, found {} in {}.'.format(
                expected_iteration, metadata.get('iteration'), path
            )
        )
    if metadata.get('training_mode') != 'latest':
        raise ValueError('Checkpoint is not resumable latest-mode state: {}'.format(path))
    if 'optimizer_state_dict' not in payload:
        raise ValueError('Checkpoint lacks optimizer state: {}'.format(path))
    return metadata


def _validate_diagnostic_contract(contract, resume_checkpoint, resume_replay):
    if contract.get('contract') != 'santorini_v4_p2_finetuning_diagnostic_result':
        raise ValueError('Wrong source diagnostic contract.')
    if contract.get('arm') != 'D' or contract.get('status') != 'completed':
        raise ValueError('Source contract is not the completed Arm D result.')
    if contract.get('completed_iterations') != [2, 3, 4]:
        raise ValueError('Source Arm D did not complete exactly iterations 2-4.')
    if int(contract.get('last_iteration', -1)) != 4:
        raise ValueError('Source Arm D contract does not end at iteration 4.')
    if contract.get('final_test_touched') or contract.get('final_arena_seeds_touched'):
        raise ValueError('Source diagnostic touched held-out final data.')
    expected_outputs = {
        'latest-training.pth.tar': resume_checkpoint,
        'latest.examples.npz': resume_replay,
    }
    for name, path in expected_outputs.items():
        expected = contract.get('outputs', {}).get(name, {}).get('sha256')
        if expected != file_sha256(path):
            raise ValueError('Source Arm D artifact does not match its contract: {}'.format(path))


def build_bundle(args):
    root = Path(__file__).resolve().parent
    resume_checkpoint = Path(args.resume_checkpoint).resolve()
    resume_replay = Path(args.resume_replay).resolve()
    diagnostic_contract_path = Path(args.diagnostic_contract).resolve()
    iteration1_checkpoint = Path(args.iteration1_checkpoint).resolve()
    value_anchor = Path(args.value_anchor).resolve()
    seam_suite = Path(args.seam_suite).resolve()
    linux_oracle = Path(args.linux_oracle_binary).resolve()
    license_path = Path(args.santorini_ai_license).resolve()
    required = (
        resume_checkpoint,
        resume_replay,
        diagnostic_contract_path,
        iteration1_checkpoint,
        value_anchor,
        seam_suite,
        linux_oracle,
        license_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.end_iteration != 11:
        raise ValueError('This frozen continuation must stop at iteration 11.')

    diagnostic_contract = json.loads(diagnostic_contract_path.read_text())
    _validate_diagnostic_contract(
        diagnostic_contract, resume_checkpoint, resume_replay
    )
    metadata = _load_checkpoint(resume_checkpoint, expected_iteration=4)
    iteration1_metadata = _load_checkpoint(
        iteration1_checkpoint, expected_iteration=1
    )
    windows = load_compact_replay(resume_replay)
    if len(windows) != 4 or any(not window for window in windows):
        raise ValueError('Arm D iteration-4 replay must contain four nonempty windows.')

    seam_hash = file_sha256(seam_suite)
    value_anchor_hash = file_sha256(value_anchor)
    if metadata.get('v4_seam_suite_fingerprint') != seam_hash:
        raise ValueError('Arm D checkpoint does not match the frozen seam suite.')
    if iteration1_metadata.get('v4_seam_suite_fingerprint') != seam_hash:
        raise ValueError('Iteration-1 anchor does not match the frozen seam suite.')
    if metadata.get('value_target_mode') != CONFIGURATION['value_target_mode']:
        raise ValueError('Arm D checkpoint has the wrong value-target mode.')
    if metadata.get('value_target_anchor_checkpoint_sha256') != value_anchor_hash:
        raise ValueError('Arm D checkpoint has the wrong value anchor.')
    if not _close(metadata.get('value_target_beta'), 5.0 / 12.0):
        raise ValueError('Arm D checkpoint is not at the declared iteration-4 beta.')
    learning_rates = metadata.get('optimizer_group_learning_rates', {})
    for group in ('trunk', 'policy_head', 'value_head'):
        if not _close(learning_rates.get(group), 1e-4):
            raise ValueError('Arm D checkpoint has the wrong {} LR.'.format(group))
    if not _close(metadata.get('replay_reuse'), 2.0):
        raise ValueError('Arm D checkpoint did not use fixed 2x replay reuse.')
    if int(metadata.get('replay_reuse_warmup_iters', -1)) != 0:
        raise ValueError('Arm D checkpoint unexpectedly used a reuse warm-up.')
    if int(metadata.get('oracle_sparring_nodes', -1)) != 5000:
        raise ValueError('Arm D checkpoint does not use ladder-v2 5k sparring.')
    if int(metadata.get('oracle_sparring_ladder_version', -1)) != 2:
        raise ValueError('Arm D checkpoint does not use ladder version 2.')
    if metadata.get('v4_teacher_objective_current') is None:
        raise ValueError('Arm D checkpoint lacks its teacher objective.')
    if metadata.get('v4_teacher_objective_reference') is None:
        raise ValueError('Arm D checkpoint lacks the iteration-1 teacher reference.')
    if not _close(
        metadata['v4_teacher_objective_reference'],
        iteration1_metadata['v4_teacher_objective_current'],
    ):
        raise ValueError('Arm D lost the original iteration-1 teacher reference.')

    sources = sorted(root.glob('*.py')) + sorted((root / 'santorini').rglob('*.py'))
    input_paths = {
        RESUME_CHECKPOINT_NAME: resume_checkpoint,
        RESUME_REPLAY_NAME: resume_replay,
        ITERATION1_CHECKPOINT_NAME: iteration1_checkpoint,
        VALUE_ANCHOR_NAME: value_anchor,
        SEAM_SUITE_NAME: seam_suite,
    }
    manifest = {
        'schema_version': 1,
        'contract': 'santorini_v4_p2_d_continuation',
        'configuration': CONFIGURATION,
        'lineage': {
            'source_arm': 'D',
            'start_iteration': 4,
            'end_iteration': 11,
            'new_iterations': 7,
            'resume_teacher_objective': float(
                metadata['v4_teacher_objective_current']
            ),
            'teacher_objective_reference': float(
                metadata['v4_teacher_objective_reference']
            ),
            'resume_ratchet_pair_scores': [
                float(value)
                for value in metadata.get('oracle_sparring_pair_score_history', [])
            ],
            'resume_replay_windows': len(windows),
            'resume_replay_examples': sum(len(window) for window in windows),
            'source_diagnostic_contract_sha256': file_sha256(
                diagnostic_contract_path
            ),
        },
        'inputs': {
            name: {
                'bytes': path.stat().st_size,
                'sha256': file_sha256(path),
            }
            for name, path in input_paths.items()
        },
        'protocol': {
            'games_per_iteration': 240,
            'new_iterations': 7,
            'replay_reuse': 2.0,
            'replay_reuse_warmup_iters': 0,
            'optimizer': 'adamw',
            'weight_decay': 1e-4,
            'lr_schedule': [],
            'full_simulations': 96,
            'fast_simulations': 32,
            'full_search_probability': 0.25,
            'self_play_gumbel_scale': 1.0,
            'evaluation_gumbel_scale': 0.0,
            'oracle_sparring_probability': 0.10,
            'oracle_nodes': 5000,
            'oracle_ladder_version': 2,
            'teacher_step_threshold': 0.05,
            'teacher_cumulative_threshold': 0.10,
            'arena_games_per_gate_per_anchor': 40,
            'arena_anchor_iterations': [4, 1],
            'per_iteration_snapshots': True,
            'console_log_mode': 'compact_no_progress_bars',
        },
        'oracle_build': {
            'oracle_version': '0.2.0',
            'platform': 'linux-x86_64',
            'linux_binary_bytes': linux_oracle.stat().st_size,
            'linux_binary_sha256': file_sha256(linux_oracle),
            'runtime_cargo_required': False,
        },
        'python_source_files': len(sources),
        'final_test_touched': False,
        'final_arena_seeds_touched': False,
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with zipfile.ZipFile(temporary, 'w', allowZip64=True) as archive:
        for path in sources:
            archive.write(
                path,
                path.relative_to(root).as_posix(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
        for name, path in input_paths.items():
            archive.write(
                path,
                'inputs/{}'.format(name),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
        archive.write(
            linux_oracle,
            'oracle-build/santorini-oracle-linux-x86_64',
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        )
        archive.write(license_path, 'oracle-build/SANTORINI_AI_LICENSE')
        archive.writestr(
            'p2-d-continuation-manifest.json',
            json.dumps(manifest, indent=2, sort_keys=True),
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        )
    os.replace(temporary, output)
    report = {
        **manifest,
        'output': str(output),
        'output_bytes': output.stat().st_size,
        'output_sha256': file_sha256(output),
    }
    output.with_suffix(output.suffix + '.report.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n'
    )
    return report


def main():
    print(json.dumps(build_bundle(parse_args()), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
