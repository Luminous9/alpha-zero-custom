"""Build the shared extracted Kaggle bundle for P2 fine-tuning arms A-D."""

import argparse
import json
import os
from pathlib import Path
import zipfile

import torch

from santorini.OracleResearch import file_sha256
from santorini.ReplayBuffer import load_compact_replay


CHECKPOINT_NAME = 'p2-iteration1-training.pth.tar'
REPLAY_NAME = 'p2-iteration1.examples.npz'
VALUE_ANCHOR_NAME = 'p1c-value-anchor.pth.tar'
SEAM_SUITE_NAME = 'v4-seam-telemetry-suite.npz'

ARMS = {
    'A': {
        'description': 'High-LR pure-z control',
        'trunk_learning_rate': 3e-4,
        'policy_head_learning_rate': 3e-4,
        'value_head_learning_rate': 3e-4,
        'value_target_mode': 'outcome_z',
        'value_beta': None,
    },
    'B': {
        'description': 'Global fine-tuning LR',
        'trunk_learning_rate': 1e-4,
        'policy_head_learning_rate': 1e-4,
        'value_head_learning_rate': 1e-4,
        'value_target_mode': 'outcome_z',
        'value_beta': None,
    },
    'C': {
        'description': 'Conservative trunk/value with faster policy head',
        'trunk_learning_rate': 1e-4,
        'policy_head_learning_rate': 3e-4,
        'value_head_learning_rate': 3e-5,
        'value_target_mode': 'outcome_z',
        'value_beta': None,
    },
    'D': {
        'description': 'Global fine-tuning LR plus P1c-to-z value bridge',
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
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint',
        default=(
            'temp/santorini_v4_p2_smoke_results/transition/'
            'latest-training.pth.zip'
        ),
    )
    parser.add_argument(
        '--replay',
        default=(
            'temp/santorini_v4_p2_smoke_results/transition/'
            'latest.examples.zip'
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
    parser.add_argument('--end-iteration', type=int, default=4)
    parser.add_argument(
        '--output', default='temp/santorini_v4_p2_diagnostic_bundle.zip'
    )
    return parser.parse_args()


def _checkpoint_metadata(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    metadata = dict(payload.get('training_metadata', {}))
    if int(metadata.get('iteration', -1)) != 1:
        raise ValueError('Diagnostic must start from accepted P2 iteration 1.')
    if metadata.get('training_mode') != 'latest':
        raise ValueError('Diagnostic checkpoint is not resumable latest-mode state.')
    if 'optimizer_state_dict' not in payload:
        raise ValueError('Diagnostic checkpoint lacks optimizer state.')
    return metadata


def build_bundle(args):
    root = Path(__file__).resolve().parent
    checkpoint = Path(args.checkpoint).resolve()
    replay = Path(args.replay).resolve()
    value_anchor = Path(args.value_anchor).resolve()
    seam_suite = Path(args.seam_suite).resolve()
    linux_oracle = Path(args.linux_oracle_binary).resolve()
    license_path = Path(args.santorini_ai_license).resolve()
    required = (checkpoint, replay, value_anchor, seam_suite, linux_oracle, license_path)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = _checkpoint_metadata(checkpoint)
    if args.end_iteration <= 1:
        raise ValueError('--end-iteration must be after iteration 1.')
    if args.end_iteration > 11:
        raise ValueError('The diagnostic bundle may not run beyond the declared bridge.')
    windows = load_compact_replay(replay)
    if len(windows) != 1 or not windows[0]:
        raise ValueError('Accepted iteration-1 replay must contain exactly one window.')
    seam_hash = file_sha256(seam_suite)
    if metadata.get('v4_seam_suite_fingerprint') != seam_hash:
        raise ValueError('Accepted checkpoint does not match the frozen seam suite.')
    if metadata.get('v4_teacher_objective_current') is None:
        raise ValueError('Accepted checkpoint lacks the teacher-objective baseline.')
    if int(metadata.get('oracle_sparring_nodes', -1)) != 5000:
        raise ValueError('Accepted checkpoint does not use ladder-v2 5k sparring.')
    if int(metadata.get('oracle_sparring_ladder_version', -1)) != 2:
        raise ValueError('Accepted checkpoint does not use ladder version 2.')

    sources = sorted(root.glob('*.py')) + sorted((root / 'santorini').rglob('*.py'))
    input_paths = {
        CHECKPOINT_NAME: checkpoint,
        REPLAY_NAME: replay,
        VALUE_ANCHOR_NAME: value_anchor,
        SEAM_SUITE_NAME: seam_suite,
    }
    manifest = {
        'schema_version': 1,
        'contract': 'santorini_v4_p2_finetuning_diagnostic',
        'lineage': {
            'start_iteration': 1,
            'end_iteration': int(args.end_iteration),
            'new_iterations': int(args.end_iteration) - 1,
            'resume_teacher_objective': float(
                metadata['v4_teacher_objective_current']
            ),
            'resume_ratchet_pair_scores': [
                float(value)
                for value in metadata.get('oracle_sparring_pair_score_history', [])
            ],
            'resume_replay_windows': len(windows),
            'resume_replay_examples': sum(len(window) for window in windows),
        },
        'inputs': {
            name: {
                'bytes': path.stat().st_size,
                'sha256': file_sha256(path),
            }
            for name, path in input_paths.items()
        },
        'arms': ARMS,
        'protocol': {
            'games_per_iteration': 240,
            'new_iterations_per_arm': int(args.end_iteration) - 1,
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
            'arena_games_per_gate': 40,
            'arena_opponent_iteration': 1,
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
        'recommended_launch_order': [['A', 'B'], ['C', 'D']],
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
            'p2-diagnostic-manifest.json',
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
