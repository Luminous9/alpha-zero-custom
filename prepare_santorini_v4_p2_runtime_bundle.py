"""Build the reusable code-and-fixed-input Kaggle runtime for V4 P2 training."""

import argparse
import json
import os
from pathlib import Path
import zipfile

import torch

from santorini.OracleResearch import file_sha256


ITERATION1_CHECKPOINT_NAME = 'p2-iteration1-training.pth.tar'
VALUE_ANCHOR_NAME = 'p1c-value-anchor.pth.tar'
SEAM_SUITE_NAME = 'v4-seam-telemetry-suite.npz'
DEEP_VALUE_SUITE_NAME = 'v4-deep-value-telemetry-suite.npz'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
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
        '--deep-value-suite',
        default='temp/santorini_v4_p2_deep_value_telemetry.npz',
    )
    parser.add_argument(
        '--linux-oracle-binary',
        default=(
            'temp/santorini_v4_p2_linux_build/target/release/'
            'santorini-oracle'
        ),
    )
    parser.add_argument('--santorini-ai-license', default='../santorini-ai/LICENSE')
    parser.add_argument(
        '--output', default='temp/santorini_v4_p2_runtime_bundle.zip'
    )
    return parser.parse_args()


def build_bundle(args):
    root = Path(__file__).resolve().parent
    iteration1 = Path(args.iteration1_checkpoint).resolve()
    value_anchor = Path(args.value_anchor).resolve()
    seam_suite = Path(args.seam_suite).resolve()
    deep_value_suite = Path(args.deep_value_suite).resolve()
    linux_oracle = Path(args.linux_oracle_binary).resolve()
    license_path = Path(args.santorini_ai_license).resolve()
    for path in (
        iteration1, value_anchor, seam_suite, deep_value_suite,
        linux_oracle, license_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    payload = torch.load(iteration1, map_location='cpu', weights_only=False)
    metadata = dict(payload.get('training_metadata', {}))
    if int(metadata.get('iteration', -1)) != 1:
        raise ValueError('Longitudinal anchor is not P2 iteration 1.')
    if metadata.get('training_mode') != 'latest':
        raise ValueError('Iteration-1 anchor is not latest-mode state.')
    seam_hash = file_sha256(seam_suite)
    if metadata.get('v4_seam_suite_fingerprint') != seam_hash:
        raise ValueError('Iteration-1 anchor does not match the seam suite.')
    if metadata.get('v4_teacher_objective_current') is None:
        raise ValueError('Iteration-1 anchor lacks the teacher reference.')

    sources = sorted(root.glob('*.py')) + sorted((root / 'santorini').rglob('*.py'))
    source_records = {
        path.relative_to(root).as_posix(): {
            'bytes': path.stat().st_size,
            'sha256': file_sha256(path),
        }
        for path in sources
    }
    input_paths = {
        ITERATION1_CHECKPOINT_NAME: iteration1,
        VALUE_ANCHOR_NAME: value_anchor,
        SEAM_SUITE_NAME: seam_suite,
        DEEP_VALUE_SUITE_NAME: deep_value_suite,
    }
    manifest = {
        'schema_version': 1,
        'contract': 'santorini_v4_p2_runtime',
        'lineage': {
            'iteration1_checkpoint_sha256': file_sha256(iteration1),
            'teacher_objective_reference': float(
                metadata['v4_teacher_objective_current']
            ),
            'seam_suite_sha256': seam_hash,
            'deep_value_suite_sha256': file_sha256(deep_value_suite),
        },
        'inputs': {
            name: {
                'bytes': path.stat().st_size,
                'sha256': file_sha256(path),
            }
            for name, path in input_paths.items()
        },
        'sources': source_records,
        'protocol': {
            'architecture': 'v4-ordinary-6x192-canonical-d4',
            'games_per_iteration': 240,
            'replay_reuse': 2.0,
            'learning_rate': 1e-4,
            'history_iterations': 20,
            'full_simulations': 96,
            'fast_simulations': 32,
            'bridge_end_iteration': 11,
            'post_bridge_value_target_mode': 'outcome_z',
            'reference_gpu': 'P100',
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
            'v4-p2-runtime-manifest.json',
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
