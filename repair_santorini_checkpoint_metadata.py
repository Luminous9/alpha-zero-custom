import argparse
import os
import tempfile

import torch


def main():
    parser = argparse.ArgumentParser(
        description='Correct iteration metadata in a resumable Santorini latest-training checkpoint.'
    )
    parser.add_argument('--input', required=True, help='Source latest-training.pth.tar checkpoint.')
    parser.add_argument('--output', required=True, help='Destination checkpoint; may equal --input.')
    parser.add_argument('--iteration', required=True, type=int, help='Last completed global iteration.')
    args = parser.parse_args()

    if args.iteration < 0:
        parser.error('--iteration cannot be negative')
    if not os.path.isfile(args.input):
        parser.error('input checkpoint does not exist: {}'.format(args.input))

    try:
        checkpoint = torch.load(args.input, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.input, map_location='cpu')

    if checkpoint.get('architecture') != 'v3':
        raise ValueError('Expected a V3 checkpoint, found {!r}.'.format(checkpoint.get('architecture')))
    if 'optimizer_state_dict' not in checkpoint:
        raise ValueError('Checkpoint has no optimizer state and is not a complete training checkpoint.')

    metadata = dict(checkpoint.get('training_metadata') or {})
    old_iteration = metadata.get('iteration')
    metadata['iteration'] = args.iteration
    metadata.setdefault('training_mode', 'latest')
    checkpoint['training_metadata'] = metadata

    output = os.path.abspath(args.output)
    output_dir = os.path.dirname(output)
    os.makedirs(output_dir, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.checkpoint-metadata-', suffix='.pth.tar', dir=output_dir)
    os.close(descriptor)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)

    print('Corrected checkpoint iteration {} -> {}: {}'.format(old_iteration, args.iteration, output))


if __name__ == '__main__':
    main()
