import argparse
import json

from santorini.ReplayBuffer import (
    collapse_compact_replay_symmetries,
    trim_compact_replay,
)


def main():
    parser = argparse.ArgumentParser(
        description="Retain only the newest iteration windows in a compact Santorini replay."
    )
    parser.add_argument("replay", help="Path to latest.examples.npz")
    parser.add_argument("--keep-last-windows", type=int, required=True)
    parser.add_argument("--output", help="Optional output archive; defaults to atomic in-place replacement.")
    parser.add_argument(
        "--collapse-symmetry-group-size",
        type=int,
        help="Also retain one example from each consecutive legacy symmetry group.",
    )
    args = parser.parse_args()

    result = {"trim": trim_compact_replay(
        args.replay,
        keep_last_windows=args.keep_last_windows,
        output_path=args.output,
    )}
    destination = args.output or args.replay
    if args.collapse_symmetry_group_size is not None:
        result["collapse"] = collapse_compact_replay_symmetries(
            destination,
            group_size=args.collapse_symmetry_group_size,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
