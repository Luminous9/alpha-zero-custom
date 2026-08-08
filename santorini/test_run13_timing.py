from types import SimpleNamespace
import unittest

from benchmark_santorini_run13_timing import run13_command, timing_summary
from summarize_santorini_run13_timing import combine_profiles


class TestRun13TimingBenchmark(unittest.TestCase):
    @staticmethod
    def args(**changes):
        values = {
            "source": "temp/source",
            "output": "temp/output",
            "profile": "ordinary",
            "smoke": False,
            "seed": 7,
        }
        values.update(changes)
        return SimpleNamespace(**values)

    def test_representative_command_pins_run13_workload(self):
        command = run13_command(self.args())
        self.assertEqual(command[command.index("--num-eps") + 1], "240")
        self.assertEqual(command[command.index("--num-mcts-sims") + 1], "96")
        self.assertEqual(command[command.index("--playout-cap-fast-sims") + 1], "32")
        self.assertIn("--load-examples", command)
        self.assertIn("--keep-loaded-examples", command)

    def test_milestone_and_smoke_profiles_are_explicit(self):
        milestone = run13_command(self.args(profile="milestone"))
        self.assertEqual(milestone[milestone.index("--milestone-interval") + 1], "1")
        smoke = run13_command(self.args(smoke=True))
        self.assertIn("--no-telemetry-matches", smoke)
        self.assertEqual(smoke[smoke.index("--num-eps") + 1], "2")

    def test_summary_extracts_every_phase(self):
        row = {
            "iteration": 301,
            "games": 240,
            "num_mcts_sims": 96,
            "training_steps": 87,
            "wall_total_seconds": 10.0,
        }
        for index, phase in enumerate(
            ("self_play", "training", "arena_telemetry", "serialization", "other"),
            start=1,
        ):
            row["wall_{}_seconds".format(phase)] = float(index)
            row["wall_{}_fraction".format(phase)] = index / 15.0
        summary = timing_summary(self.args(), row, ["python"], 11.0)
        self.assertEqual(summary["games"], 240)
        self.assertEqual(set(summary["phases"]), {
            "self_play", "training", "arena_telemetry", "serialization", "other"
        })

    def test_profile_combination_amortizes_milestones(self):
        hardware = {"cuda_available": True, "cuda_device": "test"}
        ordinary = {
            "profile": "ordinary",
            "smoke": False,
            "games": 240,
            "num_mcts_sims": 96,
            "hardware": hardware,
            "output": "ordinary",
            "phases": {},
        }
        milestone = dict(ordinary, profile="milestone", output="milestone", phases={})
        for phase in (
            "self_play", "training", "arena_telemetry", "serialization", "other"
        ):
            ordinary["phases"][phase] = {"seconds": 10.0}
            milestone["phases"][phase] = {"seconds": 20.0}
        result = combine_profiles(ordinary, milestone, 10)
        self.assertEqual(result["phases"]["self_play"]["seconds"], 11.0)
        self.assertEqual(result["wall_total_seconds"], 55.0)


if __name__ == "__main__":
    unittest.main()
