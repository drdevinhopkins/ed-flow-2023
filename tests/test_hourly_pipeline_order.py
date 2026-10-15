"""Exercise the real shell wrapper without models, network calls or uploads."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


WRAPPER = Path(__file__).resolve().parents[1] / "scripts/run_ed_flow_update.sh"
V21 = "scripts/hourly_forecast_v2_1.py"
VALIDATE = "scripts/validate_forecast_inputs.py"
LEGACY = "scripts/chronos_forecast.py"
V2 = "scripts/hourly_forecast_v2.py"
BLURB = "scripts/automation/blurb_automation_wrapper.py"


class HourlyPipelineOrderTests(unittest.TestCase):
    def run_wrapper(self, *, fail_script=""):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            activation = root / ".venv/bin/activate"
            activation.parent.mkdir(parents=True)
            activation.write_text('export PATH="$ED_FLOW_TEST_BIN:$PATH"\n')
            calls_path = root / "calls.txt"

            python = bin_dir / "python"
            python.write_text(
                '#!/bin/sh\n'
                'printf "%s\\n" "$1" >> "$ED_FLOW_TEST_CALLS"\n'
                'if [ "$1" = "$ED_FLOW_TEST_FAIL_SCRIPT" ]; then\n'
                '    exit 17\n'
                'fi\n'
            )
            python.chmod(0o755)

            # Always exercise the ordinary hourly path, regardless of the
            # wall clock on the machine running these tests.
            date = bin_dir / "date"
            date.write_text(
                '#!/bin/sh\n'
                'case "$1" in\n'
                '    +%H) printf "12\\n" ;;\n'
                '    +%F) printf "2026-10-15\\n" ;;\n'
                '    *) exit 1 ;;\n'
                'esac\n'
            )
            date.chmod(0o755)

            env = os.environ.copy()
            env.update(
                ED_FLOW_REPO=str(root),
                ED_FLOW_VENV=".venv",
                ED_FLOW_TEST_BIN=str(bin_dir),
                ED_FLOW_TEST_CALLS=str(calls_path),
                ED_FLOW_TEST_FAIL_SCRIPT=fail_script,
            )
            result = subprocess.run(
                ["bash", str(WRAPPER)],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            calls = calls_path.read_text().splitlines()
            return result, calls

    def test_dashboard_forecast_publishes_first_after_input_gate(self):
        result, calls = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            calls[1:4],
            ["scripts/get_current.py", VALIDATE, V21],
        )
        self.assertEqual(calls.count(V21), 1)
        self.assertLess(calls.index(V21), calls.index(LEGACY))
        self.assertLess(calls.index(V21), calls.index(V2))
        self.assertLess(calls.index(V21), calls.index(BLURB))

    def test_priority_failure_preserves_legacy_outputs_and_blocks_blurb(self):
        result, calls = self.run_wrapper(fail_script=V21)
        self.assertEqual(result.returncode, 17)
        for script in (
            LEGACY,
            "scripts/forecast_oncall_impact.py",
            "scripts/forecast_oncall_probability.py",
            V2,
        ):
            self.assertIn(script, calls)
        self.assertNotIn(BLURB, calls)
        self.assertNotIn("scripts/automation/llm_blurb_automation_wrapper.py", calls)
        self.assertIn("Skipping blurb publication", result.stderr)
        self.assertIn("=== FAILED", result.stderr)
        self.assertNotIn(f"=== DONE [00:00:00]: python {V21}", result.stdout)

    def test_failed_input_gate_blocks_every_forecast(self):
        result, calls = self.run_wrapper(fail_script=VALIDATE)
        self.assertEqual(result.returncode, 17)
        self.assertNotIn(V21, calls)
        self.assertNotIn(LEGACY, calls)
        self.assertNotIn(V2, calls)
        self.assertNotIn(BLURB, calls)

    def test_legacy_failure_cannot_delay_priority_forecast(self):
        result, calls = self.run_wrapper(fail_script=LEGACY)
        self.assertEqual(result.returncode, 17)
        self.assertLess(calls.index(V21), calls.index(LEGACY))
        self.assertNotIn(BLURB, calls)

    def test_optional_v2_failure_does_not_block_blurb(self):
        result, calls = self.run_wrapper(fail_script=V2)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(BLURB, calls)


if __name__ == "__main__":
    unittest.main()
