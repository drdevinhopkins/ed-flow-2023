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
    def run_wrapper(self, *, fail_script="", hour="12", last_run="2026-10-15", runs=1, hours=None):
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

            # Freeze the local clock and completion marker independently.
            date = bin_dir / "date"
            date.write_text(
                '#!/bin/sh\n'
                'case "$1" in\n'
                '    +%H) printf "%s\\n" "$ED_FLOW_TEST_HOUR" ;;\n'
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
                ED_FLOW_TEST_HOUR=hour,
            )
            marker = root / ".state/daily-arrival-forecast.last-run"
            if last_run is not None:
                marker.parent.mkdir(parents=True)
                marker.write_text(last_run + "\n")
            for current_hour in (hours if hours is not None else [hour] * runs):
                env["ED_FLOW_TEST_HOUR"] = current_hour
                result = subprocess.run(
                    ["bash", str(WRAPPER)], env=env, capture_output=True,
                    text=True, timeout=10,
                )
            result.daily_last_run = marker.read_text().strip() if marker.exists() else None
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


class DailyForecastCatchupTests(unittest.TestCase):
    run_wrapper = HourlyPipelineOrderTests.run_wrapper

    def test_completed_day_does_not_repeat(self):
        result, calls = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("scripts/forecast_daily_visits_from_daily.py", calls)

    def test_not_before_six(self):
        result, calls = self.run_wrapper(hour="05", last_run=None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("scripts/forecast_daily_visits_from_daily.py", calls)
        self.assertIsNone(result.daily_last_run)

    def test_missed_daily_run_catches_up_later_once(self):
        for hour in ["06", "08", "13", "23"]:
            for last_run in [None, "2026-10-14"]:
                with self.subTest(hour=hour, last_run=last_run):
                    result, calls = self.run_wrapper(hour=hour, last_run=last_run, runs=2)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    for script in ["scripts/forecast_daily_visits_from_daily.py",
                                   "scripts/explain_daily_visits_forecast.py",
                                   "scripts/build_daily_arrival_outlook.py"]:
                        self.assertEqual(calls.count(script), 1)
                    self.assertEqual(result.daily_last_run,
                                     "2026-10-15 " + ("evening" if int(hour) >= 18 else "morning"))

    def test_failed_daily_chain_can_retry_and_never_marks_success(self):
        for failed in ["scripts/forecast_daily_visits_from_daily.py",
                       "scripts/explain_daily_visits_forecast.py",
                       "scripts/build_daily_arrival_outlook.py"]:
            with self.subTest(failed=failed):
                result, calls = self.run_wrapper(last_run="2026-10-14", fail_script=failed, runs=2)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.daily_last_run, "2026-10-14")
                self.assertEqual(calls.count("scripts/forecast_daily_visits_from_daily.py"), 2)
                self.assertIn("additive step failed", result.stderr)

    def test_legacy_morning_marker_does_not_suppress_evening(self):
        result, calls = self.run_wrapper(hour="18", last_run="2026-10-15")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls.count("scripts/forecast_daily_visits_from_daily.py"), 1)
        self.assertEqual(result.daily_last_run, "2026-10-15 evening")
        self.assertIn("--issue-slot evening", result.stdout)

    def test_both_windows_run_once_and_retry_is_not_hourly_republishing(self):
        result, calls = self.run_wrapper(last_run=None, hours=["05", "06", "12", "17", "18", "19", "23"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls.count("scripts/forecast_daily_visits_from_daily.py"), 2)
        self.assertEqual(result.daily_last_run, "2026-10-15 evening")

    def test_failed_evening_keeps_morning_completion(self):
        result, calls = self.run_wrapper(hour="18", last_run="2026-10-15 morning",
                                        fail_script="scripts/build_daily_arrival_outlook.py", runs=2)
        self.assertEqual(result.daily_last_run, "2026-10-15 morning")
        self.assertEqual(calls.count("scripts/forecast_daily_visits_from_daily.py"), 2)


if __name__ == "__main__":
    unittest.main()
