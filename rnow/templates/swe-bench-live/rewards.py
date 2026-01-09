"""
SWE-bench Live Rewards

Test-based evaluation for issue resolution. The reward runs tests and checks
if the expected test transitions occurred (FAIL_TO_PASS and PASS_TO_PASS).
"""

from __future__ import annotations

import re
import subprocess

from rnow.core import RewardArgs, reward


def parse_pytest_output(output: str) -> dict[str, str]:
    """Parse pytest -rA output to extract test statuses.

    Returns dict mapping test name -> status (PASSED, FAILED, ERROR, SKIPPED)
    """
    results = {}

    # Match lines like: PASSED tests/test_foo.py::test_bar
    # or: FAILED tests/test_foo.py::test_bar - AssertionError
    patterns = [
        (r"^PASSED\s+(\S+)", "PASSED"),
        (r"^FAILED\s+(\S+)", "FAILED"),
        (r"^ERROR\s+(\S+)", "ERROR"),
        (r"^SKIPPED\s+(\S+)", "SKIPPED"),
    ]

    for line in output.split("\n"):
        line = line.strip()
        for pattern, status in patterns:
            match = re.match(pattern, line)
            if match:
                test_name = match.group(1)
                results[test_name] = status
                break

    return results


@reward(sandbox=True)
def resolved(args: RewardArgs, messages: list) -> float:
    """Check if the issue was resolved by running tests.

    Evaluates based on:
    1. FAIL_TO_PASS: Tests that should now pass (were failing before the fix)
    2. PASS_TO_PASS: Sample of tests that should still pass (explicit check)
    3. No new test failures (catch-all regression check)

    Returns:
        1.0 if all FAIL_TO_PASS tests pass AND no regressions
        0.5 if all FAIL_TO_PASS tests pass but introduced regressions
        0.0 if any FAIL_TO_PASS tests still fail
    """
    test_cmds = args.metadata.get("test_cmds", [])
    fail_to_pass = set(args.metadata.get("fail_to_pass", []))
    pass_to_pass = set(args.metadata.get("pass_to_pass", []))

    if not test_cmds:
        return 0.0

    # Run test commands
    all_output = ""
    for cmd in test_cmds:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd="/testbed",
            timeout=300,
        )
        all_output += result.stdout + "\n" + result.stderr + "\n"

    # Parse test results
    test_results = parse_pytest_output(all_output)

    # Check FAIL_TO_PASS transitions
    f2p_success = True
    for test in fail_to_pass:
        status = test_results.get(test)
        if status != "PASSED":
            f2p_success = False
            break

    if not f2p_success:
        return 0.0

    # Check PASS_TO_PASS (explicit regression tests)
    has_regressions = False
    for test in pass_to_pass:
        status = test_results.get(test)
        if status == "FAILED":
            has_regressions = True
            break

    # Also check for any other new failures
    if not has_regressions:
        for test, status in test_results.items():
            if status == "FAILED" and test not in fail_to_pass:
                has_regressions = True
                break

    if f2p_success and not has_regressions:
        return 1.0
    elif f2p_success:
        return 0.5  # Fixed the issue but introduced regressions
    else:
        return 0.0


@reward(precondition=True)
def has_changes(args: RewardArgs, messages: list) -> float:
    """Gate reward: Check if the agent made any file changes.

    Returns 0.0 if no tool calls were made that could modify files.
    This prevents rewarding agents that give up without trying.
    """
    modification_tools = {"write_file", "edit_file", "bash"}

    for msg in messages:
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                func_name = tc.get("function", {}).get("name", "")
                if func_name in modification_tools:
                    return 1.0

    return 0.0
