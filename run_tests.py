"""Run the complete test suite."""
import subprocess
import sys


def run_tests():
    """Run all tests with pytest.

    The suite contains both unittest-style and pytest-style tests, so pytest is
    the only runner that exercises every file under tests/.
    """
    return subprocess.call([sys.executable, "-m", "pytest", "-q"])


if __name__ == "__main__":
    sys.exit(run_tests())
