import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from examples import laminarFlame, optimization


class TestExamples(unittest.TestCase):
    _directory_prefix = "test/resources"

    def test_laminar_flame(self):
        with TemporaryDirectory(dir=self._directory_prefix) as temp_dir:
            laminarFlame.main(show_results=False, results_location=temp_dir)
            results_filename = os.path.join(temp_dir, "laminarFlame.npz")
            baseline_filename = os.path.join(self._directory_prefix, "laminarFlame.npz")
            self.assertResultsCloseToBaseline(results_filename, baseline_filename)

    def assertResultsCloseToBaseline(self, results_filename: str, baseline_filename: str, msg=None):
        # Loads the results and baseline npz files and checks that they are close
        results = np.load(results_filename)
        baseline = np.load(baseline_filename)
        for name in baseline:
            if name not in results:
                msg = self._formatMessage(msg, f"'{name}' not found in results.")
                raise self.failureException(msg)
            if not np.allclose(baseline[name], results[name]):
                msg = self._formatMessage(msg, f"'{name}' is not found to be close to the baseline.")
                raise self.failureException(msg)


if __name__ == '__main__':
    unittest.main()
