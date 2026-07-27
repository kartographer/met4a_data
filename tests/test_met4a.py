import pickle
import unittest
from pathlib import Path

import numpy as np

from met4a import (
    MET4A_SAMPLE_DTYPE,
    find_time_gaps,
    load_met4a_pickle,
    load_met4a_pickles,
    met4a_dict_to_recarray,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "2026_02_12"
PICKLE_PATHS = sorted(DATA_DIRECTORY.glob("*.met4a.pkl"))


class TestMet4aData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PICKLE_PATHS:
            raise unittest.SkipTest(
                f"No MET4A pickle files found in {DATA_DIRECTORY}"
            )

        cls.first_path = PICKLE_PATHS[0]
        cls.first_dictionary = load_met4a_pickle(cls.first_path)
        cls.first_samples = met4a_dict_to_recarray(cls.first_dictionary)
        cls.all_samples = load_met4a_pickles(DATA_DIRECTORY)

    def test_pickle_contains_expected_dictionary_fields(self):
        expected_fields = {
            "start_time",
            "ref_pressure",
            "scale_fac",
            "nsamp",
            "times",
            "pressure",
            "temp",
            "rh",
        }

        self.assertIsInstance(self.first_dictionary, dict)
        self.assertEqual(set(self.first_dictionary), expected_fields)

    def test_single_dictionary_converts_to_recarray(self):
        data = self.first_dictionary
        samples = self.first_samples

        self.assertIsInstance(samples, np.recarray)
        self.assertEqual(samples.dtype, MET4A_SAMPLE_DTYPE)
        self.assertEqual(len(samples), data["nsamp"])
        np.testing.assert_allclose(
            samples.time,
            data["start_time"] + data["times"],
        )
        np.testing.assert_allclose(samples.pressure, data["pressure"])
        np.testing.assert_allclose(samples.temp, data["temp"])
        np.testing.assert_allclose(samples.rh, data["rh"])
        np.testing.assert_allclose(samples.ref_pressure, data["ref_pressure"])
        np.testing.assert_allclose(samples.scale_fac, data["scale_fac"])

    def test_directory_loads_as_one_recarray(self):
        self.assertIsInstance(self.all_samples, np.recarray)
        self.assertEqual(self.all_samples.dtype, MET4A_SAMPLE_DTYPE)

        expected_count = 0
        for path in PICKLE_PATHS:
            with path.open("rb") as file:
                expected_count += pickle.load(file)["nsamp"]

        self.assertEqual(len(self.all_samples), expected_count)

    def test_concatenated_samples_are_chronological(self):
        time_differences = np.diff(self.all_samples.time)

        self.assertTrue(np.all(time_differences > 0))

    def test_concatenation_preserves_first_and_last_times(self):
        block_times = []
        for path in PICKLE_PATHS:
            data = load_met4a_pickle(path)
            block_times.append(data["start_time"] + data["times"])

        expected_first = min(times[0] for times in block_times)
        expected_last = max(times[-1] for times in block_times)

        self.assertEqual(self.all_samples.time[0], expected_first)
        self.assertEqual(self.all_samples.time[-1], expected_last)

    def test_gap_report_matches_large_time_intervals(self):
        gaps = find_time_gaps(self.all_samples)
        differences = np.diff(self.all_samples.time)
        sample_period = np.median(differences)
        expected_indices = np.flatnonzero(differences > 5.0 * sample_period)

        np.testing.assert_array_equal(gaps.index_before, expected_indices)
        np.testing.assert_allclose(
            gaps.duration,
            differences[expected_indices],
        )


if __name__ == "__main__":
    unittest.main()
