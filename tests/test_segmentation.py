import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segmentation import KMeans, load_csv, main  # noqa: E402


class KMeansTests(unittest.TestCase):
    def setUp(self):
        self.rows = [[1.0, 1.0], [1.2, 0.9], [8.0, 8.0], [8.2, 7.9]]

    def test_finds_two_obvious_groups(self):
        model = KMeans(n_clusters=2, random_state=7).fit(self.rows)
        self.assertEqual(len(model.labels_), 4)
        self.assertEqual(model.labels_[0], model.labels_[1])
        self.assertEqual(model.labels_[2], model.labels_[3])
        self.assertNotEqual(model.labels_[0], model.labels_[2])

    def test_seed_makes_results_repeatable(self):
        first = KMeans(n_clusters=2, random_state=11).fit(self.rows)
        second = KMeans(n_clusters=2, random_state=11).fit(self.rows)
        self.assertEqual(first.labels_, second.labels_)
        self.assertAlmostEqual(first.inertia_, second.inertia_)

    def test_predict_uses_fitted_centers(self):
        model = KMeans(n_clusters=2).fit(self.rows)
        predicted = model.predict([[1.1, 1.1], [8.1, 8.1]])
        self.assertEqual(predicted[0], model.labels_[0])
        self.assertEqual(predicted[1], model.labels_[2])

    def test_predict_before_fit_fails(self):
        with self.assertRaises(RuntimeError):
            KMeans().predict([[1, 2]])

    def test_rejects_empty_rows(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            KMeans().fit([])

    def test_rejects_inconsistent_row_width(self):
        with self.assertRaisesRegex(ValueError, "same"):
            KMeans(n_clusters=1).fit([[1, 2], [3]])

    def test_rejects_too_many_clusters(self):
        with self.assertRaisesRegex(ValueError, "n_clusters"):
            KMeans(n_clusters=5).fit(self.rows)

    def test_rejects_prediction_feature_mismatch(self):
        model = KMeans(n_clusters=2).fit(self.rows)
        with self.assertRaisesRegex(ValueError, "Expected"):
            model.predict([[1, 2, 3]])

    def test_loads_sample_data(self):
        names, rows = load_csv(ROOT / "data" / "customers.csv")
        self.assertEqual(names, ["annual_income_k", "spending_score", "age"])
        self.assertEqual(len(rows), 18)

    def test_load_csv_feature_selection(self):
        names, rows = load_csv(ROOT / "data" / "customers.csv", ["age"])
        self.assertEqual(names, ["age"])
        self.assertEqual(len(rows[0]), 1)

    def test_load_csv_rejects_unknown_feature(self):
        with self.assertRaisesRegex(ValueError, "Unknown feature"):
            load_csv(ROOT / "data" / "customers.csv", ["loyalty"])

    def test_cli_smoke(self):
        self.assertEqual(main(["--data", str(ROOT / "data" / "customers.csv"), "--k", "3"]), 0)


if __name__ == "__main__":
    unittest.main()
