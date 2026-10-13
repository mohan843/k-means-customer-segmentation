"""Dependency-free K-Means customer segmentation for numeric CSV data."""
from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class KMeans:
    """K-Means clustering with feature standardization and seeded initialization."""

    n_clusters: int = 3
    max_iter: int = 100
    tolerance: float = 1e-6
    random_state: int = 42

    def fit(self, rows: Sequence[Sequence[float]]) -> "KMeans":
        if not rows:
            raise ValueError("Cannot fit K-Means to an empty dataset")
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            raise ValueError("All rows must have the same non-zero number of features")
        if self.n_clusters < 1 or self.n_clusters > len(rows):
            raise ValueError("n_clusters must be between 1 and the number of rows")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")

        self.means_ = [sum(row[j] for row in rows) / len(rows) for j in range(width)]
        self.scales_ = []
        for j, mean in enumerate(self.means_):
            variance = sum((row[j] - mean) ** 2 for row in rows) / len(rows)
            self.scales_.append(variance ** 0.5 or 1.0)
        normalized = [self._normalize(row) for row in rows]

        rng = random.Random(self.random_state)
        centers = [list(normalized[i]) for i in rng.sample(range(len(rows)), self.n_clusters)]
        labels = [-1] * len(rows)
        for _ in range(self.max_iter):
            new_labels = [self._nearest(point, centers) for point in normalized]
            new_centers = []
            for cluster_id, old_center in enumerate(centers):
                members = [point for point, label in zip(normalized, new_labels) if label == cluster_id]
                new_centers.append(
                    [sum(point[j] for point in members) / len(members) for j in range(width)]
                    if members else list(old_center)
                )
            shift = max(self._distance(a, b) for a, b in zip(centers, new_centers))
            centers, labels = new_centers, new_labels
            if shift <= self.tolerance:
                break

        self.labels_ = labels
        self.cluster_centers_ = centers
        self.inertia_ = sum(self._distance(point, centers[label]) ** 2 for point, label in zip(normalized, labels))
        self.n_features_in_ = width
        return self

    def predict(self, rows: Sequence[Sequence[float]]) -> list[int]:
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("Call fit before predict")
        if any(len(row) != self.n_features_in_ for row in rows):
            raise ValueError(f"Expected {self.n_features_in_} features per row")
        return [self._nearest(self._normalize(row), self.cluster_centers_) for row in rows]

    def _normalize(self, row: Sequence[float]) -> list[float]:
        if any(not isinstance(value, (int, float)) for value in row):
            raise ValueError("Features must be numeric")
        return [(value - mean) / scale for value, mean, scale in zip(row, self.means_, self.scales_)]

    @staticmethod
    def _distance(a: Sequence[float], b: Sequence[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    @classmethod
    def _nearest(cls, point: Sequence[float], centers: Sequence[Sequence[float]]) -> int:
        return min(range(len(centers)), key=lambda index: cls._distance(point, centers[index]))


def load_csv(path: str | Path, features: Sequence[str] | None = None) -> tuple[list[str], list[list[float]]]:
    """Read numeric customer features from a CSV with a header row."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must include a header row")
        selected = list(features) if features else list(reader.fieldnames)
        missing = [name for name in selected if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Unknown feature(s): {', '.join(missing)}")
        rows = []
        for line_number, record in enumerate(reader, start=2):
            try:
                rows.append([float(record[name]) for name in selected])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric or missing value on CSV line {line_number}") from exc
    return selected, rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cluster customers using standardized K-Means.")
    parser.add_argument("--data", default="data/customers.csv", help="CSV file path")
    parser.add_argument("--features", nargs="+", help="Numeric CSV columns (default: all)")
    parser.add_argument("--k", type=int, default=3, help="Number of customer segments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable results")
    args = parser.parse_args(argv)
    try:
        feature_names, rows = load_csv(args.data, args.features)
        model = KMeans(n_clusters=args.k, random_state=args.seed).fit(rows)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Customers: {len(rows)} | Features: {', '.join(feature_names)} | Segments: {args.k}")
    print(f"Within-cluster sum of squares (standardized): {model.inertia_:.3f}")
    print("\nCustomer segments:")
    for cluster_id in range(args.k):
        members = [i + 1 for i, label in enumerate(model.labels_) if label == cluster_id]
        center = [model.cluster_centers_[cluster_id][j] * model.scales_[j] + model.means_[j]
                  for j in range(len(feature_names))]
        print(f"  Segment {cluster_id}: {len(members)} customers; rows {members}; "
              f"average profile {dict(zip(feature_names, (round(value, 1) for value in center)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
