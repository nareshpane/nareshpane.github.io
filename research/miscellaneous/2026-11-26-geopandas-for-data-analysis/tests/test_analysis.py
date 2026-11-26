"""Analytical checks on small geometries with known solutions."""
import importlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import zipfile

from esda.moran import Moran
import geopandas as gpd
from libpysal.weights import W
import numpy as np
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import weighted_quantile

acquire = importlib.import_module('00_download_data')
health = importlib.import_module('02_healthcare_accessibility')
wildfire = importlib.import_module('03_wildfire_exposure')
flood = importlib.import_module('04_flood_infrastructure')
stats = importlib.import_module('05_socioeconomic_geography')


class MathChecks(unittest.TestCase):
    def test_weighted_quantile(self):
        self.assertEqual(weighted_quantile([40, 2, 10], [1, 3, 6], .5), 10)

    def test_finite_candidate_search(self):
        updated, means = health.placement(np.array([6, 4]), np.array([[1, 5], [3, 1]]),
                                          np.array([10, 2]))
        np.testing.assert_array_equal(updated, [[1, 5], [3, 1]])
        np.testing.assert_allclose(means, [16 / 12, 52 / 12])

    def test_web_distances_keep_coverage_cutoffs(self):
        measured = np.array([10, 10.000267, 24.99997, 50.00049, 100.000198, 110.00112])
        encoded = np.array(health.web_distances(measured))
        for cutoff in health.THRESHOLDS:
            np.testing.assert_array_equal(encoded <= cutoff, measured <= cutoff)
        self.assertEqual(encoded[-1], 110.001)

    def test_unique_burn_area(self):
        left = box(0, 0, 2, 2)
        right = box(1, 0, 3, 2)
        grid = gpd.GeoDataFrame(geometry=[box(0, 0, 4, 2), box(4, 0, 5, 2)], crs=3400)
        union = gpd.GeoSeries([left, right], crs=3400).union_all()
        self.assertEqual(union.area, 6)
        np.testing.assert_allclose(wildfire.grid_fraction(grid, union), [.75, 0])

    def test_flood_intersection_excludes_boundary_point(self):
        roads = gpd.GeoDataFrame(geometry=[LineString([(0, 1), (4, 1)]),
                                           LineString([(0, 0), (1, 0)])], crs=3400)
        wet = flood.intersect(roads, box(1, 0, 3, 2), 'LineString', 'metres')
        self.assertEqual(len(wet), 1)
        self.assertAlmostEqual(wet.metres.iloc[0], 2)
        self.assertAlmostEqual(wet.fraction.iloc[0], .5)

    def test_flood_roof_fraction(self):
        roofs = gpd.GeoDataFrame(geometry=[box(0, 0, 4, 2)], crs=3400)
        wet = flood.intersect(roofs, box(1, -1, 2, 3), 'Polygon', 'square_metres')
        self.assertAlmostEqual(wet.square_metres.iloc[0], 2)
        self.assertAlmostEqual(wet.fraction.iloc[0], .25)

    def test_moran_and_seed_reproducibility(self):
        weights = W({0: [1], 1: [0], 2: [3], 3: [2]}, silence_warnings=True)
        weights.transform = 'r'
        values = np.array([0., 0., 1., 1.])
        left = stats.permutation(values, weights, simulations=29)
        right = stats.permutation(values, weights, simulations=29)
        self.assertAlmostEqual(left[0]['I'], 1)
        self.assertAlmostEqual(left[0]['I'], Moran(values, weights, permutations=0).I)
        self.assertAlmostEqual(left[0]['expected'], -1 / 3)
        np.testing.assert_array_equal(left[4], right[4])

    def test_bh_adjustment(self):
        np.testing.assert_allclose(stats.fdr(np.array([.01, .04, .03, .002])),
                                   [.02, .04, .04, .008])

    def test_archive_cannot_escape(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'unsafe.zip'
            with zipfile.ZipFile(archive, 'w') as handle:
                handle.writestr('../outside.txt', 'outside')
            with self.assertRaises(ValueError):
                acquire.extract(archive, root / 'raw')
            self.assertFalse((root / 'outside.txt').exists())


if __name__ == '__main__':
    unittest.main()

