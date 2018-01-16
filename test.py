#!/usr/bin/env python
import unittest

import numpy

from process import load_model, load_dataset


class SmokeTest(unittest.TestCase):
    def test_model(self):
        model = load_model()

        arr = numpy.array([
            [3, 92.6, 109.3, 2, 12, 26],
            [2, 10.4, 43.5, 3, 26, 5]
        ])

        score = model.predict(arr)

        self.assertAlmostEqual(score[0], 244.9)
        self.assertAlmostEqual(score[1], 89.9)


class ProcessTest(unittest.TestCase):
    def test_load_dataset(self):
        result = load_dataset("data/test.csv")

        self.assertEqual(result.shape, (1, 6))
        self.assertEqual(result.revenue[0], numpy.float32(98.76))
        self.assertEqual(result.num_items.dtype, numpy.uint16)


if __name__ == '__main__':
    unittest.main()
