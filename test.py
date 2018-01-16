#!/usr/bin/env python
import unittest

import numpy
import pandas

from process import (
    load_model, load_dataset, get_max_items_in_order, get_max_revenue_in_order,
    get_total_customer_revenue)


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

    def test_load_dataset(self):
        result = load_dataset("data/test.csv")

        self.assertEqual(result.shape, (1, 6))
        self.assertEqual(result.revenue[0], numpy.float32(98.76))
        self.assertEqual(result.num_items.dtype, numpy.uint16)


fake_orders = pandas.DataFrame({
    'customer_id': {
        0: 'a', 1: 'a', 2: 'b', 3: 'b', 4: 'b'},
    'item_id': {
        0: '10', 1: '15', 2: '30', 3: '40', 4: '45'},
    'num_items': {
        0: 1, 1: 2, 2: 1, 3: 2, 4: 2},
    'order_id': {
        0: '1', 1: '2', 2: '3', 3: '3', 4: '4'},
    'revenue': {
        0: 55, 1: 100, 2: 35, 3: 90, 4: 200},
})


class ProcessTest(unittest.TestCase):
    def test_get_max_items_in_order(self):
        result = get_max_items_in_order(fake_orders)

        self.assertEqual(result.max_items[0], 2)
        self.assertEqual(result.max_items[1], 3)

    def test_max_revenue(self):
        result = get_max_revenue_in_order(fake_orders)

        self.assertEqual(result.max_revenue[0], 100)
        self.assertEqual(result.max_revenue[1], 200)

    def test_total_revenue(self):
        result = get_total_customer_revenue(fake_orders)

        self.assertEqual(result.total_revenue[0], 155)
        self.assertEqual(result.total_revenue[1], 325)


if __name__ == '__main__':
    unittest.main()
