#!/usr/bin/env python
import numpy   # Without this serialized model code crashes.
import logging
from flask import Flask, abort, jsonify

from process import load_dataset, prepare_features, compute_clv


log = logging.getLogger(__name__)


class OrdersApp(Flask):
    dataset = None

    def preload_data(self):
        df = load_dataset()

        transformed = prepare_features(df)
        clv_series = compute_clv(transformed)

        transformed["clv"] = clv_series
        self.dataset = transformed


app = OrdersApp(__name__)


@app.route("/customer/<string:customer_id>/clv")
def get_customer_clv(customer_id):
    # Verify the customer exists in dataset.
    if customer_id not in app.dataset.index:
        return abort(404)

    # Return preloaded value.
    clv = app.dataset.clv[customer_id]
    return jsonify({
        "customer_id": customer_id,
        "predicted_clv": clv
    })


def main():
    log.warn("Preloading data and model. This can take some time.")
    app.preload_data()
    log.warn("Finished preloading.")
    app.run()


if __name__ == '__main__':
    main()
