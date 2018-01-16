#!/usr/bin/env python
import dill
import numpy
import pandas


def load_dataset():
    "Load example dataset from file and clean the data."
    try:
        df = pandas.read_csv(
            "data/orders.csv",
            dtype={
                "customer_id": numpy.str,
                "order_id": numpy.uint32,
                "order_item_id": numpy.float64,  # Workaround for bad data
                "num_items": numpy.float32,  # Same here
                "revenue": numpy.float32,
            },
            parse_dates=["created_at_date"]
        )
    except FileNotFoundError:
        raise RuntimeError("Orders data not found. Make sure it exists.")
    except ValueError:
        raise RuntimeError(
            "Dataset malformed. Ensure proper orders dataset.")

    # Drop "bad", assuming orders cannot be empty.
    df.dropna(inplace=True)
    # Convert columns to original type.
    df["order_item_id"] = df.order_item_id.astype(numpy.uint32)
    df["num_items"] = df.num_items.astype(numpy.uint16)

    return df


def get_max_items_in_order(df):
    """For every customer compute biggest ever number of items
       in any of their historical orders.

       Resulting DataFrame uses customer_id as index.
    """
    # First sum up items in each order.
    result = df.groupby(["customer_id", "order_id"]).agg(
        {"num_items": numpy.sum})
    # Then find biggest order for each customer.
    result = result.groupby("customer_id").agg({"num_items": numpy.max})
    # Rename column for clarity.
    return result.rename(columns={"num_items": "max_items"})


def get_max_revenue_in_order(df):
    """For every customer compute biggest ever revenue from any
       single historical order.

       Resulting DataFrame uses customer_id as index.
    """
    # Calculate total revenue per order.
    result = df.groupby(["customer_id", "order_id"]).agg(
        {"revenue": numpy.sum})
    # Find highest revenue order for each customer.
    result = result.groupby("customer_id").agg({"revenue": numpy.max})
    # Rename column for clarity.
    return result.rename(columns={"revenue": "max_revenue"})


def get_total_customer_revenue(df):
    """For every customer compute total revenue across all their orders.

       Resulting DataFrame uses customer_id as index.
    """
    result = df.groupby("customer_id").agg({"revenue": numpy.sum})
    # Rename column for clarity.
    return result.rename(columns={"revenue": "total_revenue"})


def get_total_customer_orders(df):
    """For every customer compute total number of their historical orders.

       Resulting DataFrame uses customer_id as index.
    """
    # Count unique order_ids for every customer_id.
    result = df.groupby("customer_id").agg({"order_id": "nunique"})
    # Rename column for clarity.
    return result.rename(columns={"order_id": "total_orders"})


def get_days_since_last_order(df, until="2017-10-17"):
    """For every customer compute number of days since their last order.

       Resulting DataFrame uses customer_id as index.
    """
    ts = pandas.Timestamp(until)
    # Find latest order for each customer.
    result = df.groupby("customer_id").agg({"created_at_date": numpy.max})
    # Calculate time difference and extract days.
    diff = (ts - result.created_at_date)
    result["days_since"] = diff / numpy.timedelta64(1, "D")
    # Drop unnecessary data.
    return result.drop(columns=["created_at_date"])


def get_longest_order_interval(df, days_since_df=None):
    """For every customer compute the longest interval in days between any two
       of their consecutive orders.

       In case of customer placing only one order the value is
       based on average longest interval and number days since the last order.

       Resulting DataFrame uses customer_id as index.
    """
    # Extract distinct orders dates for each customer.
    result = df.groupby(["customer_id", "order_id"]).agg(
        {"created_at_date": "first"})
    # Drop unnecessary data.
    result = result.reset_index("order_id").drop(columns="order_id")

    # Separate customers with multiple orders and just one order.
    group_sizes = result.groupby("customer_id").size()
    multiple = result[group_sizes > 1]
    single = result[group_sizes == 1]

    # Calculate difference in days between consecutive orders.
    difference = (multiple.groupby("customer_id")["created_at_date"].diff() /
                  numpy.timedelta64(1, "D"))
    multiple["difference"] = difference
    # Finally find longest intervals for repeated customers.
    ml = multiple.groupby("customer_id").agg({"difference": numpy.max})
    # Rename column for clarity.
    ml.rename(columns={"difference": "longest_interval"}, inplace=True)

    # Baseline value for single-order customers.
    base = ml.longest_interval.mean()

    # Variable component (based on last order).
    if days_since_df is None:
        sl = get_days_since_last_order(
            # Only crunch orders by previously found single-order customers.
            df[df.customer_id.isin(single.index)]
        )
    else:  # Cache data provided, just take relevant customers.
        sl = days_since_df[days_since_df.index.isin(single.index)]

    # Offset by baseline.
    sl = sl + base
    # Rename column for clarity.
    sl.rename(columns={"days_since": "longest_interval"}, inplace=True)

    # Merge two datasets.
    return pandas.concat([sl, ml])


def prepare_features(df):
    "Based on orders dataset prepares combined dataframe for the model."
    max_items = get_max_items_in_order(df)
    max_revenue = get_max_revenue_in_order(df)
    total_revenue = get_total_customer_revenue(df)
    total_orders = get_total_customer_orders(df)
    days_since = get_days_since_last_order(df)
    longest_interval = get_longest_order_interval(
        df, days_since_df=days_since)
    # Join results side-by-side.
    result = pandas.concat(
            [
                max_items, max_revenue, total_revenue,
                total_orders, days_since, longest_interval
            ],
            axis=1)

    return result


def load_model():
    try:
        with open("data/model.dill", "rb") as f:
            return dill.load(f)
    except FileNotFoundError:
        raise RuntimeError("Couldn't file the model file. Ensure it exists.")


def compute_clv(df):
    "Computes CLV data series for pre-formatted input data."
    if not isinstance(df, pandas.DataFrame):
        raise ValueError("Argument should be a DataFrame.")
    if df.columns.size != 6:
        raise ValueError("Expected properly transformed input (6 columns).")

    model = load_model()
    return model.predict(df.values)


def run_model():
    "Loads the orders data and computes CLV for each customer."
    df = load_dataset()

    transformed = prepare_features(df)

    clv_series = compute_clv(transformed)

    # Attach CLV column.
    transformed["clv"] = clv_series

    # Prepare CSV output with only necessary columns.
    output = transformed["clv"].to_csv(
        index_label="customer_id", header=["predicted_clv"])
    # Just send to standard output.
    print(output)


if __name__ == '__main__':
    run_model()
