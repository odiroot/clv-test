
# Customer Lifetime Value Project

## Source and data files

Download using Git:

    $ git clone https://github.com/odiroot/clv-test.git
    $ cd clv-test
## Requirements

1. Ensure Python 3.6.x is installed.
2. Use a virtual Python environment if possible (virtualenv).
3. Install required libraries:

        $ pip install -r requirements.txt

## Running

### Console tool

Run like any usual executable:

    $ ./process.py

Dataset from `data/orders.csv` will be loaded and processed using the model from `data/model.dill`. The result will be sent to the standard output.

In order to capture the output into a file use stream redirection:

    $ ./process.py > output.csv


### Unit tests

Execute directly the test file:

    $ ./test.py

It's possible to run the test suite using `nosetests` or `py.test` as well.

### Web application

Execute the simple application server:

    $ ./app.py

It will take a few seconds before the application is ready to accept HTTP requests. The server will listen on the local network interface at `5000` port.

The *CLV* value for a specific customer can be obtained at the single defined endpoint:

    http://localhost:5000/customer/<customer_id>/clv

For example:

    $ curl http://localhost:5000/customer/00739721734f20419f6544459a1f9983/clv
    {
        "customer_id": "00739721734f20419f6544459a1f9983",
        "predicted_clv": 309.9453539967175
    }
