
# Customer lifetime Value Project
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
