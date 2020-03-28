# Parallel Deep Network CLI
A command-line interface to train various neural network architectures using asynchronous stochastic gradient descent (Hogwild!) using PyTorch.

## Getting started

Create a virtual environment `python3 -m venv env; source env/bin/activate` and install dependencies `pip install -r requirements`.

You can then launch the application by running `python src/main.py`, which will train a feedforward neural network with 1 epoch, using 1 process and a batch size of 12. All logs will be saved in the `src/logs` folder.

## Settings

To view available settings run `python src/main.py --help`.

Available options are:

```
--epochs INTEGER    number of epochs to train neural network.
--arch TEXT         neural network architecture to benchmark (conv or ff).
--distributed TEXT  whether to distribute data or not (y or n).
--procs INTEGER     number of processes to spawn.
--nodes INTEGER     number of cores to use.
--batches INTEGER   minibatch size to use.
--help              Show this message and exit.
```


