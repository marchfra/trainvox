# trainvox

This library provides useful strategies to display progress during machine
learning training/evaluation loops.

These strategies are implemented using a common interface (as the strategy pattern
dictates), but vary in how they display the progress.

## Installation

To install the library you can either clone this repository or use the command

```shell
pip install trainvox@git+https://github.com/marchfra/trainvox.git
```

## Usage

To use the library in your training loop, simply import your chosen strategy,
instantiate the relative object, and call the relevant method at the appropriate
time.

Below is an example code:

```python
...  # Other imports
from trainvox import TqdmStrategy

...  # Problem setup

v = TqdmStrategy()

v.on_train_begin(num_epochs)
for epoch in v.wrap_epoch_iterator(range(num_epochs)):
    v.on_epoch_begin(epoch)

    for idx, batch in enumerate(v.wrap_batch_iterator(loader)):

        ...  # Training loop

        v.on_batch_end(idx, total_loss)

    v.on_epoch_end(epoch)

v.on_train_end()
```

If you don't want to display the progress over batches, you can just omit the
call of `v.wrap_batch_iterator` and `v.on_batch_end`.

## Implementing new strategies

Implementing a new strategy is as simple as inheriting from
`torchvox.VerbosityStrategy` and defining all the required methods.
