from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

from tqdm.auto import tqdm


class VerbosityStrategy(ABC):
    """Abstract base class for verbosity strategies."""

    @abstractmethod
    def on_train_begin(self, num_epochs: int) -> None:
        """Call when training starts."""
        self.num_epochs = num_epochs

    @abstractmethod
    def on_train_end(self) -> None:
        """Call when training ends."""

    @abstractmethod
    def on_epoch_begin(self, epoch: int) -> None:
        """Call at the beginning of each epoch."""

    @abstractmethod
    def on_epoch_end(self, epoch: int, avg_loss: float | None = None) -> None:
        """Call at the end of each epoch."""

    @abstractmethod
    def on_batch_end(self, batch_idx: int, loss: float | None = None) -> None:
        """Call after each batch."""

    @abstractmethod
    def wrap_epoch_iterator[T](self, iterable: Iterable[T]) -> Iterable[T]:
        """Wrap the epoch iterator (e.g., with tqdm)."""

    @abstractmethod
    def wrap_batch_iterator[T](self, iterable: Iterable[T]) -> Iterable[T]:
        """Wrap the batch iterator (e.g., with tqdm)."""


class SilentStrategy(VerbosityStrategy):
    """No output during training."""

    def on_train_begin(self, num_epochs: int) -> None:
        super().on_train_begin(num_epochs)

    def on_train_end(self) -> None:
        pass

    def on_epoch_begin(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int, avg_loss: float | None = None) -> None:
        pass

    def on_batch_end(self, batch_idx: int, loss: float | None = None) -> None:
        pass

    def wrap_epoch_iterator[T](self, iterable: Iterable[T]) -> Iterable[T]:
        return iterable

    def wrap_batch_iterator[T](self, iterable: Iterable[T]) -> Iterable[T]:
        return iterable


class PrintStrategy(VerbosityStrategy):
    """Use print statements for progress."""

    def on_train_begin(self, num_epochs: int) -> None:
        super().on_train_begin(num_epochs)
        self.max_epoch_len = len(str(num_epochs))

        print(f"Starting training for {self.num_epochs} epochs...")

    def on_epoch_begin(self, epoch: int) -> None:
        print(f"Epoch {epoch + 1:{self.max_epoch_len}d}/{self.num_epochs}")

    def on_batch_end(self, batch_idx: int, loss: float | None = None) -> None:
        if loss is None:
            print(f"  Batch {batch_idx + 1}")
        else:
            print(f"  Batch {batch_idx + 1}, Loss: {loss:.4g}")

    def on_epoch_end(self, epoch: int, avg_loss: float | None = None) -> None:
        if avg_loss is None:
            print(
                f"Epoch {epoch + 1:{self.max_epoch_len}d}/{self.num_epochs} completed",
            )
        else:
            print(
                f"Epoch {epoch + 1:{self.max_epoch_len}d}/{self.num_epochs} completed"
                f" - Average Loss: {avg_loss:.4g}",
            )

    def on_train_end(self) -> None:
        print("Training completed!")

    def wrap_epoch_iterator[T](self, iterable: Iterable[T]) -> Iterable[T]:
        return iterable

    def wrap_batch_iterator[T](self, iterable: Iterable[T]) -> Iterable[T]:
        return iterable


class TqdmStrategy(VerbosityStrategy):
    """Use tqdm for progress bars."""

    def __init__(self) -> None:
        self.epoch_bar = None
        self.batch_bar = None

    def on_train_begin(self, num_epochs: int) -> None:
        super().on_train_begin(num_epochs)

    def on_train_end(self) -> None:
        if self.epoch_bar:
            self.epoch_bar.close()

    def on_epoch_begin(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int, avg_loss: float | None = None) -> None:  # noqa: ARG002
        if self.epoch_bar is not None and avg_loss is not None:
            self.epoch_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
        if self.batch_bar is not None:
            self.batch_bar.close()

    def on_batch_end(self, batch_idx: int, loss: float | None = None) -> None:  # noqa: ARG002
        if self.batch_bar is not None and loss is not None:
            self.batch_bar.set_postfix({"loss": f"{loss:.4f}"})

    def wrap_epoch_iterator[T](
        self,
        iterable: Iterable[T],
        desc: str = "Training",
    ) -> Iterable[T]:
        self.epoch_bar = tqdm(iterable, desc=desc, unit="epoch")
        return self.epoch_bar

    def wrap_batch_iterator[T](
        self,
        iterable: Iterable[T],
        desc: str = "  Batches",
    ) -> Iterable[T]:
        self.batch_bar = tqdm(iterable, desc=desc, leave=False, unit="batch")
        return self.batch_bar


class FileLoggingStrategy(VerbosityStrategy):
    """Log training progress to a file."""

    def __init__(self, log_file: str | Path) -> None:
        self.log_file = Path(log_file)
        self.log_file.mkdir(parents=True, exist_ok=True)

        self.file: TextIO | None = None

    def _log(self, message: str) -> None:
        if self.file is not None:
            self.file.write(message + "\n")
            self.file.flush()

    def on_train_begin(self, num_epochs: int) -> None:
        super().on_train_begin(num_epochs)
        self.max_epoch_len = len(str(num_epochs))

        self.file = self.log_file.open("w")
        self._log(f"Starting training for {self.num_epochs} epochs...")

    def on_train_end(self) -> None:
        self._log("Training completed!")
        if self.file:
            self.file.close()

    def on_epoch_begin(self, epoch: int) -> None:
        self._log(f"\nEpoch {epoch + 1:{self.max_epoch_len}d}/{self.num_epochs}")

    def on_epoch_end(self, epoch: int, avg_loss: float | None = None) -> None:
        if avg_loss is None:
            self._log(
                f"Epoch {epoch + 1:{self.max_epoch_len}d}/{self.num_epochs} completed",
            )
        else:
            self._log(
                f"Epoch {epoch + 1:{self.max_epoch_len}d}/{self.num_epochs} completed"
                f" - Average Loss: {avg_loss:.4f}",
            )

    def on_batch_end(self, batch_idx: int, loss: float | None = None) -> None:
        if loss is None:
            self._log(f"  Batch {batch_idx + 1}")
        else:
            self._log(f"  Batch {batch_idx + 1}, Loss: {loss:.4f}")

    def wrap_epoch_iterator[T](self, iterable: Iterable[T]) -> Iterable[T]:
        return iterable

    def wrap_batch_iterator[T](self, iterable: Iterable[T]) -> Iterable[T]:
        return iterable


class CompositeStrategy(VerbosityStrategy):
    """Combine multiple strategies."""

    def __init__(self, *strategies: VerbosityStrategy) -> None:
        self.strategies = strategies

    def on_train_begin(self, num_epochs: int) -> None:
        for strategy in self.strategies:
            strategy.on_train_begin(num_epochs)

    def on_train_end(self) -> None:
        for strategy in self.strategies:
            strategy.on_train_end()

    def on_epoch_begin(self, epoch: int) -> None:
        for strategy in self.strategies:
            strategy.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch: int, avg_loss: float | None = None) -> None:
        for strategy in self.strategies:
            strategy.on_epoch_end(epoch, avg_loss)

    def on_batch_end(self, batch_idx: int, loss: float | None = None) -> None:
        for strategy in self.strategies:
            strategy.on_batch_end(batch_idx, loss)

    def wrap_epoch_iterator[T](self, iterable: Iterable[T]) -> Iterable[T]:
        # Use the first strategy that actually wraps
        for strategy in self.strategies:
            iterable = strategy.wrap_epoch_iterator(iterable)
        return iterable

    def wrap_batch_iterator[T](self, iterable: Iterable[T]) -> Iterable[T]:
        for strategy in self.strategies:
            iterable = strategy.wrap_batch_iterator(iterable)
        return iterable
