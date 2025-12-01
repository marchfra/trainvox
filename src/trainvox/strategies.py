from abc import ABC, abstractmethod
from collections.abc import Iterable, Sized
from typing import TypeVar

from tqdm.auto import tqdm
from tqdm.contrib.telegram import tqdm_telegram

from .utils import send_telegram_message

T = TypeVar("T")


class VerbosityStrategy(ABC):
    """Abstract base class for verbosity strategies."""

    @abstractmethod
    def on_train_begin(self, num_epochs: int, msg: str) -> None:
        """Call when training starts."""
        self.num_epochs = num_epochs

    @abstractmethod
    def on_train_end(self, msg: str) -> None:
        """Call when training ends."""

    @abstractmethod
    def on_epoch_begin(self, epoch: int) -> None:
        """Call at the beginning of each epoch."""

    @abstractmethod
    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float | None = None,
        val_loss: float | None = None,
    ) -> None:
        """Call at the end of each epoch."""

    @abstractmethod
    def on_batch_end(
        self,
        batch_idx: int | None = None,
        loss: float | None = None,
    ) -> None:
        """Call after each batch."""

    @abstractmethod
    def wrap_epoch_iterator(
        self,
        iterable: Iterable[T],
        desc: str,
        unit: str,
    ) -> Iterable[T]:
        """Wrap the epoch iterator (e.g., with tqdm)."""

    @abstractmethod
    def wrap_batch_iterator(
        self,
        iterable: Iterable[T],
        desc: str,
        unit: str,
    ) -> Iterable[T]:
        """Wrap the batch iterator (e.g., with tqdm)."""


class SilentStrategy(VerbosityStrategy):
    """No output during training."""

    def on_train_begin(self, num_epochs: int, msg: str) -> None:
        super().on_train_begin(num_epochs, msg)

    def on_train_end(self, msg: str) -> None:
        pass

    def on_epoch_begin(self, epoch: int) -> None:
        pass

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float | None = None,
        val_loss: float | None = None,
    ) -> None:
        pass

    def on_batch_end(
        self,
        batch_idx: int | None = None,
        loss: float | None = None,
    ) -> None:
        pass

    def wrap_epoch_iterator(
        self,
        iterable: Iterable[T],
        desc: str = "",  # noqa: ARG002
        unit: str = "",  # noqa: ARG002
    ) -> Iterable[T]:
        return iterable

    def wrap_batch_iterator(
        self,
        iterable: Iterable[T],
        desc: str = "",  # noqa: ARG002
        unit: str = "",  # noqa: ARG002
    ) -> Iterable[T]:
        return iterable


class PrintStrategy(VerbosityStrategy):
    """Use print statements for progress."""

    def __init__(self) -> None:
        self.num_batches: int | None = None

    @property
    def max_batch_len(self) -> int:
        if self.num_batches is not None:
            return len(str(self.num_batches))
        return -1

    def on_train_begin(self, num_epochs: int, msg: str = "Starting training") -> None:
        super().on_train_begin(num_epochs, msg=msg)
        self.max_epoch_len = len(str(num_epochs))

        print(f"{msg} for {self.num_epochs} epochs...")

    def on_epoch_begin(self, epoch: int) -> None:
        print(f"Epoch {epoch + 1:{self.max_epoch_len}d}/{self.num_epochs}")

    def on_batch_end(
        self,
        batch_idx: int | None = None,
        loss: float | None = None,
    ) -> None:
        msg = "  "
        if batch_idx is not None:
            if self.num_batches is not None:
                msg += f"Batch {batch_idx + 1:{self.max_batch_len}d}/{self.num_batches}"
            else:
                msg += f"Batch {batch_idx + 1}"
        if loss is not None:
            if batch_idx is not None:
                msg += ", "
            msg += f"Loss: {loss:.4g}"

        print(msg)

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float | None = None,
        val_loss: float | None = None,
    ) -> None:
        msg = f"Epoch {epoch + 1:{self.max_epoch_len}d}/{self.num_epochs} completed"
        if train_loss is not None:
            msg += f" - Training Loss: {train_loss:.4g}"
        if val_loss is not None:
            msg += f" - Validation Loss: {val_loss:.4g}"

        print(msg)

    def on_train_end(self, msg: str = "Training completed!") -> None:
        print(msg)

    def wrap_epoch_iterator(
        self,
        iterable: Iterable[T],
        desc: str = "",  # noqa: ARG002
        unit: str = "",  # noqa: ARG002
    ) -> Iterable[T]:
        return iterable

    def wrap_batch_iterator(
        self,
        iterable: Iterable[T],
        desc: str = "",  # noqa: ARG002
        unit: str = "",  # noqa: ARG002
    ) -> Iterable[T]:
        if isinstance(iterable, Sized):
            self.num_batches = len(iterable)
        return iterable


class TqdmStrategy(VerbosityStrategy):
    """Use tqdm for progress bars."""

    def __init__(self) -> None:
        self.epoch_bar = None
        self.batch_bar = None

    def on_train_begin(self, num_epochs: int, msg: str) -> None:
        super().on_train_begin(num_epochs, msg)

    def on_train_end(self, msg: str) -> None:  # noqa: ARG002
        if self.epoch_bar:
            self.epoch_bar.close()

    def on_epoch_begin(self, epoch: int) -> None:
        pass

    def on_epoch_end(
        self,
        epoch: int,  # noqa: ARG002
        train_loss: float | None = None,
        val_loss: float | None = None,
    ) -> None:
        if self.epoch_bar is not None:
            postfix: dict[str, str] = {}
            if train_loss is not None:
                postfix["train_loss"] = f"{train_loss:.4g}"
            if val_loss is not None:
                postfix["val_loss"] = f"{val_loss:.4g}"
            self.epoch_bar.set_postfix(postfix)

        if self.batch_bar is not None:
            self.batch_bar.close()

    def on_batch_end(
        self,
        batch_idx: int | None = None,  # noqa: ARG002
        loss: float | None = None,
    ) -> None:
        if self.batch_bar is not None and loss is not None:
            self.batch_bar.set_postfix({"loss": f"{loss:.4g}"})

    def wrap_epoch_iterator(
        self,
        iterable: Iterable[T],
        desc: str = "Training",
        unit: str = "epoch",
    ) -> Iterable[T]:
        self.epoch_bar = tqdm(iterable, desc=desc, unit=unit)
        return self.epoch_bar

    def wrap_batch_iterator(
        self,
        iterable: Iterable[T],
        desc: str = "Batches",
        unit: str = "batch",
    ) -> Iterable[T]:
        self.batch_bar = tqdm(iterable, desc="  " + desc, leave=False, unit=unit)
        return self.batch_bar


class TelegramTqdmStrategy(TqdmStrategy):
    """Use tqdm for progress bars."""

    def __init__(self, token: str, chat_id: str) -> None:
        super().__init__()

        self.token = token
        self.chat_id = chat_id

    def on_train_begin(
        self,
        num_epochs: int,
        msg: str = "Starting training",
    ) -> None:
        super().on_train_begin(num_epochs, msg)

        msg += f" for {num_epochs} epochs"

        try:
            send_telegram_message(msg, token=self.token, chat_id=self.chat_id)
        except (ValueError, RuntimeError) as e:
            print(f"Failed to send message: {e}")

    def on_train_end(
        self,
        msg: str = "Training completed!",
    ) -> None:
        super().on_train_end(msg)

        try:
            send_telegram_message(msg, token=self.token, chat_id=self.chat_id)
        except (ValueError, RuntimeError) as e:
            print(f"Failed to send message: {e}")

    def wrap_epoch_iterator(
        self,
        iterable: Iterable[T],
        desc: str = "Training",
        unit: str = "epoch",
    ) -> Iterable[T]:
        self.epoch_bar = tqdm_telegram(
            iterable,
            desc=desc,
            unit=unit,
            token=self.token,
            chat_id=self.chat_id,
        )
        return self.epoch_bar

    def wrap_batch_iterator(
        self,
        iterable: Iterable[T],
        desc: str = "Batches",
        unit: str = "batch",
    ) -> Iterable[T]:
        self.batch_bar = tqdm_telegram(
            iterable,
            desc="  " + desc,
            leave=False,
            unit=unit,
            token=self.token,
            chat_id=self.chat_id,
        )
        return self.batch_bar


class CompositeStrategy(VerbosityStrategy):
    """Combine multiple strategies."""

    def __init__(self, *strategies: VerbosityStrategy) -> None:
        self.strategies = list(strategies)

    def on_train_begin(self, num_epochs: int, msg: str = "Starting training") -> None:
        for strategy in self.strategies:
            strategy.on_train_begin(num_epochs, msg)

    def on_train_end(self, msg: str = "Training completed!") -> None:
        for strategy in self.strategies:
            strategy.on_train_end(msg)

    def on_epoch_begin(self, epoch: int) -> None:
        for strategy in self.strategies:
            strategy.on_epoch_begin(epoch)

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float | None = None,
        val_loss: float | None = None,
    ) -> None:
        for strategy in self.strategies:
            strategy.on_epoch_end(epoch, train_loss, val_loss)

    def on_batch_end(
        self,
        batch_idx: int | None = None,
        loss: float | None = None,
    ) -> None:
        for strategy in self.strategies:
            strategy.on_batch_end(batch_idx, loss)

    def wrap_epoch_iterator(
        self,
        iterable: Iterable[T],
        desc: str = "Training",
        unit: str = "epoch",
    ) -> Iterable[T]:
        for strategy in self.strategies:
            iterable = strategy.wrap_epoch_iterator(iterable, desc=desc, unit=unit)
        return iterable

    def wrap_batch_iterator(
        self,
        iterable: Iterable[T],
        desc: str = "Batches",
        unit: str = "batch",
    ) -> Iterable[T]:
        for strategy in self.strategies:
            iterable = strategy.wrap_batch_iterator(iterable, desc=desc, unit=unit)
        return iterable

    def add_strategy(self, strategy: VerbosityStrategy) -> None:
        self.strategies.append(strategy)
