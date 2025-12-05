import re
from pathlib import Path

import requests
from requests.exceptions import HTTPError, RequestException


def _escape_markdown_v2(text: str) -> str:
    """Escape all special MarkdownV2 characters in the text."""
    special_chars = r"._*[]()~`>#+-=|{}!"

    return re.sub(f"([{re.escape(special_chars)}])", r"\\\1", text)



def send_telegram_message(msg: str, token: str, chat_id: int | str) -> None:
    r"""Send a message on Telegram.

    Args:
        msg: The message to send. Can be formatted using MarkdownV2
        token: The token of the Telegram bot
        chat_id: The unique identifier for the target chat

    Raises:
        RuntimeError: If a network error, HTTP error, or Telegram API error occurs

    """
    msg = _escape_markdown_v2(msg)

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "parse_mode": "MarkdownV2",
        "text": msg,
    }

    try:
        response = requests.get(url, params=payload, timeout=10)

        # Raise for 4xx/5xx
        response.raise_for_status()
    except (RequestException, HTTPError) as e:
        # Networking issues, timeouts, DNS errors, etc.
        raise RuntimeError(f"Network error while sending message: {e}") from e
    except Exception as e:
        # Any other unexpected error
        raise RuntimeError(f"Unexpected error while sending message: {e}") from e

    # Telegram may return 200 but still include an error in JSON
    data = response.json()
    if not data.get("ok", False):
        raise RuntimeError(f"Telegram API error: {data}")


def send_telegram_photo(
    photo_path: str | Path,
    caption: str | None,
    token: str,
    chat_id: int | str,
) -> None:
    r"""Send a local photo on Telegram.

    Args:
        photo_path: The path to an image file on disk
        caption: The caption of the photo. Can be formatted using MarkdownV2
        token: The token of the Telegram bot
        chat_id: The unique identifier for the target chat

    Raises:
        FileNotFoundError: If the supplied photo doesn't exist
        RuntimeError: If a network error, HTTP error, or Telegram API error occurs

    """
    url = f"https://api.telegram.org/bot{token}/sendPhoto"

    photo_path = Path(photo_path)

    payload = {"chat_id": chat_id}
    if caption:
        caption = _escape_markdown_v2(caption)
        payload["caption"] = caption
        payload["parse_mode"] = "MarkdownV2"

    try:
        with photo_path.open("rb") as img:
            files = {"photo": img}
            response = requests.post(url, data=payload, files=files, timeout=20)

        # Raise for 4xx/5xx
        response.raise_for_status()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Photo file not found: '{photo_path}'") from e
    except (RequestException, HTTPError) as e:
        # Networking issues, timeouts, connection errors, etc
        raise RuntimeError(f"Network error while sending photo: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error while sending photo: {e}") from e

    # Telegram may return 200 but still include an error in JSON
    data = response.json()
    if not data.get("ok", False):
        raise RuntimeError(f"Telegram API error: {data}")
        caption = _escape_markdown_v2(caption)
        caption = _escape_markdown_v2(caption)
