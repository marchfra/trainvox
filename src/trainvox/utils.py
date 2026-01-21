import json
import re
from pathlib import Path
from typing import Any

import requests
from requests.exceptions import HTTPError, RequestException


def _escape_markdown_v2(text: str) -> str:
    """Escape all special MarkdownV2 characters in the text."""
    special_chars = r".#-{}!"

    return re.sub(f"([{re.escape(special_chars)}])", r"\\\1", text)


def send_telegram_message(msg: str, token: str, chat_id: int | str) -> int | None:
    r"""Send a message on Telegram.

    Args:
        msg: The message to send. Can be formatted using MarkdownV2
        token: The token of the Telegram bot
        chat_id: The unique identifier for the target chat

    Returns:
        The message ID.

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
        msg = f"Network error while sending message: {e}"
        raise RuntimeError(msg) from e
    except Exception as e:
        # Any other unexpected error
        msg = f"Unexpected error while sending message: {e}"
        raise RuntimeError(msg) from e

    # Telegram may return 200 but still include an error in JSON
    data = response.json()
    if not data.get("ok", False):
        msg = f"Telegram API error: {data}"
        raise RuntimeError(msg)

    result: dict[str, Any] | None = data.get("result", None)
    if result is not None:
        return result.get("message_id", None)
    return None


def send_telegram_photo(
    photo_path: str | Path,
    token: str,
    chat_id: int | str,
    caption: str | None = None,
) -> int | None:
    r"""Send a local photo on Telegram.

    Args:
        photo_path: The path to an image file on disk
        caption: The caption of the photo. Can be formatted using MarkdownV2
        token: The token of the Telegram bot
        chat_id: The unique identifier for the target chat

    Returns:
        The message ID.

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
        msg = f"Photo file not found: '{photo_path}'"
        raise FileNotFoundError(msg) from e
    except (RequestException, HTTPError) as e:
        # Networking issues, timeouts, connection errors, etc
        msg = f"Network error while sending photo: {e}"
        raise RuntimeError(msg) from e
    except Exception as e:
        msg = f"Unexpected error while sending photo: {e}"
        raise RuntimeError(msg) from e

    # Telegram may return 200 but still include an error in JSON
    data = response.json()
    if not data.get("ok", False):
        msg = f"Telegram API error: {data}"
        raise RuntimeError(msg)

    result: dict[str, Any] | None = data.get("result", None)
    if result is not None:
        return result.get("message_id", None)
    return None


def delete_telegram_message(message_id: int, token: str, chat_id: int | str) -> bool:
    """Delete a message from a Telegram chat.

    Args:
        message_id: The ID of the message to delete
        token: The token of the Telegram bot
        chat_id: The unique identifier for the target chat

    Returns:
        True if the message was successfully deleted, False otherwise.

    Raises:
        RuntimeError: If a network error, HTTP error, or Telegram API error occurs

    """
    url = f"https://api.telegram.org/bot{token}/deleteMessage"

    payload = {
        "message_id": message_id,
        "chat_id": chat_id,
    }

    try:
        response = requests.get(url, params=payload, timeout=10)

        # Raise for 4xx/5xx
        response.raise_for_status()
    except (RequestException, HTTPError) as e:
        # Networking issues, timeouts, DNS errors, etc.
        msg = f"Network error while deleting messages: {e}"
        raise RuntimeError(msg) from e
    except Exception as e:
        # Any other unexpected error
        msg = f"Unexpected error while deleting messages: {e}"
        raise RuntimeError(msg) from e

    # Telegram may return 200 but still include an error in JSON
    data = response.json()
    if not data.get("ok", False):
        msg = f"Telegram API error: {data}"
        raise RuntimeError(msg)

    return data.get("result", False)


def edit_telegram_media(
    photo_path: str | Path,
    message_id: int,
    token: str,
    chat_id: int | str,
    caption: str | None = None,
) -> None:
    r"""Edit the media of an existing Telegram message.

    Args:
        photo_path: The path to the new image file on disk
        message_id: ID of the message to edit
        token: Telegram bot token
        chat_id: Identifier of the chat where the message exists
        caption: New caption for the image, formatted in MarkdownV2

    Raises:
        FileNotFoundError: If the supplied photo doesn't exist
        RuntimeError: If a network, HTTP, or Telegram API error occurs

    """
    url = f"https://api.telegram.org/bot{token}/editMessageMedia"

    photo_path = Path(photo_path)

    # Build media object
    media = {
        "type": "photo",
        "media": "attach://photo",  # Reference for multipart upload
    }

    if caption:
        caption = _escape_markdown_v2(caption)
        media["caption"] = caption
        media["parse_mode"] = "MarkdownV2"

    # The 'media' field must be JSON-encoded string
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "media": json.dumps(media),
    }

    try:
        with photo_path.open("rb") as img:
            files = {"photo": img}
            response = requests.post(url, data=payload, files=files, timeout=20)

        response.raise_for_status()

    except FileNotFoundError as e:
        msg = f"Photo file not found: '{photo_path}'"
        raise FileNotFoundError(msg) from e
    except (RequestException, HTTPError) as e:
        msg = f"Network error while editing message media: {e}"
        raise RuntimeError(msg) from e
    except Exception as e:
        msg = f"Unexpected error while editing message media: {e}"
        raise RuntimeError(msg) from e

    data = response.json()
    if not data.get("ok", False):
        msg = f"Telegram API error: {data}"
        raise RuntimeError(msg)
