from pathlib import Path

import requests


def send_telegram_message(msg: str, token: str, chat_id: int | str) -> None:
    """Send a message on Telegram.

    Args:
        msg: The message to send. Can be formatted using MarkdownV2
        token: The token of the Telegram bot
        chat_id: The unique identifier for the target chat

    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": True,
        "text": msg,
    }

    r = requests.get(url, params=payload, timeout=20)
    r.raise_for_status()


def send_telegram_photo(
    photo_path: str | Path,
    caption: str | None,
    token: str,
    chat_id: int | str,
) -> None:
    """Send a local photo on Telegram.

    Args:
        photo_path: The path to an image file on disk
        caption: The caption of the photo. Can be formatted using MarkdownV2
        token: The token of the Telegram bot
        chat_id: The unique identifier for the target chat

    """
    url = f"https://api.telegram.org/bot{token}/sendPhoto"

    photo_path = Path(photo_path)

    payload = {
        "chat_id": chat_id,
        "parse_mode": "MarkdownV2",
    }
    if caption:
        payload["caption"] = caption

    with photo_path.open("rb") as img:
        files = {"photo": img}
        r = requests.post(url, data=payload, files=files, timeout=20)

    r.raise_for_status()
