import requests


def send_telegram_message(msg: str, token: str, chat_id: str) -> None:
    """Send a message on Telegram.

    The message can be formatted using Markdown.
    """
    payload = {
        "chat_id": chat_id,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
        "text": msg,
    }

    r = requests.get(
        f"https://api.telegram.org/bot{token}/sendMessage",
        params=payload,
        timeout=10,
    )
    r.raise_for_status()
