# src/models/ollama_client.py
import base64
import io

import ollama
from PIL import Image

from config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_FAST_MODEL

_client = ollama.Client(host=OLLAMA_HOST)

OLLAMA_NOT_RUNNING_MESSAGE = "Ollama server not found. Run `ollama serve` and try again."


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server can't be reached."""


def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8") # A = 65


def ask(prompt: str, image_b64: str = None, model: str = None) -> str:
    """
    Send a prompt, with an optional base64-encoded image, to the local Ollama
    server. Returns the plain text response.

    Raises OllamaUnavailableError with a clear message if the server isn't running.
    """
    model = model or OLLAMA_MODEL
    message = {"role": "user", "content": prompt}
    if image_b64 is not None:
        message["images"] = [image_b64]

    try:
        response = _client.chat(model=model, messages=[message])
    except ConnectionError as e:
        raise OllamaUnavailableError(OLLAMA_NOT_RUNNING_MESSAGE) from e

    return response["message"]["content"]


def ask_about_image(image: Image.Image, question: str, model: str = None) -> str:
    """Send one page image and one question to the local VLM. Returns plain text."""
    return ask(question, image_b64=_image_to_base64(image), model=model)


def ask_text_only(question: str, model: str = None) -> str:
    """For reasoning steps that do not need an image, e.g. summarizing retrieved captions."""
    return ask(question, model=model or OLLAMA_FAST_MODEL)


def is_ollama_running() -> bool:
    """Quick health check before you build anything on top of this."""
    try:
        _client.list()
        return True
    except Exception:
        return False