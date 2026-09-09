# src/models/florence_engine.py
import logging

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from config import FLORENCE_MODEL_ID, FLORENCE_DEVICE, FLORENCE_ATTN_IMPLEMENTATION

logger = logging.getLogger(__name__)

_model = None
_processor = None


def _load():
    """Load Florence-2 once. Never call this per-request, it is slow."""
    global _model, _processor
    if _model is not None:
        return

    _model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_ID,
        trust_remote_code=True,
        attn_implementation=FLORENCE_ATTN_IMPLEMENTATION,
        torch_dtype=torch.float32 if FLORENCE_DEVICE == "cpu" else torch.float16,
    ).to(FLORENCE_DEVICE)

    _processor = AutoProcessor.from_pretrained(
        FLORENCE_MODEL_ID, trust_remote_code=True
    )


def _run_task(image: Image.Image, task_prompt: str, text_input: str = None) -> dict:
    _load()
    prompt = task_prompt if text_input is None else task_prompt + text_input

    inputs = _processor(text=prompt, images=image, return_tensors="pt").to(FLORENCE_DEVICE)

    generated_ids = _model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False,
    )
    generated_text = _processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed = _processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed


def _safe_run_task(image: Image.Image, task_prompt: str, text_input: str = None, default=None) -> dict:
    """
    Guarded entry point for every Florence-2 task. Rejects None/zero-size
    images up front and swallows any model failure (bad weights, OOM,
    corrupted image data surfacing during processing, etc.), logging instead
    of raising, so one bad page can't crash an ingestion batch.
    """
    if default is None:
        default = {}

    if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        logger.warning("Skipping Florence-2 task %s: invalid or empty image", task_prompt)
        return default

    try:
        return _run_task(image, task_prompt, text_input)
    except Exception:
        logger.exception("Florence-2 task %s failed", task_prompt)
        return default


def caption_image(image: Image.Image) -> str:
    """One sentence describing the page. Good for a quick index entry."""
    result = _safe_run_task(image, "<MORE_DETAILED_CAPTION>")
    return result.get("<MORE_DETAILED_CAPTION>", "")


def ocr_page(image: Image.Image) -> str:
    """Plain OCR text for the whole page, no region breakdown."""
    result = _safe_run_task(image, "<OCR>")
    return result.get("<OCR>", "")


def extract_text(image: Image.Image) -> dict:
    """
    OCR text plus rectangular bounding boxes, normalized for the citation UI.
    Returns {"text": str, "boxes": [{"text": str, "bbox": [x1, y1, x2, y2]}, ...]}.
    """
    result = _safe_run_task(image, "<OCR_WITH_REGION>")
    region_result = result.get("<OCR_WITH_REGION>", {})
    quad_boxes = region_result.get("quad_boxes", [])
    labels = region_result.get("labels", [])

    boxes = []
    for quad, label in zip(quad_boxes, labels):
        xs = quad[0::2]
        ys = quad[1::2]
        boxes.append({"text": label, "bbox": [min(xs), min(ys), max(xs), max(ys)]})

    return {"text": " ".join(labels).strip(), "boxes": boxes}


def ocr_with_regions(image: Image.Image) -> dict:
    """OCR text plus bounding boxes, for when you need to point at a specific spot."""
    result = _safe_run_task(image, "<OCR_WITH_REGION>")
    return result.get("<OCR_WITH_REGION>", {})


def detect_regions(image: Image.Image) -> list:
    """Object detection, useful for charts and diagrams with distinct visual elements."""
    result = _safe_run_task(image, "<OD>")
    od = result.get("<OD>", {})
    bboxes = od.get("bboxes", [])
    labels = od.get("labels", [])
    return [{"label": label, "bbox": bbox} for bbox, label in zip(bboxes, labels)]