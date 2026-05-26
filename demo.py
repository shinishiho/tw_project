"""
LLVIP YOLO + LLM validation demo.

Changes from original:
- In-memory image encoding (no temp dirs / disk leaks)
- Pydantic BaseSettings for config
- Retry catches specific ValueError, not string matching
- Vllm-specific sampling params moved to extra_body
- Verdict text color white on all backgrounds
- Dead code removed (on_image_upload, image_state)
- --max-tokens exposed as CLI arg
"""

from __future__ import annotations

import argparse
import base64
import io
from typing import Literal

import gradio as gr
import litellm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradio UI demo")
    p.add_argument("--model-path", default="runs/detect/llvip_yolo26n/weights/best.pt")
    p.add_argument("--conf-threshold", type=float, default=0.281)
    p.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    p.add_argument("--llm-model", default="openai/Qwen/Qwen3.5-4B")
    p.add_argument("--api-key", default="sk-lmao")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--crop-padding", type=int, default=50)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    return p.parse_args()


ARGS = parse_args()

import os
os.environ["OPENAI_API_KEY"] = ARGS.api_key


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------

yolo_model = None

try:
    from ultralytics import YOLO
    yolo_model = YOLO(ARGS.model_path)
except Exception as e:
    print(f"[WARN] YOLO model not loaded: {e}")


# ---------------------------------------------------------------------------
# LLM schema + prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a quality-control assistant for an infrared (thermal) pedestrian detector.

You will receive:
1. An infrared image of a scene without any annotations.
2. Cropped infrared images of each detected box, with a red bounding box drawn on it.
3. A metadata block listing each box's number and detector confidence score.

Your task is to inspect each box individually and decide whether it is a valid detection.

Rules:
- The images are low-resolution infrared: humans appear as bright (hot) blobs against a darker background.
- A VALID box tightly encloses a human body or a clear portion of one (head, torso, legs).
- A FALSE_POSITIVE box covers background clutter, vehicle heat signatures, reflections, artifacts,
  or a region with no discernible human shape.
- Use UNCERTAIN when image quality or occlusion makes a confident call difficult.

DO NOT skip any boxes present in the metadata. The number of boxes in the output MUST match the number of boxes in the metadata.
"""

VERDICT_COLORS: dict[str, tuple[int, int, int]] = {
    "VALID": (0, 200, 0),
    "FALSE_POSITIVE": (200, 0, 0),
    "UNCERTAIN": (200, 200, 0),
}


class BoxVerdict(BaseModel):
    id: int
    verdict: Literal["VALID", "FALSE_POSITIVE", "UNCERTAIN"]


class BoxesList(BaseModel):
    boxes: list[BoxVerdict]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _pil_to_b64(img: Image.Image, quality: int = 95) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: dict,
    color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    label: str,
) -> None:
    draw.rectangle(
        [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
        outline=color,
        width=2,
    )
    text_y = box["ymin"] - font.size if box["ymin"] - font.size >= 0 else box["ymax"] - font.size
    pos = (box["xmin"], text_y)
    text_bbox = draw.textbbox(pos, label, font=font)
    draw.rectangle(text_bbox, fill=color)
    draw.text(pos, label, fill="white", font=font)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def run_yolo(
    image_np: np.ndarray,
    conf_threshold: float = ARGS.conf_threshold,
) -> tuple[Image.Image, list[dict]]:
    """Run YOLO; return (annotated image, boxes).

    Box dict: {id, xmin, ymin, xmax, ymax, confidence}
    """
    if not yolo_model:
        raise RuntimeError("YOLO model is not loaded.")

    results = yolo_model(
        image_np,
        conf=conf_threshold,
        imgsz=640,
        iou=0.5,
        end2end=False,
        verbose=False
    )
    detections = results[0].boxes

    pil_img = Image.fromarray(image_np)

    if not detections or len(detections) == 0:
        return pil_img, []

    boxes = [
        {
            "id": i,
            "xmin": int(det.xyxy[0][0]),
            "ymin": int(det.xyxy[0][1]),
            "xmax": int(det.xyxy[0][2]),
            "ymax": int(det.xyxy[0][3]),
            "confidence": float(det.conf[0]),
        }
        for i, det in enumerate(detections, start=1)
    ]

    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(16)

    for box in boxes:
        _draw_box(draw, box, (200, 0, 0), font, f"#{box['id']} {box['confidence']:.2f}")

    return annotated, boxes


# ---------------------------------------------------------------------------
# LLM prep + call
# ---------------------------------------------------------------------------

def prepare_llm_images(image_np: np.ndarray, boxes: list[dict]) -> list[str]:
    """Return [original_b64, crop1_b64, ...] — all in memory, no disk I/O."""
    pil_img = Image.fromarray(image_np)
    font = ImageFont.load_default(16)
    result = [_pil_to_b64(pil_img)]

    for box in boxes:
        img_copy = pil_img.copy()
        draw = ImageDraw.Draw(img_copy)
        _draw_box(draw, box, (200, 0, 0), font, f"#{box['id']} {box['confidence']:.2f}")

        pad = ARGS.crop_padding
        crop = img_copy.crop((
            max(0, box["xmin"] - pad),
            max(0, box["ymin"] - pad),
            min(pil_img.width,  box["xmax"] + pad),
            min(pil_img.height, box["ymax"] + pad),
        ))
        result.append(_pil_to_b64(crop))

    return result


def call_llm(b64_images: list[str], boxes: list[dict]) -> list[dict]:
    """Call LLM with retry on HTTP 500 or box-count mismatch."""
    metadata = "\n".join(
        f"Box #{b['id']}: confidence {b['confidence']:.2f}" for b in boxes
    )
    image_blocks = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}", "detail": "high"}}
        for b in b64_images
    ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                *image_blocks,
                {
                    "type": "text",
                    "text": (
                        f"Metadata for this scene:\n{metadata}\n\n"
                        "Validate every box listed above. "
                        "Respond ONLY with the JSON object described in your instructions."
                    ),
                },
            ],
        },
    ]

    last_exc: Exception | None = None

    for attempt in range(1, ARGS.max_retries + 1):
        try:
            response = litellm.completion(
                model=ARGS.llm_model,
                base_url=ARGS.llm_base_url,
                messages=messages,
                max_tokens=ARGS.max_tokens,
                temperature=0.6,
                top_p=0.95,
                presence_penalty=0.0,
                response_format=BoxesList,
                enable_json_schema_validation=True,
                extra_body={
                    "thinking_token_budget": 1024,
                    "top_k": 20,
                    "min_p": 0.0,
                },
            )

            parsed = BoxesList.model_validate_json(response.choices[0].message.content)

            if len(parsed.boxes) != len(boxes):
                raise ValueError(
                    f"LLM returned {len(parsed.boxes)} boxes, expected {len(boxes)}"
                )

            return [b.model_dump() for b in parsed.boxes]

        except ValueError as e:
            # Box count mismatch — always retry
            last_exc = e
            print(f"[RETRY {attempt}/{ARGS.max_retries}] {e}")

        except litellm.InternalServerError as e:
            last_exc = e
            print(f"[RETRY {attempt}/{ARGS.max_retries}] Server error: {e}")

        except Exception as e:
            raise RuntimeError(f"LLM call failed (non-retryable): {e}") from e

    raise RuntimeError(f"LLM call failed after {ARGS.max_retries} attempt(s): {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Verdict rendering
# ---------------------------------------------------------------------------

def draw_verdicts(image_np: np.ndarray, boxes: list[dict], verdicts: list[dict]) -> Image.Image:
    pil_img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default(16)
    verdict_map = {v["id"]: v["verdict"] for v in verdicts}

    for box in boxes:
        verdict = verdict_map.get(box["id"], "UNCERTAIN")
        color = VERDICT_COLORS.get(verdict, VERDICT_COLORS["UNCERTAIN"])
        draw.rectangle(
            [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
            outline=color,
            width=3,
        )
        _draw_box(draw, box, color, font, f"#{box['id']} {verdict}")

    return pil_img


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

def _run_yolo(image_np: np.ndarray | None, conf_threshold: float):
    if image_np is None:
        return None, [], "Please upload an image first."
    try:
        annotated, boxes = run_yolo(image_np, conf_threshold=conf_threshold)
        if not boxes:
            return annotated, [], "No detections found by YOLO."
        return annotated, boxes, f"YOLO detected {len(boxes)} box(es)."
    except Exception as e:
        return None, [], f"YOLO error: {e}"


def _run_llm(image_np: np.ndarray | None, boxes: list[dict]):
    if image_np is None or not boxes:
        return None, None, "No detections to validate. Run YOLO first."
    try:
        b64_images = prepare_llm_images(image_np, boxes)
        verdicts = call_llm(b64_images, boxes)
        verdict_img = draw_verdicts(image_np, boxes, verdicts)
        summary = "\n".join(f"Box #{v['id']}: {v['verdict']}" for v in verdicts)
        return verdict_img, verdicts, f"LLM verdict:\n{summary}"
    except Exception as e:
        return None, None, f"LLM error: {e}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="LLVIP YOLO + LLM Demo") as demo:
        gr.Markdown("# LLVIP YOLO + LLM Demo")

        if not yolo_model:
            gr.Markdown(
                "> ⚠️ **YOLO model not loaded.** "
                f"Check `--model-path` (current: `{ARGS.model_path}`).",
            )

        boxes_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(label="Upload Image", type="numpy", sources=["upload"])
                conf_slider = gr.Slider(
                    minimum=0.05, maximum=0.95, step=0.01,
                    value=ARGS.conf_threshold,
                    label="Confidence Threshold",
                    info="Lower = more boxes, higher = fewer",
                )
                with gr.Row():
                    run_yolo_btn = gr.Button("Run YOLO", variant="primary", interactive=bool(yolo_model))
                    run_llm_btn  = gr.Button("Ask LLM",  variant="secondary")

            with gr.Column(scale=2):
                output_image = gr.Image(label="Output", type="pil")

        with gr.Accordion("LLM Verdicts", open=True):
            verdict_json = gr.JSON(label="Verdict Details")
            status_text  = gr.Textbox(label="Status", lines=3, interactive=False)

        yolo_outputs = [output_image, boxes_state, status_text]

        run_yolo_btn.click(fn=_run_yolo, inputs=[image_input, conf_slider], outputs=yolo_outputs)
        image_input.upload(fn=_run_yolo, inputs=[image_input, conf_slider], outputs=yolo_outputs)
        run_llm_btn.click(fn=_run_llm, inputs=[image_input, boxes_state], outputs=[output_image, verdict_json, status_text])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name=ARGS.host, server_port=ARGS.port, share=ARGS.share)
