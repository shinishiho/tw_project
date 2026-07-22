"""Pinned LocateAnything worker adapted from NVIDIA's official model card."""

from __future__ import annotations

from typing import Any


class LocateAnythingWorker:
    """Load LocateAnything once and serve hybrid grounding queries."""

    def __init__(self, model_id: str, revision: str, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self.torch = torch
        self.device = device
        self.dtype = torch.bfloat16
        common = {
            "revision": revision,
            "trust_remote_code": True,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
        self.processor = AutoProcessor.from_pretrained(model_id, **common)
        self.model = (
            AutoModel.from_pretrained(model_id, torch_dtype=self.dtype, **common)
            .to(device)
            .eval()
        )

    def predict(
        self,
        image: Any,
        question: str,
        *,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)
        with self.torch.inference_mode():
            response = self.model.generate(
                pixel_values=inputs["pixel_values"].to(self.dtype),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=False,
            )
        result = {"answer": response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result["stats"] = response[2]
        return result
