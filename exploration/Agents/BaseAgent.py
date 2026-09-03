import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import is_dataclass, replace
from typing import Any, List, Optional, Tuple

import cv2
from PIL import Image
import numpy as np



class BaseAgent(ABC):
    """
    Common base class for GUI agents.

    This class handles:
    - OpenAI-compatible server URL normalization
    - /v1/models health check
    - served model id discovery
    - shared runtime config construction

    Subclasses only need to implement `_build_client`.
    """

    def __init__(
        self,
        url: str,
        agent_settings: Optional[dict] = None,
        debugger=None,
    ) -> None:
        settings = agent_settings or {}

        self.agent_url = self._normalize_url(url)
        self.history_n = int(settings.get("history_n", 5))
        self.model_id = self.get_model_id()
        self.runtime_conf = dict(settings)
        self.debugger = debugger
        self.resize_factor = settings.get("resize_factor", 1.0)
        self.verbose_print = settings.get("verbose_print", False)

        for key in (
            "history_n",
            "max_tokens",
            "max_context_length",
            "temperature",
            "top_k",
            "top_p",
        ):
            if key in settings:
                self.runtime_conf[key] = settings[key]

        if not self.check_health():
            raise ValueError(
                f"Agent server is not healthy: {self.agent_url}. "
                "Check the server URL and make sure the model is running."
            )
        
        if self.debugger:
            self.debugger.log(f"Agent server is healthy: {self.agent_url}. Model ID: {self.model_id}", depth=1, color="grey")

        self.agent_client = self._build_client()

    @staticmethod
    def _normalize_url(url: str) -> str:
        agent_url = str(url).strip().rstrip("/")
        if not agent_url.endswith("/v1"):
            agent_url = f"{agent_url}/v1"
        return agent_url

    def check_health(self) -> bool:
        try:
            result = subprocess.run(
                ["curl", "-sSf", f"{self.agent_url}/models"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_model_id(self) -> Optional[str]:
        try:
            resp = subprocess.run(
                ["curl", "-sSf", f"{self.agent_url}/models"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )

            data = json.loads(resp.stdout.decode().strip())

            if (
                isinstance(data, dict)
                and isinstance(data.get("data"), list)
                and data["data"]
            ):
                return str(data["data"][0].get("id") or "")

        except Exception:
            pass

        return None

    @abstractmethod
    def _build_client(self):
        """
        Build the real backend client.

        Each subclass decides how to initialize its own model client.
        """
        raise NotImplementedError

    def _resize_img(self, img, f):
        if f == 1.0:
            return img
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_LINEAR)

    def _scale_coord_dict(self, coords, f: float):
        if not coords or f == 1.0:
            return coords
        scaled = {}
        for key, xy in coords.items():
            if xy is not None and len(xy) >= 2:
                scaled[key] = (
                    int(round(float(xy[0]) / f)),
                    int(round(float(xy[1]) / f)),
                )
            else:
                scaled[key] = xy
        return scaled

    def _scale_coords(self, result, f: float):
        """Map agent coords from resize_factor-scaled input back to full screenshot space."""
        if f == 1.0 or not isinstance(result, (tuple, list)) or not result:
            return result

        parsed = result[0]
        if not hasattr(parsed, "orig_coords") and not (
            isinstance(parsed, dict) and "orig_coords" in parsed
        ):
            return result

        orig = getattr(parsed, "orig_coords", None) if not isinstance(parsed, dict) else parsed.get("orig_coords")
        sent = getattr(parsed, "sent_coords", None) if not isinstance(parsed, dict) else parsed.get("sent_coords")

        scaled_orig = self._scale_coord_dict(orig or {}, f)
        scaled_sent = self._scale_coord_dict(sent or {}, f)

        if is_dataclass(parsed):
            parsed = replace(parsed, orig_coords=scaled_orig, sent_coords=scaled_sent)
        elif isinstance(parsed, dict):
            parsed = {**parsed, "orig_coords": scaled_orig, "sent_coords": scaled_sent}
        else:
            parsed.orig_coords = scaled_orig
            parsed.sent_coords = scaled_sent

        return (parsed,) + tuple(result[1:])

    def clear_history(self) -> None:
        """Clear navigation history on the underlying agent client."""
        client = self.agent_client
        if client is None:
            return
        if hasattr(client, "reset") and callable(client.reset):
            client.reset()
        if hasattr(client, "assistant_history"):
            client.assistant_history = []
        if hasattr(client, "messages"):
            client.messages = []

    def step(self, instruction: str, screenshot: np.ndarray) -> Tuple[Any, int, int]:
        f = getattr(self, "resize_factor", 1.0)
        img = self._resize_img(screenshot, f)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.debugger and self.verbose_print:
            with self.debugger.time_block("Agent step: ", color="grey"):
                result, meta = self.agent_client.step(instruction, pil)
                result = self._scale_coords(result, f)
                self.debugger.agent_result(agent_name=self.model_id, instruction=instruction, result=result, meta=meta)
        else:
            result, meta = self.agent_client.step(instruction, pil)
            result = self._scale_coords(result, f)
        return result, meta

    def grounding_action(
        self, action_description: str, screenshot: np.ndarray
    ) -> Tuple[Any, int, int]:
        f = getattr(self, "resize_factor", 1.0)
        img = self._resize_img(screenshot, f)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.debugger and self.verbose_print:
            with self.debugger.time_block("Agent grounding action: ", color="grey"):
                result, meta = self.agent_client.grounding_action(pil, action_description)
                result = self._scale_coords(result, f)
                self.debugger.agent_result(agent_name=self.model_id, instruction=action_description, result=result, meta=meta)
        else:
            result, meta = self.agent_client.grounding_action(pil, action_description)
            result = self._scale_coords(result, f)
        return result, meta

   