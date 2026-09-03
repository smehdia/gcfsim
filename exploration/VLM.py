import os
import re
import cv2
import copy
import json
import math
import textwrap
import time
import base64
import random
import string
import backoff
import logging
import numpy as np
from PIL import Image
from io import BytesIO
import dashscope
import networkx as nx
from dashscope import MultiModalConversation, Generation

def resize_and_encode_to_base64(img, target_size=1344, jpeg_quality=90):
    """
    Resize an image so that its longest side is approximately target_size,
    preserving aspect ratio and rounding dimensions to multiples of 28.

    No square padding is applied. The "pad_value" argument is kept for compatibility but not used.

    Args:
        img (np.ndarray): Input image with shape (H, W, C).
        target_size (int): The desired size for the longest side (default=1344).
        pad_value (int): Not used.

    Returns:
        out_img (np.ndarray): The resized image (no square padding applied).
        scale (float): The scaling factor used for resizing.
        pad (tuple): Tuple (top, bottom, left, right), always (0, 0, 0, 0).
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(round((w * scale)))
    new_h = int(round((h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    return resized, scale, f"data:image/jpeg;base64,{b64}"


def parse_json_from_model_response(text):
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Could not parse model JSON response: empty text")
    # Remove markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    def _balanced_json_slice(s):
        start = next((i for i, ch in enumerate(s) if ch in "{["), -1)
        if start < 0: return None
        stack, in_str, esc = [], False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"': in_str = True; continue
            if ch in "{[": stack.append(ch)
            elif ch in "}]":
                if not stack: return None
                top = stack.pop()
                if (top, ch) not in {("{", "}"), ("[", "]")}:
                    return None
                if not stack:
                    return s[start:i + 1]
        return None

    candidates = [raw]
    balanced = _balanced_json_slice(raw)
    if balanced and balanced != raw:
        candidates.append(balanced)
    candidates.append(re.sub(r",\s*([}\]])", r"\1", raw))
    if balanced:
        candidates.append(re.sub(r",\s*([}\]])", r"\1", balanced))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception as e:
            last_error = e
    raise ValueError(f"Could not parse model JSON response. error={last_error}; preview={raw[:500]!r}")


class VLM:
    def __init__(self, configs, debugger):
        self.total_calls = {}
        self.token_usage_details = {}
        dashscope.api_key = configs.vlm.alibaba_api_key 
        dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
        self.debugger = debugger
        self.configs = configs

    def _extract_token_usage_from_response(self, resp):
        """Return (image_input_tokens, text_input_tokens, output_tokens), defaulting to 0 when unavailable."""
        usage = getattr(resp, "usage", None)
        image_input_tokens = 0
        text_input_tokens = 0
        output_tokens = 0

        if usage is None:
            print("input_tokens_details was not found: resp.usage is None")
            return image_input_tokens, text_input_tokens, output_tokens

        details = getattr(usage, "input_tokens_details", None)
        if details is None:
            print("input_tokens_details was not found")
        elif isinstance(details, dict):
            image_input_tokens = details.get("image_tokens", 0)
            text_input_tokens = details.get("text_tokens", 0)
        else:
            image_input_tokens = getattr(details, "image_tokens", 0)
            text_input_tokens = getattr(details, "text_tokens", 0)

        out_details = getattr(usage, "output_tokens_details", None)
        if out_details is not None:
            if isinstance(out_details, dict):
                output_tokens = out_details.get("text_tokens", 0)
            else:
                output_tokens = getattr(out_details, "text_tokens", 0)

        return image_input_tokens, text_input_tokens, output_tokens

    def _get_message_content_from_response(self, resp):
        output = getattr(resp, "output", None)
        if output is None:
            print(output)
            return None
        choices = getattr(output, "choices", None)
        if not choices:
            print(output)
            return None
        message = getattr(choices[0], "message", None)
        if message is None:
            print(output)
            return None
        return getattr(message, "content", None)

    def extract_elements_from_page(self, screenshot, application_description="", action_history=None):
        """
        Extract actionable UI elements from a screenshot.
        This function uses only the screenshot, not XML layout information.

        Output schema is unchanged:
        {
            "elements": [
                {
                    "type": "icon|button|nav|card|scroll|swipe|input",
                    "bbox": [x1, y1, x2, y2],   
                    "description": "..."
                }
            ]
        }
        """
        history_lines = [str(item).strip() for item in (action_history or []) if str(item or "").strip() and "scroll" not in str(item).lower() and "swipe" not in str(item).lower()]
        history_block = (
            "\n".join(f"- {line}" for line in history_lines)
            if history_lines
            else "- None"
        )

        history_context = (
            "Recent non-scroll/swipe actions that led here "
            "(use only if screenshot is ambiguous; visible foreground always wins):\n"
            f"{history_block}\n"
        )

        resized, scale, data_uri = resize_and_encode_to_base64(screenshot, target_size=self.configs.vlm.screenshot_longest_side_for_element_extraction_model, jpeg_quality=self.configs.vlm.jpeg_quality_for_element_extraction_model)
        app_context_line = (
            f"Application context: {self.configs.app.description}\n"
            if self.configs.app.description
            else "Application context: General-purpose mobile app exploration.\n"
        )
        
        active_surface_rules = (
            "ACTIVE SURFACE / Z-ORDER RULES:\n"
            "Before output, silently choose the single topmost interactive surface receiving taps now.\n"
            "This is a z-order decision, not only a bbox decision.\n"
            "Foreground surfaces include modal, dialog, alert, bottom sheet, action sheet, popup, menu, dropdown, picker, drawer, flyout, expanded FAB menu, or any panel over a dimmed/blurred/darkened/de-emphasized background.\n"
            "If any foreground surface exists, the underlying page is inactive even if its buttons, tabs, cards, toolbar, bottom nav, or text are still visible.\n"
            "Return only controls visually attached to that topmost surface itself: same panel/sheet/dialog/menu/elevation/layer.\n"
            "Never return lower-z background controls behind an overlay, even if visible, important, or overlapping the foreground region.\n"
            "If uncertain whether a candidate is foreground or background, exclude it.\n"
            "Only if no foreground surface exists, use the main page as the active surface.\n"
        )

        schema_rules = (
            "OUTPUT SCHEMA:\n"
            'Return strictly valid JSON exactly as {"elements": [...]} and nothing else.\n'
            "Each element must have exactly: type, bbox, description.\n"
            "type must be one of: icon, button, nav, card, scroll, swipe, input.\n"
            "bbox must be [x1,y1,x2,y2] normalized to 0..1000.\n"
            "description must be a short action sentence with relative position. For icons, mention visual form, e.g. gear icon, avatar, three-dot icon.\n"
            "Do not output active_layer, surface_id, z_order, confidence, rationale, or extra fields.\n"
        )

        extraction_rules = (
            "EXTRACTION ORDER:\n"
            "On the active surface only, extract: (1) top chrome of that surface, (2) scope tabs/chips, (3) remaining app-level actionable controls.\n"
            "Be sensitive to small icons, icon-only buttons, colored icons, and visual symbols, but only if they are on the active surface.\n"
            "False positives from inactive background layers are worse than missing ambiguous elements.\n"
        )

        exclusion_rules = (
            "EXCLUDE:\n"
            "Back/up/close/cancel/dismiss/X controls that only exit or close; ads/promotional/sponsored/recommendation cards; decorative visuals; detailed personal content; keyboard/search/query/chat entry bars; sample prompt chips; emojis; dial/key/num pads; media controls; delete/remove/trash/sign-out/delete-account actions.\n"
            "Goal is app-level navigation/persistent chrome: section tabs, bottom nav, top utility icons, overflow/menu, profile/account, settings/about, global actions.\n"
            "If those controls are behind a modal/sheet/menu/drawer/popup, exclude them.\n"
            "Data lists/dashboards: do not return repeating item rows/cards, per-item toggles, per-row plus/chevron/overflow, inbox rows, alarms, orders, playlists, messages, or feed cards.\n"
            "Settings/preferences exception: if the active surface is a uniform preferences list, include preference rows and their switch/slider/segmented controls.\n"
            "Cards: include only stable app-navigation cards such as category hubs or settings/menu destinations, not feed/content/product/promotional cards unless app context says product cards are primary navigation.\n"
            "Expanded dropdown/accordion/menu: include actionable subitems that change scope/destination; skip informational or duplicate lines.\n"
            "Horizontal scope chips/tabs: include each tappable peer only if the row belongs to the active surface.\n"
            "Utility icons in header/search area: include if tappable and on active surface; exclude if the header/search area is behind an overlay.\n"
        )

        z_order_examples = (
            "Z-ORDER MISTAKES TO AVOID:\n"
            "- Dialog open: do not return app buttons/tabs/cards/toolbar behind it.\n"
            "- Bottom sheet open: do not return bottom nav/list rows behind it.\n"
            "- Overflow menu open: do not return toolbar/page controls behind it.\n"
            "- Dropdown open: do not return controls behind it unless they are part of the dropdown.\n"
            "- Dimmed/greyed/blurred/darkened background means background UI is inactive.\n"
        )

        # ------------------------------------------------------------------
        # System message
        # ------------------------------------------------------------------

        system_msg = {
            "role": "system",
            "content": (
                app_context_line
                + "Extract actionable UI elements from a mobile screenshot for automated app navigation. "
                + "Only output elements on the single topmost active interaction surface. "
                + "Do not output inactive background/lower-z UI. "
                + 'Return strictly valid JSON {"elements": [...]} only.\n\n'
                + active_surface_rules
                + "\n"
                + schema_rules
                + "\n"
                + extraction_rules
                + "\n"
                + exclusion_rules
                + "\n"
                + z_order_examples
            ),
        }

        # ------------------------------------------------------------------
        # User prompt
        # ------------------------------------------------------------------

        prompt = (
            app_context_line
            + history_context
            + 'Return JSON only: {"elements": [...]}.\n'
            + "Keep the schema unchanged: no active_layer, surface_id, z_order, confidence, or explanations.\n\n"

            + "Silently do this before output:\n"
            + "1. Identify the single topmost active interaction surface by z-order, not bbox alone.\n"
            + "2. If a modal/dialog/sheet/menu/popup/dropdown/drawer/picker/overlay is open, the main page behind it is inactive.\n"
            + "3. Reject candidates from the background page even if visible, recognizable, tappable when overlay closes, or spatially overlapping the foreground.\n"
            + "4. Reject dimmed, blurred, darkened, occluded, scrimmed, lower-z, or visually-underlayed controls.\n"
            + "5. Output only actionable controls visually attached to the active surface itself.\n"
            + "6. If foreground/background membership is uncertain, exclude the candidate.\n\n"

            + "For each output element provide only: type, bbox, description.\n"
            + "Allowed types: icon, button, nav, card, scroll, swipe, input.\n"
            + "bbox must be normalized [x1,y1,x2,y2] in 0..1000.\n"
            + "Description must be short and include relative position; for icons mention visual form.\n\n"

            + "Include on active surface: top utility icons, overflow/menu, compose/edit-as-chrome, profile/account/avatar, bottom nav, main section tabs, scope chips/tabs, settings/about/global navigation.\n"
            + "Do not include these if they are behind an overlay.\n\n"

            + "Exclude: back/up/close/cancel/dismiss/X; ads/promos/banners; decorative visuals; search/query/chat/keyboard entry bars; sample prompt chips; dial/key/num pads; media controls; delete/remove/trash/sign-out/delete-account; repeating item rows/cards and their inline toggles/plus/chevrons/overflow.\n"
            + "Settings exception: if the active surface is a uniform preferences/settings list, include preference rows and switches/sliders/segments.\n"
            + "Expanded menus/dropdowns/accordions: include actionable destination/scope subitems only.\n"
            + "For horizontal scope chips/tabs on the active surface, include each tappable peer.\n\n"

            + "Common mistakes: do not return page buttons behind a dialog; do not return bottom nav behind a bottom sheet; do not return toolbar icons behind an overflow menu; do not return controls behind a dropdown; dimmed background is inactive.\n"
        )

        user_content = [
            {"image": data_uri},
            {"text": prompt},
        ]

        user_msg = {
            "role": "user",
            "content": user_content,
        }

        messages = [system_msg, user_msg]
        self.prompt = messages

        resp = MultiModalConversation.call(
            model=self.configs.vlm.model_name_for_elements_extraction,
            messages=messages,
            response_format={"type": "json_object"},
            vl_high_resolution_images=True,
            enable_thinking=False,
            temperature=0.0,
            top_p=1.0,
            seed=42
        )

        image_input_tokens, text_input_tokens, output_tokens = self._extract_token_usage_from_response(resp)

        if self.configs.vlm.model_name_for_elements_extraction in self.token_usage_details.keys():
            self.token_usage_details[self.configs.vlm.model_name_for_elements_extraction]["image_input_tokens"] += image_input_tokens
            self.token_usage_details[self.configs.vlm.model_name_for_elements_extraction]["text_input_tokens"] += text_input_tokens
            self.token_usage_details[self.configs.vlm.model_name_for_elements_extraction]["output_tokens"] += output_tokens
        else:
            self.token_usage_details[self.configs.vlm.model_name_for_elements_extraction] = {
                "image_input_tokens": image_input_tokens,
                "text_input_tokens": text_input_tokens,
                "output_tokens": output_tokens
            }

        raw = self._get_message_content_from_response(resp)
        raw_text = (
            "".join(chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in raw).strip()
            if isinstance(raw, list) else str(raw or "").strip()
        )
        if not raw_text:
            print("VLM returned empty text for element extraction; using empty elements list")
            elements = []
        else:
            elements = (parse_json_from_model_response(raw_text) or {}).get("elements", [])
        H0, W0 = screenshot.shape[:2]
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

        def map_back(b):
            if not (b and len(b) == 4): return None
            try: x1, y1, x2, y2 = map(float, b)
            except: return None
            Hp = int(round(H0 * scale)) + pad_top + pad_bottom
            Wp = int(round(W0 * scale)) + pad_left + pad_right
            x1, x2 = [(v * Wp / 1000.0 - pad_left) / scale for v in (x1, x2)]
            y1, y2 = [(v * Hp / 1000.0 - pad_top) / scale for v in (y1, y2)]
            x1, x2 = sorted([max(0, min(W0 - 1, x1)), max(0, min(W0 - 1, x2))])
            y1, y2 = sorted([max(0, min(H0 - 1, y1)), max(0, min(H0 - 1, y2))])
            return [x1, y1, x2, y2]

        def maybe_add_swipe(items):
            """
            Conditionally add a synthetic 'swipe' control to the items list if it looks like
            a horizontal control strip is present.
            
            Flow:
            1. If "swipe" is already present, do nothing and return.
            2. Collect controls of type nav/button. If less than 2, do nothing.
            3. Group controls into approximate horizontal rows (using vertical center and vertical tolerance).
            4. For each row, find candidates that:
               - are not too high vertically (near the top)
               - are not too "tall" (the row is horizontally oriented)
               - span a wide enough range horizontally
            5. Pick the best row, and synthesize a "swipe" element for it.
            6. Return the list, with the swipe element added if applicable.
            """
            if any(e.get("type") == "swipe" for e in items):
                return items
            controls = [e for e in items if e.get("type") in {"nav", "button"}]
            if len(controls) < 2:
                return items
            # Sort controls by vertical center
            controls.sort(key=lambda e: 0.5 * (e["bbox"][1] + e["bbox"][3]))
            rows, tol = [], 0.05 * H0
            for e in controls:
                cy = 0.5 * (e["bbox"][1] + e["bbox"][3])
                for row in rows:
                    # Group controls in the same horizontal row
                    if abs(cy - row["cy"]) <= tol:
                        row["items"].append(e)
                        row["cy"] = sum(0.5 * (it["bbox"][1] + it["bbox"][3]) for it in row["items"]) / len(row["items"])
                        break
                else:
                    rows.append({"cy": cy, "items": [e]})
            # Try to find the best row that looks like a swipe strip
            best = max(
                (
                    (
                        len(r["items"])
                        + (2 if (0.5 * (min(e["bbox"][1] for e in r["items"]) + max(e["bbox"][3] for e in r["items"])) < 0.35 * H0) else 0)
                        + (1 if (0.5 * (min(e["bbox"][1] for e in r["items"]) + max(e["bbox"][3] for e in r["items"])) < 0.22 * H0) else 0),
                        {
                            "type": "swipe",
                            "bbox": [
                                min(e["bbox"][0] for e in r["items"]),
                                min(e["bbox"][1] for e in r["items"]),
                                max(e["bbox"][2] for e in r["items"]),
                                max(e["bbox"][3] for e in r["items"]),
                            ],
                            "description": "Swipe left or right on this control strip.",
                        }
                    )
                    for r in rows
                    if len(r["items"]) >= 2
                    # Only rows not too low, not too tall, and wide enough
                    and 0.5 * (min(e["bbox"][1] for e in r["items"]) + max(e["bbox"][3] for e in r["items"])) <= 0.82 * H0
                    and (max(e["bbox"][3] for e in r["items"]) - min(e["bbox"][1] for e in r["items"])) <= 0.16 * H0
                    and (max(e["bbox"][2] for e in r["items"]) - min(e["bbox"][0] for e in r["items"])) >= 0.22 * W0
                ),
                key=lambda pair: pair[0],
                default=(None, None),
            )
            if best[1]:
                items.append(best[1])
            return items
       

        def maybe_add_scroll(items):
            """
            Conditionally add a synthetic 'scroll' control to the items list if it looks like
            there is a large vertical area with navigable controls.
            
            Flow:
            1. If "scroll" is already present, do nothing and return.
            2. Collect controls of types nav/button/card/icon. If less than 4 found, do nothing.
            3. If the vertical span of these controls is less than 55% of the screen, do nothing.
            4. Otherwise, generate a scroll bbox that covers most of the vertical region (not full height, clipped a bit at top/bottom).
            5. Only add scroll if the vertical span is wide enough.
            6. Add the "scroll" element and return the list.
            """
            if any(e.get("type") == "scroll" for e in items):
                return items
            c = [e for e in items if e.get("type") in {"nav", "button", "card", "icon"}]
            if len(c) < 4:
                return items
            top, bot = min(e["bbox"][1] for e in c), max(e["bbox"][3] for e in c)
            if (bot - top) < 0.55 * H0:
                return items
            y1, y2 = max(0.10 * H0, top), min(0.96 * H0, bot)
            x1, x2 = 0.03 * W0, 0.97 * W0
            if y2 - y1 < 0.30 * H0:
                return items
            items.append({"type": "scroll", "bbox": [x1, y1, x2, y2], "description": "Scroll vertically to view more content."})
            return items

        mapped_elements = []
        for e in elements:
            if not isinstance(e, dict): continue
            bbox = map_back(e.get("bbox"))
            if not bbox: continue
            desc = str(e.get("description", e.get("action_description", ""))).strip()
            mapped_elements.append({"type": str(e.get("type", "")).strip().lower(), "bbox": bbox, "description": desc})

        mapped_elements = maybe_add_swipe(mapped_elements)
        mapped_elements = maybe_add_scroll(mapped_elements)


        return {"elements": mapped_elements, "image_input_tokens": image_input_tokens, "text_input_tokens": text_input_tokens, "output_tokens": output_tokens}
   

    def extract_page_summary(
        self,
        screenshot,
        application_description="",
        action_history=None,
        interactive_action_descriptions=None,
    ):
        """
        Extract compact structured page identity for screenshot matching.

        Returns:
            {
                "active_tab": str | None,
                "active_subtab": str | None,
                "page_purpose": str,
                "image_input_tokens": int,
                "text_input_tokens": int,
                "output_tokens": int,
            }
        """

        history_lines = [
            str(item).strip()
            for item in (action_history or [])
            if (
                str(item or "").strip()
                and "scroll" not in str(item).lower()
                and "swipe" not in str(item).lower()
            )
        ]

        history_block = (
            "\n".join(f"- {line}" for line in history_lines)
            if history_lines
            else "- None"
        )

        resized, scale, data_uri = resize_and_encode_to_base64(
            screenshot,
            target_size=self.configs.vlm.screenshot_longest_side_for_page_summary_model,
            jpeg_quality=self.configs.vlm.jpeg_quality_for_page_summary_model,
        )

        interactive_actions_json = json.dumps(
            {
                "interactive_actions": [
                    dict(
                        {
                            "description": (
                                str(
                                    item.get("description")
                                    or item.get("action_description")
                                    or ""
                                ).strip()
                                if isinstance(item, dict)
                                else str(item or "").strip()
                            )
                        },
                        **(
                            {
                                "type": str(
                                    item.get("type")
                                    or item.get("vlm_type")
                                    or ""
                                )
                                .strip()
                                .lower()
                            }
                            if (
                                isinstance(item, dict)
                                and str(
                                    item.get("type")
                                    or item.get("vlm_type")
                                    or ""
                                ).strip()
                            )
                            else {}
                        ),
                    )
                    for item in (interactive_action_descriptions or [])
                    if (
                        (
                            isinstance(item, dict)
                            and str(
                                item.get("description")
                                or item.get("action_description")
                                or ""
                            ).strip()
                        )
                        or (
                            not isinstance(item, dict)
                            and str(item or "").strip()
                        )
                    )
                ]
            },
            ensure_ascii=False,
        )

        app_description = (
            str(application_description).strip()
            or str(getattr(self.configs.app, "description", "") or "").strip()
        )

        app_context_line = (
            f"Application context: {app_description}\n"
            if app_description
            else "Application context: General-purpose mobile app exploration.\n"
        )

        system_msg = {
            "role": "system",
            "content": (
                app_context_line
                + "You extract compact page identity information for mobile UI screenshot matching.\n"
                + "Use the screenshot as the primary evidence.\n"
                + "Use recent non-scroll/swipe action history only to disambiguate the selected tab, subtab, route, or page purpose when the screenshot alone is unclear.\n"
                + "Use the provided interactive action descriptions only as supporting evidence.\n"
                + "Describe only the active foreground surface the user can interact with now.\n"
                + "If a modal, dialog, sheet, menu, popup, or scrim-backed overlay is open, treat that overlay as the active surface, not the dimmed page behind it.\n"
                + "Identify active_tab as the currently selected top-level tab or main navigation section, if visible or strongly implied. Use null if unclear.\n"
                + "Identify active_subtab as the selected subtab, filter, segmented control, secondary tab, or active subsection, if visible or strongly implied. Use null if none or unclear.\n"
                + "Write page_purpose as one concise sentence describing what the active screen is used for.\n"
                + "page_purpose should capture the functional purpose of the current screen, not just list visible UI elements.\n"
                + "Do not invent tabs, subtabs, routes, controls, or purposes that are not supported by the screenshot, action history, or provided action list.\n"
                + "Return strictly valid JSON and nothing else using exactly this schema:\n"
                + "{"
                + '"active_tab": string or null, '
                + '"active_subtab": string or null, '
                + '"page_purpose": string'
                + "}.\n"
            ),
        }

        prompt = (
            app_context_line
            + "Recent non-scroll/swipe action descriptions that led to this screen:\n"
            + f"{history_block}\n\n"
            + "Known interactive actions on this node:\n"
            + f"{interactive_actions_json}\n\n"
            + "Analyze the active foreground screen for screenshot matching.\n"
            + "Extract:\n"
            + "1. active_tab: the currently selected main navigation tab or main section, or null if not visible/clear.\n"
            + "2. active_subtab: the selected subtab, filter, segmented control, secondary tab, or subsection, or null if none is visible/clear.\n"
            + "3. page_purpose: one concise sentence describing what this active screen is used for.\n\n"
            + "Rules:\n"
            + "- The screenshot is the main evidence.\n"
            + "- Use action history only when the visual state is ambiguous.\n"
            + "- If a popup/dialog/sheet/menu is open, summarize that foreground layer as the active screen.\n"
            + "- Do not describe hidden or background pages as active.\n"
            + "- Do not invent tab or subtab names if no selected state is visible or strongly implied.\n"
            + "- page_purpose should be useful for matching this screenshot against future screenshots.\n\n"
            + "Return JSON only in this exact format:\n"
            + "{"
            + '"active_tab": null, '
            + '"active_subtab": null, '
            + '"page_purpose": "..."'
            + "}"
        )

        user_msg = {
            "role": "user",
            "content": [
                {"image": data_uri},
                {"text": prompt},
            ],
        }

        messages = [system_msg, user_msg]
        self.prompt = messages

        resp = MultiModalConversation.call(
            model=self.configs.vlm.model_name_for_page_summary,
            messages=messages,
            response_format={"type": "json_object"},
            vl_high_resolution_images=True,
            enable_thinking=False,
            temperature=0.0,
            top_p=1.0,
            seed=42,
        )

        image_input_tokens, text_input_tokens, output_tokens = self._extract_token_usage_from_response(resp)

        model_name = self.configs.vlm.model_name_for_page_summary

        if model_name in self.token_usage_details.keys():
            self.token_usage_details[model_name]["image_input_tokens"] += image_input_tokens
            self.token_usage_details[model_name]["text_input_tokens"] += text_input_tokens
            self.token_usage_details[model_name]["output_tokens"] += output_tokens
        else:
            self.token_usage_details[model_name] = {
                "image_input_tokens": image_input_tokens,
                "text_input_tokens": text_input_tokens,
                "output_tokens": output_tokens,
            }

        raw = self._get_message_content_from_response(resp)

        if isinstance(raw, list):
            raw_text = "".join(
                chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                for chunk in raw
            ).strip()
        else:
            raw_text = str(raw or "").strip()

        content = parse_json_from_model_response(raw_text) or {}

        active_tab = content.get("active_tab")
        active_subtab = content.get("active_subtab")
        page_purpose = str(content.get("page_purpose", "") or "").strip()

        if isinstance(active_tab, str):
            active_tab = active_tab.strip() or None

        if isinstance(active_subtab, str):
            active_subtab = active_subtab.strip() or None

        if not page_purpose:
            page_purpose = "Unknown active screen purpose."

        return {
            "active_tab": active_tab,
            "active_subtab": active_subtab,
            "page_purpose": page_purpose,
            "image_input_tokens": image_input_tokens,
            "text_input_tokens": text_input_tokens,
            "output_tokens": output_tokens,
        }


    def extract_text_embedding(self, text):
        """
        Single-string text embedding (DashScope ``text-embedding-v4``).

        Args:
            text: String (may be None or empty).

        Returns:
            np.ndarray of shape (D,) float32, the embedding vector for the text.
        """
        import numpy as np

        t = "" if text is None else str(text)
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=[t],
        )
        if getattr(resp, "status_code", None) != 200 or not getattr(resp, "output", None):
            raise RuntimeError(
                f"TextEmbedding status={getattr(resp, 'status_code', None)} "
                f"code={getattr(resp, 'code', None)} message={getattr(resp, 'message', None)} "
                f"output={getattr(resp, 'output', None)}"
            )
        out = resp.output
        if not isinstance(out, dict):
            raise KeyError(f"Unexpected TextEmbedding output type: {type(out)!r}")
        embs = out.get("embeddings")
        if not isinstance(embs, list) or len(embs) != 1:
            raise KeyError(
                f"Unexpected TextEmbedding embeddings: len={len(embs) if isinstance(embs, list) else None}, "
                f"expected 1, out={out!r}"
            )
        e = embs[0]
        if not isinstance(e, dict) or "embedding" not in e:
            raise KeyError(f"Unexpected embedding entry: {e!r}")
        arr = np.array(e["embedding"], dtype=np.float32)
        return arr



    def guess_undo_action(self, current_screenshot, parent_screenshot, forward_action):
        """
        Suggest one undo gesture from the current (child) screen back to the parent screen.

        Returns ``(undo_mode, parsed_action)`` where ``parsed_action`` is a ``ParsedAction``
        executable by the driver, or ``None`` when no gesture is needed or none is possible.
        """
        from Agents.MAI_UI.MAI_UI import ParsedAction

        forward_action = forward_action if isinstance(forward_action, dict) else {}
        fwd_type = str(forward_action.get("type", "") or "").strip().lower()
        fwd_desc = str(
            forward_action.get("description")
            or forward_action.get("action_description")
            or ""
        ).strip()

        bbox = forward_action.get("boundingBox") 
        parent_screenshot_vis = parent_screenshot
        if bbox:
            x1, y1, x2, y2 = bbox
            parent_screenshot_vis = cv2.rectangle(parent_screenshot.copy(), (x1, y1), (x2, y2), (0, 255, 0), 5)

        def _map_bbox_on_current(b):
            if not (b and len(b) == 4):
                return None
            try:
                x1, y1, x2, y2 = map(float, b)
            except Exception:
                return None
            Hp = int(round(H0 * scale_cur))
            Wp = int(round(W0 * scale_cur))
            x1, x2 = [(v * Wp / 1000.0) / scale_cur for v in (x1, x2)]
            y1, y2 = [(v * Hp / 1000.0) / scale_cur for v in (y1, y2)]
            x1, x2 = sorted([max(0, min(W0 - 1, x1)), max(0, min(W0 - 1, x2))])
            y1, y2 = sorted([max(0, min(H0 - 1, y1)), max(0, min(H0 - 1, y2))])
            return [x1, y1, x2, y2]

        def _finish(mode, action_type="unknown", *, direction=None, point=None, label=""):
            if mode in ("none", "on_parent", "scroll_unchanged"):
                return mode, None
            orig = {}
            if point is not None:
                orig["point"] = (int(round(point[0])), int(round(point[1])))
            params = {}
            if direction:
                params["direction"] = direction
            text = label or mode
            return mode, ParsedAction(
                raw_text=text,
                thought=text,
                action_type=action_type,
                params=params,
                orig_coords=orig,
            )

        if current_screenshot is None or parent_screenshot is None:
            return "none", None

        H0, W0 = current_screenshot.shape[:2]
        _, scale_cur, uri_cur = resize_and_encode_to_base64(
            current_screenshot, target_size=self.configs.graph.exploration_modes.local_bfs.screenshot_longest_side_for_guess_undo_vlm_model, jpeg_quality=self.configs.graph.exploration_modes.local_bfs.jpeg_quality_for_guess_undo_vlm_model
        )
        _, scale_par, uri_par = resize_and_encode_to_base64(
            parent_screenshot_vis, target_size=self.configs.graph.exploration_modes.local_bfs.screenshot_longest_side_for_guess_undo_vlm_model, jpeg_quality=self.configs.graph.exploration_modes.local_bfs.jpeg_quality_for_guess_undo_vlm_model
        )

        app_context_line = (
            f"Application context: {self.configs.app.description}\n"
            if getattr(self.configs.app, "description", None)
            else "Application context: General-purpose mobile app exploration.\n"
        )

        forward_json = json.dumps(
            {"type": fwd_type or "unknown", "description": fwd_desc or "(none)"},
            ensure_ascii=False,
        )

        system_msg = {
            "role": "system",
            "content": (
                app_context_line
                + "You choose the single best UNDO gesture to return from the CURRENT screen to the PARENT screen.\n"
                + "Image 1 is CURRENT (after the forward action). "
                + "Image 2 is PARENT (target state), with the action bounding box (if any) visualized as a bright green rectangle. "
                + "The forward action that was taken is provided as JSON.\n"
                + "Return strictly valid JSON only.\n\n"
                + "RULES (apply in order; pick the first matching case):\n"
                + "1) If ONLY the main bottom/top navigation bar selection changed between parent and current "
                + "(same page body, only active nav tab differs), undo_mode=main_nav_tab and action is a click on "
                + "the nav item that matches the PARENT screenshot's selected tab (bbox on image 1).\n"
                + "2) If the forward action was scroll down (type scroll/scroll_down) and the screenshots differ "
                + "in scrollable content, undo_mode=scroll_up with action_type=scroll direction=up (bbox optional).\n"
                + "   If the forward action was swipe left and a horizontal strip/carousel changed, undo_mode=swipe_right "
                + "with action_type=scroll direction=right; prefer bbox on the same strip region on image 1.\n"
                + "   If BOTH scroll-down and swipe-left effects are visible, return scroll_up first (only one action).\n"
                + "3) If a menu/accordion/expandable row opened on current vs parent, undo_mode=collapse_menu and "
                + "action is click on the expanded row/header to collapse (bbox on image 1).\n"
                + "4) If a modal/dialog/sheet blocks return to parent, or no safe undo exists, undo_mode=none and action=null.\n"
                + "5) If ONLY a subtab/filter/segment/chip row selection changed (not main nav), undo_mode=tap_subtab "
                + "and action is click the subtab that matches PARENT on image 1.\n"
                + "6) If current already matches parent (same page), undo_mode=on_parent and action=null.\n"
                + "7) Otherwise: tap_back if a back/up/close control is visible on image 1, else system_back.\n\n"
                + "Image 2 always has the forward action's bounding box shown as a bright green rectangle, if the action had a bounding box. Use this as reference for what was acted on in the parent.\n\n"
                + "Schema:\n"
                + '{"undo_mode": string, "reason": string, "action": null or '
                + '{"action_type": "click"|"scroll"|"press_back", '
                + '"direction": "up"|"down"|"left"|"right"|null, '
                + '"bbox": [x1,y1,x2,y2] normalized 0..1000 on IMAGE 1 or null, '
                + '"description": string}}.\n'
                + "undo_mode must be one of: none, on_parent, main_nav_tab, scroll_up, swipe_right, "
                + "collapse_menu, tap_subtab, tap_back, system_back, scroll_unchanged.\n"
                + "For scroll undo use action_type=scroll with direction up or right. For taps use action_type=click.\n"
                + "For system_back use action_type=press_back and bbox=null.\n"
            ),
        }

        user_msg = {
            "role": "user",
            "content": [
                {"image": uri_cur},
                {"image": uri_par},
                {
                    "text": (
                        app_context_line
                        + "Forward action that was taken from PARENT to CURRENT:\n"
                        + f"{forward_json}\n\n"
                        + "Image 1 = CURRENT screen (execute undo here).\n"
                        + "Image 2 = PARENT screen (target after undo, with the forward action visualized as green box if applicable).\n\n"
                        + "Decide the single best undo. Return JSON only."
                    ),
                },
            ],
        }

        model_name = self.configs.graph.exploration_modes.local_bfs.guess_undo_vlm_type

        resp = MultiModalConversation.call(
            model=model_name,
            messages=[system_msg, user_msg],
            response_format={"type": "json_object"},
            vl_high_resolution_images=True,
            enable_thinking=False,
            temperature=0.0,
            top_p=1.0,
            seed=42,
        )

        usage = getattr(resp, "usage", None)
        if usage is not None:
            details = getattr(usage, "input_tokens_details", None) or {}
            image_input_tokens = (
                details.get("image_tokens", 0)
                if isinstance(details, dict)
                else getattr(details, "image_tokens", 0)
            )
            text_input_tokens = (
                details.get("text_tokens", 0)
                if isinstance(details, dict)
                else getattr(details, "text_tokens", 0)
            )
            out_details = getattr(usage, "output_tokens_details", None) or {}
            output_tokens = (
                out_details.get("text_tokens", 0)
                if isinstance(out_details, dict)
                else getattr(out_details, "text_tokens", 0)
            )
            if model_name in self.token_usage_details:
                self.token_usage_details[model_name]["image_input_tokens"] += image_input_tokens
                self.token_usage_details[model_name]["text_input_tokens"] += text_input_tokens
                self.token_usage_details[model_name]["output_tokens"] += output_tokens
            else:
                self.token_usage_details[model_name] = {
                    "image_input_tokens": image_input_tokens,
                    "text_input_tokens": text_input_tokens,
                    "output_tokens": output_tokens,
                }

        raw = self._get_message_content_from_response(resp)
        raw_text = (
            "".join(
                chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                for chunk in raw
            ).strip()
            if isinstance(raw, list)
            else str(raw or "").strip()
        )


        try:
            content = parse_json_from_model_response(raw_text) or {}
        except Exception:
            return "none", None

        undo_mode = str(content.get("undo_mode", "none") or "none").strip().lower()
        reason = str(content.get("reason", "") or "").strip()
        act = content.get("action")

        if undo_mode in ("none", "on_parent", "scroll_unchanged"):
            return undo_mode, None

        if not isinstance(act, dict):
            act = {}

        action_type = str(act.get("action_type", "") or "").strip().lower()
        direction = str(act.get("direction", "") or "").strip().lower() or None
        desc = str(act.get("description", "") or "").strip() or reason or undo_mode
        bbox = _map_bbox_on_current(act.get("bbox"))
        point = None
        if bbox is not None:
            point = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        elif undo_mode == "swipe_right":
            point = _fwd_tap_xy()

        if undo_mode == "scroll_up":
            return _finish(
                undo_mode,
                "scroll",
                direction="up",
                point=point,
                label=f"contextual undo (vlm): {desc}",
            )
        if undo_mode == "swipe_right":
            return _finish(
                undo_mode,
                "scroll",
                direction="right",
                point=point,
                label=f"contextual undo (vlm): {desc}",
            )
        if undo_mode == "system_back":
            return _finish(
                undo_mode,
                "press_back",
                label=f"contextual undo (vlm): {desc}",
            )
        if undo_mode in ("main_nav_tab", "collapse_menu", "tap_subtab", "tap_back"):
            if point is None:
                return "none", None
            return _finish(
                undo_mode,
                "click",
                point=point,
                label=f"contextual undo (vlm): {desc}",
            )

        if action_type == "press_back":
            return _finish("system_back", "press_back", label=desc)
        if action_type == "scroll":
            d = direction if direction in ("up", "down", "left", "right") else "up"
            mode = "scroll_up" if d == "up" else "swipe_right" if d == "right" else undo_mode
            return _finish(mode, "scroll", direction=d, point=point, label=desc)
        if action_type == "click" and point is not None:
            return _finish(undo_mode or "click", "click", point=point, label=desc)

        return "none", None


    def prioritize_node_for_bfs(self, graph, top_k=5, MAX_LIMIT_OF_NODES_FOR_LLM=100):
        import json

        nodes = {}

        def get_depth(graph, node_id):
            g = graph.nx_graph
            tid = str(node_id)
            roots = [nid for nid, n in graph.nodes.items() if getattr(n, "is_root", False)]
            if not roots:
                roots = [nid for nid in g.nodes if g.in_degree(nid) == 0]
            min_depth = None
            for r in roots:
                rs = str(r)
                if rs == tid:
                    return 0
                try:
                    p = nx.shortest_path(g, source=rs, target=tid)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                d = len(p) - 1
                if min_depth is None or d < min_depth:
                    min_depth = d
            return min_depth if min_depth is not None else -1

        for node in graph.nodes.values():
            unexplored_actions = [action for action in node.ui_elements if not action.get("explored")]
            if (len(unexplored_actions) == 0) or (len(node.ui_elements) == 0):
                continue

            unexplored_ratio = len(unexplored_actions) / len(node.ui_elements)

            ui_element_descriptions = [
                {
                    "description": action.get("description", ""),
                    "explored": bool(action.get("explored", False)),
                }
                for action in node.ui_elements
            ]

            nodes[node.node_id] = {
                "page_summary": node.page_summary,
                "ui_element_descriptions": ui_element_descriptions,
                "unexplored_num_actions": len(unexplored_actions),
                "explored_num_actions": len(node.ui_elements) - len(unexplored_actions),
                "unexplored_ratio": unexplored_ratio,
                "num_visits": node.num_visits,
                "depth": get_depth(graph, node.node_id),
            }

        # Only keep up to MAX_LIMIT_OF_NODES_FOR_LLM
        if len(nodes) > MAX_LIMIT_OF_NODES_FOR_LLM:
            sampled_node_ids = random.sample(list(nodes.keys()), MAX_LIMIT_OF_NODES_FOR_LLM)
            nodes = {nid: nodes[nid] for nid in sampled_node_ids}

        # Prepare LLM prompt
        system_prompt = (
            "You are a mobile app graph exploration planner.\n"
            "Your job is to prioritize candidate nodes for BFS-style app exploration.\n"
            "You must rank nodes based on exploration value, navigation cost, and risk.\n"
            "Definitions:\n"
            "- Exploration value is high when a node has many unexplored actions, a high unexplored ratio, and a page purpose likely to reveal new app states.\n"
            "- Broad navigation pages such as Home, Browse, Categories, Menu, Dashboard, Search, and Account overview are usually valuable.\n"
            "Ranking rules:\n"
            "1. Prefer nodes with many unexplored actions.\n"
            "2. Prefer nodes whose unexplored UI elements look likely to reveal new states.\n"
            "3. Prefer nodes with high unexplored_ratio, but do not rank only by ratio.\n"
            "4. Prefer shallow nodes when exploration value is similar.\n"
            "5. Penalize nodes with many visits unless they still have many useful unexplored actions.\n"
            "6. Prefer broad navigation/category/home/dashboard pages.\n"
            "7. Penalize repetitive chat/assistant pages unless they are clearly useful.\n"
            "8. Do not choose a node with zero unexplored actions.\n"
            "9. Return only valid JSON.\n"
            "10. Do not include markdown or extra explanation.\n"
            "Output schema:\n"
            "{\n"
            "  \"ranked_nodes\": [\n"
            "    {\n"
            "      \"node_id\": \"string\",\n"
            "      \"rank\": 1,\n"
            "      \"reason\": \"brief reason\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )

        # Pick a current_node_id, for BFS this can be root or most recently visited
        if hasattr(graph, "current_node_id"):
            current_node_id = graph.current_node_id
        elif hasattr(graph, "last_node_id"):
            current_node_id = graph.last_node_id
        elif graph.nodes:
            current_node_id = next(iter(graph.nodes))
        else:
            current_node_id = ""

        user_prompt = (
            f"Current node:\n{current_node_id}\n\n"
            "Task:\n"
            f"Rank the following candidate nodes for the next BFS exploration step. Only return the top {top_k} ranked nodes in your output.\n\n"
            "Candidate node fields:\n"
            "- page_summary: semantic description of the page\n"
            "- ui_element_descriptions: descriptions of all actionable UI elements and whether each was explored\n"
            "- unexplored_num_actions: number of actions not yet explored\n"
            "- explored_num_actions: number of actions already explored\n"
            "- unexplored_ratio: unexplored_num_actions / total actions\n"
            "- num_visits: how many times this node has been visited\n"
            "- depth: shortest-path depth from root\n\n"
            f"Candidate nodes:\n{json.dumps(nodes, indent=2)}\n\n"
            f"Select the best next node and return only the top {top_k} ranked nodes as a list in the output.\n\n"
            "Remember:\n"
            "- Favor nodes that are likely to reveal new states.\n"
            "- Avoid risky or low-value pages when better alternatives exist.\n"
            "- Balance unexplored actions, depth, visits, and page type.\n"
            "- Each reason: one short sentence (under 12 words).\n"
            "- Return only valid JSON."
        )

        # Call Qwen (dashscope API assumed, but you can change below if needed.)
        # Handle specific dashscope error for invalid URL (code: InvalidParameter, status_code: 400)
        response = dashscope.MultiModalConversation.call(
            model=self.configs.graph.exploration_modes.model_llm_bfs.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            result_format='message',
            temperature=0.1,
            seed=42,
            top_p=0.9,
            max_tokens=max(192, top_k * 96 + 32),
        )

        _, text_input_tokens, output_tokens = self._extract_token_usage_from_response(response)

        if self.configs.vlm.model_name_for_elements_extraction in self.token_usage_details.keys():
            self.token_usage_details[self.configs.vlm.model_name_for_elements_extraction]["text_input_tokens"] += text_input_tokens
            self.token_usage_details[self.configs.vlm.model_name_for_elements_extraction]["output_tokens"] += output_tokens
        else:
            self.token_usage_details[self.configs.vlm.model_name_for_elements_extraction] = {
                "text_input_tokens": text_input_tokens,
                "output_tokens": output_tokens
            }


        raw = response["output"]["choices"][0]["message"]["content"]
        raw_text = (
            "".join(chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in raw).strip()
            if isinstance(raw, list)
            else str(raw or "").strip()
        )
        return parse_json_from_model_response(raw_text)


    def compare_navigtional_purpose(
        self,
        query_page_purpose,
        query_action_history_descriptions,
        candidates,
    ):
        system_prompt = """
    You are a mobile app graph deduplication judge.

    Decide whether the query screen has the same page/navigational purpose as one of the candidate nodes.

    Use:
    - page_purpose as the main signal
    - action_path as supporting context

    Same purpose examples:
    - search results for shoes vs search results for laptops
    - product detail page for item A vs item B
    - filter modal from one search results page vs filter modal from another
    - cart with 1 item vs cart with 3 items

    Different purpose examples:
    - search results page vs product detail page
    - product detail page vs reviews page
    - cart page vs checkout page
    - filter modal vs sort modal
    - normal page vs modal/dialog/drawer

    Be conservative. If no candidate clearly has the same purpose, return null.

    Return only valid JSON with this exact schema:
    {
    "matched_node_id": "node id or null",
    "same_purpose": true,
    "confidence": 0.0,
    "reason": "short reason"
    }
    """.strip()

        user_prompt = json.dumps(
            {
                "query": {
                    "page_purpose": query_page_purpose,
                    "action_path": query_action_history_descriptions,
                },
                "candidates": candidates,
                "rule": "If confidence is below 0.75, return matched_node_id as null and same_purpose as false.",
            },
            ensure_ascii=False,
            indent=2,
        )

        response = dashscope.MultiModalConversation.call(
            model=self.configs.graph.global_localization.find_node_page_purpose_emphasize.compare_navigational_purpose_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            result_format="message",
            temperature=0,
            top_p=1,
            seed=42,
        )

        _, text_input_tokens, output_tokens = self._extract_token_usage_from_response(response)

        usage_key = self.configs.graph.exploration_modes.model_llm_bfs.model_name

        if usage_key in self.token_usage_details:
            self.token_usage_details[usage_key]["text_input_tokens"] += text_input_tokens
            self.token_usage_details[usage_key]["output_tokens"] += output_tokens
        else:
            self.token_usage_details[usage_key] = {
                "text_input_tokens": text_input_tokens,
                "output_tokens": output_tokens,
            }

        raw = response["output"]["choices"][0]["message"]["content"]
        raw_text = (
            "".join(chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in raw).strip()
            if isinstance(raw, list)
            else str(raw or "").strip()
        )

        result = parse_json_from_model_response(raw_text)

        if result.get("matched_node_id") not in candidates:
            result["matched_node_id"] = None
            result["same_purpose"] = False

        conf_thresh = self.configs.graph.global_localization.find_node_page_purpose_emphasize.page_purpose_similarity_threshold
        if float(result.get("confidence", 0.0)) < conf_thresh:
            result["matched_node_id"] = None
            result["same_purpose"] = False

        return result



    def get_node_level_information(self, screenshot, page_summary):
        """
        Compact node-level page description.

        Input:
            screenshot: current UI screenshot
            page_summary: weak page summary string

        Output:
            {
                "page_description": {
                    "high_level": "...",
                    "medium_level": "...",
                    "low_level": "..."
                }
            }
        """

        system_prompt = """
    You describe mobile UI screenshots compactly.

    Given a screenshot and a weak page summary, describe only the visible ACTIVE UI surface.

    If a modal, dialog, popup, bottom sheet, dropdown, picker, menu, search overlay, permission prompt, or focused panel is open, treat it as the active page surface.

    When an overlay/modal is active:
    - Describe only the overlay/modal as the active surface.
    - Do not describe inactive/background page details.
    - You may mention only that the background is dimmed/inactive if visually necessary.

    If no overlay/modal is active, describe the main page normally.

    Return only three description levels:
    - high_level: what this page/surface is mainly for.
    - medium_level: the main regions, sections, and visible page state.
    - low_level: stable visible labels/text/content that identify the page.

    Do not list individual UI elements exhaustively.
    Do not describe bounding boxes, coordinates, ids, or element indexes.
    Do not infer navigation goals.
    Do not describe outgoing edges.
    Do not include confidence, ambiguity, stability, usefulness, or task relevance.
    Return JSON only. Do not add extra fields.
    """.strip()

        user_prompt = f"""
    Weak page summary:
    {page_summary}

    Return exactly this JSON schema:

    {{
    "page_description": {{
        "high_level": "one sentence describing what the active page/surface is for",
        "medium_level": "one compact paragraph describing main regions, sections, layout, and state",
        "low_level": "one or two compact paragraphs of complete listing of stable visible labels/text/content that identify this page"
    }}
    }}
    """.strip()

        max_side = self.configs.post_process.vlm_model_image_size_for_node_level_information
        quality = self.configs.post_process.vlm_model_image_quality_for_node_level_information

        _, _, data_uri = resize_and_encode_to_base64(
            screenshot,
            target_size=max_side,
            jpeg_quality=quality,
        )

        messages = [
            {
                "role": "system",
                "content": [{"text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"text": user_prompt},
                    {"image": data_uri},
                ],
            },
        ]

        model_name = self.configs.post_process.vlm_model_name_for_node_level_information
        max_attempts = int(getattr(self.configs.post_process, "request_retries", 5) or 5)
        last_error = None
        content = None
        text = ""
        for attempt in range(1, max_attempts + 1):
            response = dashscope.MultiModalConversation.call(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                vl_high_resolution_images=True,
                temperature=0.0,
                seed=42,
                top_p=0.9,
            )
            if response.status_code != 200:
                last_error = RuntimeError(
                    f"VLM call failed: status_code={response.status_code}, "
                    f"code={getattr(response, 'code', None)}, "
                    f"message={getattr(response, 'message', None)}"
                )
                if attempt < max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                raise last_error
            try:
                choices = response.output.choices
                if not choices:
                    raise ValueError("VLM response has no choices")
                content = choices[0].message.content
                if isinstance(content, list):
                    text = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    ).strip()
                else:
                    text = str(content or "").strip()
                if not text:
                    raise ValueError(
                        "VLM returned empty text content. "
                        f"raw_content={content!r}"
                    )
                output = parse_json_from_model_response(text)
            except (ValueError, json.JSONDecodeError, AttributeError, IndexError, TypeError) as exc:
                last_error = exc
                if attempt < max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                raise ValueError(
                    "Failed to parse node-level VLM response.\n"
                    f"model={model_name}, attempt={attempt}/{max_attempts}\n"
                    f"raw_content={content!r}\n"
                    f"raw_text={text!r}\n"
                    f"error={exc}"
                ) from exc
            page_description = output.get("page_description", {})
            if not isinstance(page_description, dict):
                page_description = {}
            return {
                "page_description": {
                    "high_level": str(page_description.get("high_level", "") or "").strip(),
                    "medium_level": str(page_description.get("medium_level", "") or "").strip(),
                    "low_level": str(page_description.get("low_level", "") or "").strip(),
                }
            }
        raise last_error or RuntimeError("VLM call failed after retries")


    def get_transition_info(
        self,
        action_crops,
        edge_data,
        source_node_level_information,
        target_node_level_information,
    ):

        system_prompt = """
        You describe UI actions between two mobile UI states.

        You are given:
        - source page description
        - target page description
        - one or more action descriptions
        - optional action crop images
        - normalized action bounding boxes in edge_data, if available

        Your task:
        Create a compact list of action descriptions.

        Each output item represents one action cluster.

        For each cluster:
        - low_level_action_description: visual grounding description of what to find/tap, including appearance and rough location.
        - high_level_action_description: generalized functional action.

        Low-level action description rules:
        - Describe the action target's visual appearance, visible text/icon, component type, and rough location.
        - Use normalized bounding boxes only to infer rough location such as top-left, top-right, center, lower-right, bottom navigation, upper content area, etc.
        - Do not output raw coordinates or bounding boxes.
        - Mention nearby stable visual anchors only if useful for grounding.
        - Prefer stable visual descriptions over dynamic labels, product names, prices, personal names, percentages, or temporary values.
        - If there is only one action, describe it specifically enough for grounding.
        - If multiple similar actions exist, generalize the shared visual pattern and location.

        Rules:
        - If multiple actions are visually/functionally similar, group them into one cluster.
        - If actions differ only by dynamic/specific text, product names, prices, personal details, percentages, or item-specific labels, generalize them.
        - Prefer stable descriptions over dynamic content.
        - Do not describe the target page effect.
        - Do not include confidence, ambiguity, raw coordinates, bounding boxes, node ids, or extra metadata.
        - Return JSON only. Do not add extra fields.
        """.strip()


        user_prompt = f"""
        Source node level information:
        {json.dumps(source_node_level_information, ensure_ascii=False, indent=2)}

        Target node level information:
        {json.dumps(target_node_level_information, ensure_ascii=False, indent=2)}

        Edge/action data:
        {json.dumps(edge_data, ensure_ascii=False, indent=2)}

        Return exactly this JSON schema:

        {{
        "transition_info": [
            {{
            "action_ids": [
                "edge/action id from edge_data"
            ],
            "low_level_action_description": "appearance + rough location description of the action target for grounding",
            "high_level_action_description": "generalized functional action"
            }}
        ]
        }}

        Guidelines:
        - If all actions are equivalent, return one item containing all action_ids.
        - If actions are meaningfully different, return multiple items.
        - For low_level_action_description, include what the target looks like and where it is roughly located.
        - Use boundingBox only to infer rough location; do not output coordinates.
        - Good low-level examples:
        - "Tap the Top Rankings card with a trophy/ranking icon in the upper-right quick-action row."
        - "Tap the pill-shaped Unread filter chip near the top of the message list."
        - "Tap the Messenger tab with the chat bubble icon in the bottom navigation bar."
        - Good high-level examples:
        - "Open the Top Ranking product list."
        - "Filter messages by unread status."
        - "Switch to the Messenger section."
        - Do not include transition_effect.
        - Do not include markdown or extra text.
        """.strip()

        content = [{"text": user_prompt}]

        if isinstance(action_crops, dict):
            crop_items = list(action_crops.items())
        elif isinstance(action_crops, list):
            crop_items = list(enumerate(action_crops))
        else:
            crop_items = []

        max_side = self.configs.post_process.vlm_model_action_crop_size
        quality = self.configs.post_process.vlm_model_image_quality_for_transition_info

        for action_id, crop in crop_items:
            if crop is None:
                continue

            _, _, data_uri = resize_and_encode_to_base64(
                crop,
                target_size=max_side,
                jpeg_quality=quality,
            )

            content.append({"text": f"Action crop for action_id={action_id}"})
            content.append({"image": data_uri})

        messages = [
            {
                "role": "system",
                "content": [{"text": system_prompt}],
            },
            {
                "role": "user",
                "content": content,
            },
        ]

        response = dashscope.MultiModalConversation.call(
            model=self.configs.post_process.vlm_model_name_for_transition_info,
            messages=messages,
            response_format={"type": "json_object"},
            vl_high_resolution_images=True,
            temperature=0.0,
            seed=42,
            top_p=0.9,
        )


        if response.status_code != 200:
            raise RuntimeError(
                f"VLM call failed: status_code={response.status_code}, "
                f"code={getattr(response, 'code', None)}, "
                f"message={getattr(response, 'message', None)}"
            )

        response_content = response.output.choices[0].message.content

        if isinstance(response_content, list):
            text = "".join(
                item.get("text", "")
                for item in response_content
                if isinstance(item, dict)
            )
        else:
            text = response_content

        text = text.strip()

        if text.startswith("```json"):
            text = text.removeprefix("```json").removesuffix("```").strip()
        elif text.startswith("```"):
            text = text.removeprefix("```").removesuffix("```").strip()

        return json.loads(text)


    def get_path_intents(self, trajectory):
        system_prompt = """
        You analyze one mobile UI navigation path.

        The input is an alternating sequence:
        [page description, transition actions, page description, transition actions, page description, ...]

        Your task has only two outputs:
        1. path_user_goals: what user goal(s) this path supports.
        2. path_reliability: how general, stable, and useful this path is.

        Rules for path_user_goals:
        - Write concise user-facing goals.
        - Goals should describe why a user would follow this path.
        - Example: "Navigate to Messenger", "Open the message filter overlay", "Reach the search page".
        - Do not mention node ids, edge ids, screenshots, or metadata.

        Rules for reliability score:
        - Score must be an integer from 0 to 5.
        - 5 = direct, stable, general path aligned with the final page.
        - 4 = good path with minor extra steps.
        - 3 = usable but indirect or somewhat specific.
        - 2 = weak path with unnecessary detours or dynamic/specific actions.
        - 1 = poor path with loops, unrelated steps, or unstable/personal/dynamic content.
        - 0 = invalid or not useful.

        Penalize:
        - loops or back-and-forth navigation
        - repeated switching between sections
        - actions based on personal names, product names, prices, timestamps, badges, or temporary content
        - actions not aligned with the final page
        - unnecessary intermediate pages

        Return JSON only. Do not add extra fields.
        """.strip()

        user_prompt = f"""
        Trajectory:
        {json.dumps(trajectory, ensure_ascii=False, indent=2)}

        Return exactly this JSON schema:

        {{
        "path_user_goals": [
            "concise user goal"
        ],
        "path_reliability": {{
            "score": 0,
            "reason": "brief reason"
        }}
        }}
        """.strip()

        messages = [
            {"role": "system", "content": [{"text": system_prompt}]},
            {"role": "user", "content": [{"text": user_prompt}]},
        ]

        response = dashscope.MultiModalConversation.call(
            model=self.configs.post_process.vlm_model_name_for_path_intents,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            seed=42,
            top_p=0.9,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"VLM call failed: status_code={response.status_code}, "
                f"code={getattr(response, 'code', None)}, "
                f"message={getattr(response, 'message', None)}"
            )

        content = response.output.choices[0].message.content

        if isinstance(content, list):
            text = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )
        else:
            text = content

        text = text.strip()

        if text.startswith("```json"):
            text = text.removeprefix("```json").removesuffix("```").strip()
        elif text.startswith("```"):
            text = text.removeprefix("```").removesuffix("```").strip()

        result = json.loads(text)

        score = int(result.get("path_reliability", {}).get("score", 0))
        score = max(0, min(5, score))

        return {
            "path_user_goals": result.get("path_user_goals", []),
            "path_reliability": {
                "score": score,
                "reason": result.get("path_reliability", {}).get("reason", ""),
            },
        }


    def get_top_paths(self, path_intents, k=3):
        """
        Input:
            path_intents = {
                "0": {
                    "path_user_goals": [...],
                    "path_reliability": {"score": 5, "reason": "..."},
                    "node_sequence": [...]
                },
                ...
            }

        Output:
            ["0", "2", "5"]
        """

        import json
        import dashscope

        if not path_intents:
            return []

        k = max(1, int(k))

        candidates = sorted(
            path_intents.items(),
            key=lambda x: x[1].get("path_reliability", {}).get("score", 0),
            reverse=True,
        )[: max(8, k * 4)]

        compact_candidates = {
            path_id: {
                "path_user_goals": data.get("path_user_goals", []),
                "score": data.get("path_reliability", {}).get("score", 0),
                "reason": data.get("path_reliability", {}).get("reason", ""),
                "node_sequence": data.get("node_sequence", []),
            }
            for path_id, data in candidates
        }

        system_prompt = """
    You select the best navigation paths for one target node.

    Select at least 1 and at most k path ids.

    Prefer:
    - higher reliability score
    - direct and stable paths
    - semantically different user goals
    - different route/node sequences
    - general reusable paths

    Avoid:
    - duplicate or near-duplicate paths
    - paths that only differ by scrolling
    - loops or back-and-forth navigation
    - paths relying on dynamic, personal, product-specific, price-specific, timestamp, badge, or temporary content

    Return JSON only. Do not add extra fields.
    """.strip()

        user_prompt = f"""
    k:
    {k}

    Candidate paths:
    {json.dumps(compact_candidates, ensure_ascii=False, indent=2)}

    Return exactly:

    {{
    "selected_path_ids": [
        "path id"
    ]
    }}
    """.strip()

        response = dashscope.MultiModalConversation.call(
            model="qwen3.5-flash",
            messages=[
                {"role": "system", "content": [{"text": system_prompt}]},
                {"role": "user", "content": [{"text": user_prompt}]},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            seed=42,
            top_p=0.9,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"VLM call failed: status_code={response.status_code}, "
                f"code={getattr(response, 'code', None)}, "
                f"message={getattr(response, 'message', None)}"
            )

        content = response.output.choices[0].message.content

        if isinstance(content, list):
            text = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )
        else:
            text = content

        text = text.strip()

        if text.startswith("```json"):
            text = text.removeprefix("```json").removesuffix("```").strip()
        elif text.startswith("```"):
            text = text.removeprefix("```").removesuffix("```").strip()

        selected = json.loads(text).get("selected_path_ids", [])

        valid_ids = set(path_intents.keys())
        selected = [str(x) for x in selected if str(x) in valid_ids][:k]

        if not selected:
            selected = [candidates[0][0]]

        return selected


    def get_node_user_intents(self, node_info, path_intents, out_edges):
        """
        Output:
            {
                "user_intents": [...]
            }
        """

        system_prompt = """
        You generate natural-language user intents for one mobile UI screen.

        Inputs:
        - node_info: description of the current screen
        - path_intents: goals of paths that reached this screen
        - out_edges: visible actions that lead to other screens/states

        Task:
        Generate possible user intents for the CURRENT screen only.

        Intent style:
        - Write intents like what a user would say to a navigation agent.
        - Use natural commands such as:
        - "Navigate to the page where I can filter messages"
        - "Go to the Messenger filter page"
        - "Open the page where I can search messages or suppliers"
        - "Navigate to Messenger and open message filters"
        - "Go to the page where I can apply conversation filters"

        Rules:
        - Include different abstraction levels:
        - broad page intent
        - section/subsection intent
        - specific current-screen capability
        - Do not write short labels like "Message filtering" or "Custom labels".
        - Do not include goals that belong to out_edges.
        - If an action appears in out_edges, assume its destination goal belongs to another screen/state, not this screen.
        - Do not include dynamic, personal, product-specific, price-specific, badge-specific, timestamp-specific, or temporary details.
        - Keep each intent concise and user-facing.
        - Return JSON only. Do not add extra fields.
        """.strip()

        user_prompt = f"""
        Node info:
        {json.dumps(node_info, ensure_ascii=False, indent=2)}

        Path intents:
        {json.dumps(path_intents, ensure_ascii=False, indent=2)}

        Out edges:
        {json.dumps(out_edges, ensure_ascii=False, indent=2)}

        Return exactly:

        {{
        "user_intents": [
            "natural-language user command"
        ]
        }}

        Good examples:
        - "Navigate to the page where I can filter messages"
        - "Go to the Messenger filter overlay"
        - "Navigate to Messenger and open conversation filters"
        - "Open the page where I can search messages or suppliers"
        - "Go to the page where I can clear or apply message filters"

        Bad examples:
        - "Message filtering"
        - "Unread filter"
        - "Manage labels"
        - "Lena Lu messages"
        - "Unread 19"
        """.strip()

        response = dashscope.MultiModalConversation.call(
            model=self.configs.post_process.vlm_model_name_for_node_user_intents,
            messages=[
                {"role": "system", "content": [{"text": system_prompt}]},
                {"role": "user", "content": [{"text": user_prompt}]},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            seed=42,
            top_p=0.9,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"VLM call failed: status_code={response.status_code}, "
                f"code={getattr(response, 'code', None)}, "
                f"message={getattr(response, 'message', None)}"
            )

        content = response.output.choices[0].message.content

        if isinstance(content, list):
            text = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )
        else:
            text = content

        text = text.strip()

        if text.startswith("```json"):
            text = text.removeprefix("```json").removesuffix("```").strip()
        elif text.startswith("```"):
            text = text.removeprefix("```").removesuffix("```").strip()

        result = json.loads(text)

        return {
            "user_intents": result.get("user_intents", [])
        }

        
    def get_navigation_plan(self, navigation_input):
    
        system_prompt = """
    You generate compact mobile UI navigation plans.

    Input contains:
    - target_page
    - paths; each path has num_pages (L), ordered pages, and L-1 transitions
    - each transition has alternative_actions: parallel ways to move from page[i] to page[i+1]
    - alternative_actions on one transition are OR-options, NOT sequential steps

    Task:
    Create one navigation plan for each path.

    Each plan must contain:
    - relevant_waypoint_sequence: semantic page/state names the agent lands on
    - transition_hints: ordered action hints needed to reach the target

    Each transition_hints item must contain:
    - high_level: the intended action
    - low_level: how to visually locate/perform the action

    Rules:
    - Do not generate user intents.
    - Do not include node ids, edge ids, coordinates, screenshots, scores, or reasons.
    - Use page descriptions to name waypoints.
    - len(relevant_waypoint_sequence) <= L - 1 and len(transition_hints) <= L - 1.
    - Produce exactly one transition_hint per path transition (one per hop), never one per alternative_action.
    - For a transition with multiple alternative_actions, pick the single best action for reaching target_page, OR merge suitable options as one hint (e.g. "Tap X / Tap Y"), OR drop alternatives that are noise or misaligned with target_page and the destination page description.
    - Scroll or minor in-page adjustments may share a waypoint with the previous page; do not add extra waypoints for them.
    - Use high_level_action_description for high_level and low_level_action_description for low_level.
    - low_level must describe stable visual anchors: control label, icon role, UI region, approximate location.
    - low_level must NOT contain person names, profile names, company names, private identifiers, or dynamic personal details.
    - Replace person-specific descriptions with generic terms like "profile card", "post author row", "user avatar", "feed post", or "profile entry".
    - Avoid fragile dynamic details such as exact counts, timestamps, prices, notification badges, or temporary text unless essential for grounding.
    - Remove duplicate consecutive waypoints.
    - Do not include app launch, phone home screen, or OS-level states.
    - Do not include actions after reaching the target page.
    - Return JSON only. Do not add extra fields.
    """.strip()

        user_prompt = f"""
    Navigation input:
    {json.dumps(navigation_input, ensure_ascii=False, indent=2)}

    Return exactly:

    {{
    "ui_navigation_memory": [
        {{
        "relevant_waypoint_sequence": [
            "semantic waypoint name"
        ],
        "transition_hints": [
            {{
            "high_level": "general intended action",
            "low_level": "stable visual grounding hint without person names"
            }}
        ]
        }}
    ]
    }}

    Important:
    - transition_hints must be a list of objects, not strings.
    - One transition_hint per input transition; never expand parallel alternative_actions into multiple sequential hints.
    - relevant_waypoint_sequence and transition_hints must each have at most L - 1 items (L = num_pages).
    - high_level should be short and functional.
    - low_level should help the agent find the control on screen.
    - low_level must not include person names or private/dynamic identifiers.
    - Do not include markdown or extra text.
    """.strip()

        response = dashscope.MultiModalConversation.call(
            model=self.configs.post_process.vlm_model_name_for_navigation_plan,
            messages=[
                {"role": "system", "content": [{"text": system_prompt}]},
                {"role": "user", "content": [{"text": user_prompt}]},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            seed=42,
            top_p=0.9,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"VLM call failed: status_code={response.status_code}, "
                f"code={getattr(response, 'code', None)}, "
                f"message={getattr(response, 'message', None)}"
            )

        content = response.output.choices[0].message.content

        if isinstance(content, list):
            text = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )
        else:
            text = content

        text = text.strip()

        if text.startswith("```json"):
            text = text.removeprefix("```json").removesuffix("```").strip()
        elif text.startswith("```"):
            text = text.removeprefix("```").removesuffix("```").strip()

        return json.loads(text)

    def get_agent_thought(self, node_1_info, edge_info, node2_info, user_intents_for_the_node):
        system_prompt = """
    You generate short GUI Agent-style thinking text for mobile UI navigation training in first person.

    Input contains:
    - current_node: full semantic description of the current screen/page
    - edge: action information for moving from current_node to next_node
    - next_node: full semantic description of the next screen/page reached after the edge action
    - user_intents: user goals associated with the destination/target node

    Task:
    For each user intent, generate exactly one short internal navigation thought explaining why the edge action is useful for that user intent.

    The thinking will later be wrapped as:
    <thinking>...</thinking>

    Rules:
    - Generate thoughts only. Do not generate tool calls, coordinates, actions, JSON tool arguments, or mobile_use calls.
    - Do not include <thinking> or </thinking> tags.
    - Output one thought per user intent, in exactly the same order as the input user_intents list.
    - Do not repeat the user intent in the output.
    - Use a natural internal navigation-reasoning tone, as if the agent is deciding its next local step.
    - The wording may vary naturally. It does not always need to start with "I need to" or "I should".
    - Good tone examples:
    - "I should open the account area to reach the dashboard-related options."
    - "Opening the You tab gets me closer to the saved lists section."
    - "The account area is the right next step for checking gift card details."
    - "Scrolling down should reveal the lower dashboard sections related to payments."
    - Do not mention node ids, edge ids, graph, trajectory, path, coordinates, screenshots, or JSON.
    - Do not describe the full route.
    - Do not invent pages, controls, or content not supported by the input.
    - Use the user intent to decide which part of current_node, edge, or next_node is relevant.
    - The sentence should explain the local next step, not the whole task.
    - If the edge is a click/tap, explain why using that control helps.
    - If the edge is a scroll/swipe, explain what relevant content the scroll is expected to reveal.
    - Do not say the action reaches the final target directly unless next_node clearly contains the requested target content.
    - If next_node is only intermediate, describe it as a useful waypoint, entry point, account area, section, or gateway.
    - Keep each thinking sentence concise: ideally 12-30 words.
    - Avoid uncertain wording like "maybe", "probably", "I think", or "it seems".
    - Avoid private/person-specific details from the UI. Replace them with generic terms such as "account area", "profile section", "saved list", or "dashboard section".
    - Return JSON only. Do not add extra fields.
    """.strip()

        intent_count = len(user_intents_for_the_node or [])
        example_thoughts = ",\n        ".join(
            f'"thinking sentence for user_intents[{i}]"' for i in range(intent_count)
        )
        example_json = (
            '{\n    "agent_thoughts": [\n        '
            + example_thoughts
            + "\n    ]\n}"
        )

        user_prompt = f"""
    Current node:
    {json.dumps(node_1_info, ensure_ascii=False, indent=2)}

    Edge action:
    {json.dumps(edge_info, ensure_ascii=False, indent=2)}

    Next node:
    {json.dumps(node2_info, ensure_ascii=False, indent=2)}

    User intents, in order:
    {json.dumps(user_intents_for_the_node, ensure_ascii=False, indent=2)}

    Return exactly:

    {example_json}

    Important:
    - agent_thoughts must be a list of strings.
    - The list length must equal the number of user_intents.
    - The order must match the input user_intents order exactly.
    - Do not include user_intent fields.
    - Do not include <thinking> tags.
    - Do not include tool_call, coordinates, node ids, graph terms, or full route descriptions.
    - Do not include markdown or extra text.
    """.strip()

        messages = [
            {"role": "system", "content": [{"text": system_prompt}]},
            {"role": "user", "content": [{"text": user_prompt}]},
        ]

        model_name = getattr(
            self.configs.post_process,
            "vlm_model_name_for_agent_thought",
        )
        max_attempts = int(getattr(self.configs.post_process, "request_retries", 5) or 5)
        last_error = None
        response_content = None
        text = ""
        expected_count = len(user_intents_for_the_node or [])

        for attempt in range(1, max_attempts + 1):
            response = dashscope.MultiModalConversation.call(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                seed=42,
                top_p=0.9,
            )

            if response.status_code != 200:
                last_error = RuntimeError(
                    f"VLM call failed: status_code={response.status_code}, "
                    f"code={getattr(response, 'code', None)}, "
                    f"message={getattr(response, 'message', None)}"
                )
                if attempt < max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                raise last_error

            try:
                choices = response.output.choices
                if not choices:
                    raise ValueError("VLM response has no choices")

                response_content = choices[0].message.content

                if isinstance(response_content, list):
                    text = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in response_content
                    ).strip()
                else:
                    text = str(response_content or "").strip()

                if not text:
                    raise ValueError(
                        "VLM returned empty text content. "
                        f"raw_content={response_content!r}"
                    )

                parsed = parse_json_from_model_response(text)

                if not isinstance(parsed, dict):
                    raise ValueError(f"Parsed response must be a dict, got {type(parsed)}")

                thoughts = parsed.get("agent_thoughts")

                if not isinstance(thoughts, list):
                    raise ValueError("Response must contain agent_thoughts as a list")

                if len(thoughts) != expected_count:
                    raise ValueError(
                        "agent_thoughts length mismatch. "
                        f"expected={expected_count}, got={len(thoughts)}"
                    )

                for idx, thinking in enumerate(thoughts):
                    if not isinstance(thinking, str) or not thinking.strip():
                        raise ValueError(
                            f"Invalid thinking at index {idx}: {thinking!r}"
                        )

                return parsed

            except (ValueError, json.JSONDecodeError, AttributeError, IndexError, TypeError) as exc:
                last_error = exc

                if attempt < max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue

                raise ValueError(
                    "Failed to parse agent-thought VLM response.\n"
                    f"model={model_name}, attempt={attempt}/{max_attempts}\n"
                    f"raw_content={response_content!r}\n"
                    f"raw_text={text!r}\n"
                    f"error={exc}"
                ) from exc

        raise last_error or RuntimeError("VLM call failed after retries")