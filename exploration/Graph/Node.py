import os

import cv2
import numpy as np




def elements_to_canonical_page_layout(elements_payload, *, header="page layout:"):
    """
    Drop ``image`` and ``card`` elements; sort the rest by coarse location (top→center→…)
    then by bbox center. Return a single string:

        page layout:
        icon, Search, top, Tap the magnifying glass...
        nav, Home, bottom, Open the Home tab...

    Each line is: ``type``, ``label``, ``location``, ``action_description`` (comma-separated,
    fields quoted if they contain commas or quotes).
    """
    _LOCATION_READ_ORDER = ("top", "center", "bottom", "left", "right")

    def _canonical_csv_field(s):
        t = (s if s is not None else "").replace("\n", " ").replace("\r", " ").strip()
        if not t:
            return ""
        if any(c in t for c in ',"\n\r'):
            return '"' + t.replace('"', '""') + '"'
        return t

    def _canonical_layout_sort_key(el):
        loc = str(el.get("location", "")).strip().lower()
        try:
            lo = _LOCATION_READ_ORDER.index(loc)
        except ValueError:
            lo = len(_LOCATION_READ_ORDER)
        bb = el.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) >= 4):
            bb = el.get("boundingBox")
        if isinstance(bb, (list, tuple)) and len(bb) >= 4:
            cy = (float(bb[1]) + float(bb[3])) / 2.0
            cx = (float(bb[0]) + float(bb[2])) / 2.0
        else:
            cy, cx = float("inf"), float("inf")
        return lo, cy, cx

    if isinstance(elements_payload, list):
        raw_items = elements_payload
    elif isinstance(elements_payload, dict):
        raw_items = elements_payload.get("elements", [])
    else:
        raw_items = []
    items = [
        e
        for e in raw_items
        if isinstance(e, dict)
        and str(e.get("type", "")).strip().lower() not in ("image", "card")
    ]
    items.sort(key=_canonical_layout_sort_key)
    lines = []
    for e in items:
        typ = str(e.get("type", "")).strip().lower()
        label = _canonical_csv_field(
            e.get("label")
            or e.get("text")
            or e.get("description")
            or e.get("action_description")
            or ""
        )
        loc = _canonical_csv_field(e.get("location", ""))
        desc = _canonical_csv_field(
            e.get("action_description")
            or e.get("description")
            or e.get("text")
            or e.get("label")
            or ""
        )
        lines.append(f"{typ}, {label}, {loc}, {desc}")
    body = "\n".join(lines)

    h = (header or "").strip()
    if h:
        return f"{h}\n{body}" if body else f"{h}\n"
    return body


class Node:
    def __init__(self, **kwargs):

        if "clip_embedding" in kwargs:
            self.clip_embedding = kwargs["clip_embedding"]
        else:
            raise ValueError("clip_embedding not found in kwargs")

        if "screenshot" in kwargs:
            screenshot = kwargs["screenshot"]
        else:
            raise ValueError("screenshot not found in kwargs")

        if "node_id" in kwargs:
            self.node_id = kwargs["node_id"]
        else:
            raise ValueError("node_id not found in kwargs")

        if "is_root" in kwargs:
            self.is_root = kwargs["is_root"]
        else:
            raise ValueError("is_root not found in kwargs (specify this node is root node or not)")

        if "node_history" in kwargs:
            node_history = kwargs["node_history"]
        else:
            raise ValueError("node_history not found in kwargs")

        if "action_history" in kwargs:
            action_history = kwargs["action_history"]
        else:
            raise ValueError("action_history not found in kwargs")

        if "screen_type" in kwargs:
            self.screen_type = kwargs["screen_type"]
        else:
            raise ValueError("screen_type not found in kwargs")

        if "xml_layout" in kwargs:
            self.xml_layout = kwargs["xml_layout"]
        else:
            raise ValueError("xml_layout not found in kwargs")

        if "configs" in kwargs:
            configs = kwargs["configs"]
        else:
            raise ValueError("configs not found in kwargs")

        if "debugger" in kwargs:
            debugger = kwargs["debugger"]
        else:
            raise ValueError("debugger not found in kwargs")

        if "vlm_client" not in kwargs:
            raise ValueError("vlm_client not found in kwargs")
        else:
            vlm_client = kwargs["vlm_client"]

        if "image_utils" not in kwargs:
            raise ValueError("image_utils not found in kwargs")
        else:
            image_utils = kwargs["image_utils"]

        if "page_summary_payload" in kwargs:
            page_summary_payload = kwargs["page_summary_payload"]
            self.page_summary = "Active Tab: {}, Active Subtab: {}, Page Purpose: {}".format(page_summary_payload.get("active_tab", ""), page_summary_payload.get("active_subtab", ""), page_summary_payload.get("page_purpose", ""))
            self.page_purpose = page_summary_payload.get("page_purpose", "")
            self.active_tab = page_summary_payload.get("active_tab", "")
            self.active_subtab = page_summary_payload.get("active_subtab", "")
            self.page_purpose_embedding = image_utils.text_model_embedder.encode(self.page_purpose, max_length=1024)['dense_vecs']
        else:
            self.page_summary = None
            self.page_purpose = None
            self.active_tab = None
            self.active_subtab = None
            self.page_purpose_embedding = None
        
        debugger.subrule("Creating Node {}".format(self.node_id), color="blue", symbol="=")

        self.backtracking_action = None   # useful to save undo action for backtracking
        
        self.num_visits = 1
        self.num_refreshed_elements = 0
        screenshots_dir = os.path.join(configs.logs.root, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        full_path = os.path.join(screenshots_dir, self.node_id + ".jpg")
        cv2.imwrite(full_path, screenshot, [cv2.IMWRITE_JPEG_QUALITY, configs.screenshots.jpeg_quality])
        self.active_bbox = [
            0,
            configs.screenshots.margin_from_top_screen_to_ignore,
            screenshot.shape[1],
            screenshot.shape[0],
        ]
        cv2.imwrite(
            os.path.join(configs.logs.root, "current_node_screenshot.jpg"),
            screenshot,
            [cv2.IMWRITE_JPEG_QUALITY, configs.screenshots.jpeg_quality],
        )

        with debugger.time_block("Extracting UI Elements using VLM", color="gray"):
            (
                self.ui_elements,
                draw,
                image_input_tokens_element_extraction,
                text_input_tokens_element_extraction,
                output_tokens_element_extraction,
            ) = self.extract_ui_elements_vlm(
                screenshot, vlm_client, configs, action_history=action_history
            )
            self.canonical_page_layout = elements_to_canonical_page_layout(self.ui_elements)

        with debugger.time_block("Add clip embedding to each element", color="gray"):
            self.ui_elements = self.add_clip_embeddings(image_utils, self.ui_elements, screenshot)

        if self.page_summary is None:
            with debugger.time_block("Extracting Page Summary", color="gray"):
                page_summary_payload = vlm_client.extract_page_summary(
                    screenshot,
                    application_description=configs.app.description,
                    action_history=action_history,
                    interactive_action_descriptions=self.ui_elements,
                )

                self.page_summary = "Active Tab: {}, Active Subtab: {}, Page Purpose: {}".format(page_summary_payload.get("active_tab", ""), page_summary_payload.get("active_subtab", ""), page_summary_payload.get("page_purpose", ""))
                self.page_purpose = page_summary_payload.get("page_purpose", "")
                self.active_tab = page_summary_payload.get("active_tab", "")
                self.active_subtab = page_summary_payload.get("active_subtab", "")
                self.page_purpose_embedding = image_utils.text_model_embedder.encode(self.page_purpose, max_length=1024)['dense_vecs']
                
                image_input_tokens_page_summary = page_summary_payload.get("image_input_tokens", 0)
                text_input_tokens_page_summary = page_summary_payload.get("text_input_tokens", 0)
                output_tokens_page_summary = page_summary_payload.get("output_tokens", 0)

        if configs.graph.verbose_node_print:
            debugger.node_report(
                node_id=self.node_id,
                page_summary=self.page_summary,
                elements=self.ui_elements,
                element_tokens={
                    "input_text": text_input_tokens_element_extraction,
                    "input_image": image_input_tokens_element_extraction,
                    "output": output_tokens_element_extraction,
                },
                summary_tokens={
                    "input_text": text_input_tokens_page_summary,
                    "input_image": image_input_tokens_page_summary,
                    "output": output_tokens_page_summary,
                },
            )

        with debugger.time_block("Extracting Text Embedding", color="gray"):
            self.qwen_embedding = vlm_client.extract_text_embedding(self.canonical_page_layout)

        debugger.subrule("Node {} created successfully".format(self.node_id), color="blue", symbol="=")

    def add_clip_embeddings(self, image_utils, ui_elements, screenshot):
        """Batch CLIP crop per action with bbox; skip invalid bboxes."""
        crops = []
        element_indices = []

        for idx, element in enumerate(ui_elements):
            bb = element.get("boundingBox")
            if bb is None or len(bb) != 4:
                continue
            x1, y1, x2, y2 = [int(round(val)) for val in bb]
            if x2 <= x1 or y2 <= y1:
                continue
            crop = screenshot[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            element_indices.append(idx)

        if crops:
            embs = image_utils.extract_clip_embedding_batch(crops, use_gray=False)
            embs = np.asarray(embs, dtype=np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms > 1e-12, norms, 1.0)
            embs = embs / norms

            for i, idx in enumerate(element_indices):
                ui_elements[idx]["clip_embedding"] = embs[i].reshape(-1)

        return ui_elements

    @staticmethod
    def extract_ui_elements_vlm(screenshot, vlm_client, configs, action_history=None):
        action_history_str = [
            a.get("description", a.get("type", "")) for a in action_history or []
        ]
        app_description = getattr(getattr(configs, "app", None), "description", "")
        elements_payload = vlm_client.extract_elements_from_page(
            screenshot, application_description=app_description, action_history=action_history_str
        )
        vlm_elements = (
            elements_payload.get("elements", [])
            if isinstance(elements_payload, dict)
            else (elements_payload or [])
        )
        image_input_tokens = elements_payload.get("image_input_tokens", 0)
        text_input_tokens = elements_payload.get("text_input_tokens", 0)
        output_tokens = elements_payload.get("output_tokens", 0)

        draw = np.copy(screenshot)
        parsed_elements = []
        i = 0

        for element in vlm_elements:
            bbox = element.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [int(round(val)) for val in bbox]

            if x2 <= x1 or y2 <= y1:
                continue

            draw = cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), thickness=4)
            draw = cv2.putText(
                draw,
                str(element.get("label", "")),
                (x1, y1 + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            std_elem = {
                "boundingBox": [x1, y1, x2, y2],
                "id": i,
                "type": element.get("type", ""),
                "description": element.get("description").strip(),
                "explored": False,
            }
            parsed_elements.append(std_elem)
            i += 1

        return parsed_elements, draw, image_input_tokens, text_input_tokens, output_tokens

    def refresh_elements(self, screenshot, vlm_client, configs, image_utils, action_history=None):
        """Re-run VLM element extraction; append only elements not matched by IoU or CLIP+center."""
        new_ui_elements, _, _, _, _ = self.extract_ui_elements_vlm(
            screenshot, vlm_client, configs, action_history=action_history
        )
        h, w = int(screenshot.shape[0]), int(screenshot.shape[1])

        def _iou_xyxy(a, b):
            try:
                ax1, ay1, ax2, ay2 = [float(x) for x in a]
                bx1, by1, bx2, by2 = [float(x) for x in b]
            except Exception:
                return 0.0
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            denom = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            return float(inter / denom) if denom > 1e-9 else 0.0

        def _clip_center_dup(bb, new_emb, fingerprints):
            if w <= 0 or h <= 0 or new_emb is None:
                return False
            ncx = (float(bb[0]) + float(bb[2])) / 2.0
            ncy = (float(bb[1]) + float(bb[3])) / 2.0
            v = np.asarray(new_emb, dtype=np.float32).ravel()
            nv = float(np.linalg.norm(v))
            if nv > 1e-12:
                v = v / nv
            for fp in fingerprints:
                u = fp.get("emb")
                if u is None or float(np.dot(v, u)) <= 0.8:
                    continue
                if abs(ncx - fp["cx"]) / float(w) >= 0.01:
                    continue
                if abs(ncy - fp["cy"]) / float(h) >= 0.01:
                    continue
                return True
            return False

        existing_bboxes = []
        fingerprints = []
        existing_descriptions = set()
        for el in self.ui_elements or []:
            bb = el.get("boundingBox") or el.get("bbox")
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                bb = None
            else:
                bb = [float(v) for v in bb]
                existing_bboxes.append(bb)
            em = el.get("clip_embedding")
            if em is not None:
                em = np.asarray(em, dtype=np.float32).ravel()
                en = float(np.linalg.norm(em))
                if en > 1e-12:
                    em = em / en
            if bb is not None:
                fingerprints.append({"cx": (bb[0] + bb[2]) / 2.0, "cy": (bb[1] + bb[3]) / 2.0, "emb": em})
            desc = str(el.get("description") or "").strip().lower()
            if desc:
                existing_descriptions.add(desc)

        next_id = max((int(e.get("id", -1)) for e in (self.ui_elements or [])), default=-1) + 1
        to_add = []
        for element in new_ui_elements or []:
            desc = str(element.get("description") or "").strip().lower()
            if desc and desc in existing_descriptions:
                continue
            bb = element.get("boundingBox") or element.get("bbox")
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            bb = [float(v) for v in bb]
            if any(_iou_xyxy(bb, old) >= 0.6 for old in existing_bboxes):
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in bb]
            new_emb = None
            if w > 0 and h > 0:
                xi1, yi1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
                xi2, yi2 = max(xi1 + 1, min(w, x2)), max(yi1 + 1, min(h, y2))
                crop = screenshot[yi1:yi2, xi1:xi2]
                if crop.size > 0:
                    try:
                        emb = image_utils.extract_clip_embedding(crop, use_gray=False)
                        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
                        n = float(np.linalg.norm(emb))
                        if n > 1e-12:
                            emb = emb / n
                        new_emb = emb
                    except Exception:
                        pass
            if _clip_center_dup(bb, new_emb, fingerprints):
                continue
            element = dict(element)
            element["id"] = next_id
            element["explored"] = False
            if new_emb is not None:
                element["clip_embedding"] = new_emb
            to_add.append(element)
            existing_bboxes.append(bb)
            if desc:
                existing_descriptions.add(desc)
            fingerprints.append(
                {"cx": (bb[0] + bb[2]) / 2.0, "cy": (bb[1] + bb[3]) / 2.0, "emb": new_emb}
            )
            next_id += 1

        if to_add:
            self.ui_elements.extend(to_add)
            self.canonical_page_layout = elements_to_canonical_page_layout(self.ui_elements)
        return len(to_add)


