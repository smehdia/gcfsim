import os
import io
import cv2
import time
import torch
import base64
import numpy as np
from PIL import Image
from cv2 import Laplacian, CV_64F
from torchvision.ops import box_iou
import torchvision.ops.boxes as bops 
from sklearn.preprocessing import normalize
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, Dinov2Model
from skimage.metrics import structural_similarity as ssim
from FlagEmbedding import BGEM3FlagModel

from contextlib import contextmanager, redirect_stderr, redirect_stdout

from transformers import logging as hf_logging
import warnings

hf_logging.set_verbosity_error()



@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield

class ImageUtils:
    def __init__(self):
        # with suppress_output():
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model_embedder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading CLIP Model ...")
        self.clip_model = self.clip_model.to(self.device)
    
    def extract_clip_embedding(self, opencv_bgr_image, use_gray=True):
        pil_image = Image.fromarray(cv2.cvtColor(opencv_bgr_image, cv2.COLOR_BGR2RGB))
        if use_gray:
            pil_image = pil_image.convert('L')

        inputs = self.clip_image_processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            raw_out = self.clip_model.get_image_features(**inputs)
        # Normalize different possible output types to pooled/projected CLIP vectors only.
        if isinstance(raw_out, torch.Tensor):
            image_embeds = raw_out
        elif hasattr(raw_out, "image_embeds") and raw_out.image_embeds is not None:
            image_embeds = raw_out.image_embeds
        elif hasattr(raw_out, "pooler_output") and raw_out.pooler_output is not None:
            image_embeds = raw_out.pooler_output

        if image_embeds.ndim != 2:
            raise ValueError(
                f"extract_clip_embedding expected pooled 2D image embeddings [B,D], got shape {tuple(image_embeds.shape)}."
            )
        # Normalize embeddings (for cosine similarity etc.)
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
        
        return image_embeds.cpu().detach().numpy()

    def extract_clip_embedding_batch(self, opencv_bgr_images, use_gray=True, use_text_blur=False):
        """
        Batch CLIP image embeddings for a list of OpenCV BGR crops. Shape ``(N, D)``, L2-normalized.
        If ``use_text_blur`` is True, falls back to per-image ``extract_clip_embedding`` (slower).
        """
        dim = int(getattr(self.clip_model.config, "projection_dim", 512))
        if not opencv_bgr_images:
            return np.zeros((0, dim), dtype=np.float32)
        if use_text_blur:
            rows = []
            for im in opencv_bgr_images:
                rows.append(
                    np.squeeze(
                        self.extract_clip_embedding(im, use_gray=use_gray, use_text_blur=True)
                    ).astype(np.float32)
                )
            return np.stack(rows, axis=0)
        pil_images = []
        for opencv_bgr_image in opencv_bgr_images:
            pil_image = Image.fromarray(cv2.cvtColor(opencv_bgr_image, cv2.COLOR_BGR2RGB))
            if use_gray:
                pil_image = pil_image.convert("L")
            pil_images.append(pil_image)
        inputs = self.clip_image_processor(images=pil_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            raw_out = self.clip_model.get_image_features(**inputs)
        if isinstance(raw_out, torch.Tensor):
            image_embeds = raw_out
        elif hasattr(raw_out, "image_embeds") and raw_out.image_embeds is not None:
            image_embeds = raw_out.image_embeds
        elif hasattr(raw_out, "pooler_output") and raw_out.pooler_output is not None:
            image_embeds = raw_out.pooler_output
        else:
            raise TypeError(
                f"Unsupported CLIP image batch output type {type(raw_out)}; expected Tensor/image_embeds/pooler_output."
            )
        if image_embeds.ndim != 2:
            raise ValueError(
                f"extract_clip_embedding_batch expected pooled 2D image embeddings [N,D], got shape {tuple(image_embeds.shape)}."
            )
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
        return image_embeds.cpu().detach().numpy().astype(np.float32)


    def visualize_action_on_screenshot(self, screenshot, action, color=(0, 255, 0), thickness=2, show_title=True):
        """Draw one ui_element on a BGR screenshot copy; scroll/swipe use directional arrows."""
        img_vis = screenshot.copy()
        action = action or {}
        bbox = action.get("boundingBox") or action.get("bbox")
        if not bbox or len(bbox) < 4:
            return img_vis
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        pad = max(8, min(x2 - x1, y2 - y1) // 6)
        el_type = str(action.get("type") or "").strip().lower()

        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, thickness)
        if el_type == "scroll":
            cv2.arrowedLine(img_vis, (cx, y1 + pad), (cx, y2 - pad), color, thickness, tipLength=0.3)
        elif el_type == "swipe":
            cv2.arrowedLine(img_vis, (x1 + pad, cy), (x2 - pad, cy), color, thickness, tipLength=0.3)

        if show_title:
            desc = str(action.get("description") or action.get("type") or "action")
            y_text = y1 - 10 if y1 > 20 else y1 + 20
            cv2.putText(img_vis, desc, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return img_vis


    