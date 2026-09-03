#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Defaults (paper GCFSim recipe)
# ---------------------------------------------------------------------------

SIGLIP_ID = "google/siglip-base-patch16-384"
TEXT_ENCODER_ID = "sentence-transformers/all-mpnet-base-v2"
TAU_G = 0.95
MAX_GOALS = 24
EPOCHS = 20
SEED = 42
TARGET_PAIRS_PER_GOAL = 400
PAPER_SAMPLING = {
    "positive_fraction": 0.30,
    "same_distance_negative_fraction": 0.25,
    "visual_hard_negative_fraction": 0.20,
    "goal_flip_fraction": 0.15,
    "random_negative_fraction": 0.10,
    "max_positive_pairs_per_class": 15,
    "max_goal_flips_per_app": 1200,
    "visual_hard_sim_threshold": 0.55,
}

# ---------------------------------------------------------------------------
# Graph schema + load
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    node_id: str
    screenshot_path: str
    page_description: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    action_type: str
    element_text: str | None = None
    high_level: str | None = None
    low_level: str | None = None
    description: str | None = None
    key: int = 0


@dataclass
class GoalDefinition:
    goal_id: str
    goal_text: str
    target_node_ids: list[str]
    app_id: str = ""


@dataclass
class AppGraph:
    app_id: str
    app_dir: Path
    nodes: dict[str, GraphNode]
    edges: list[GraphEdge]
    goals: list[GoalDefinition]
    _outgoing: dict[str, list[GraphEdge]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self._outgoing:
            out: dict[str, list[GraphEdge]] = {nid: [] for nid in self.nodes}
            for e in self.edges:
                if e.source_id in out:
                    out[e.source_id].append(e)
            self._outgoing = out

    def outgoing_edges(self, node_id: str) -> list[GraphEdge]:
        return self._outgoing.get(node_id, [])


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_") or "goal"
    return s[:max_len]


def _edge_descriptions(
    edge_level: dict, src: str, tgt: str, key: int, fallback: str
) -> tuple[str, str]:
    entries = edge_level.get(f"{src}|{tgt}") or []
    if isinstance(entries, dict):
        entries = [entries]
    for ent in entries:
        aids = [str(a) for a in (ent.get("action_ids") or [])]
        if not aids or str(key) in aids:
            return (
                ent.get("high_level_action_description") or fallback,
                ent.get("low_level_action_description") or fallback,
            )
    if entries:
        ent = entries[0]
        return (
            ent.get("high_level_action_description") or fallback,
            ent.get("low_level_action_description") or fallback,
        )
    return fallback, fallback


def build_goals(
    app_id: str,
    user_intents: dict,
    valid_nodes: set[str],
    max_goals: int | None = MAX_GOALS,
    seed: int = SEED,
) -> list[GoalDefinition]:
    intent_to_nodes: dict[str, list[str]] = defaultdict(list)
    for nid, payload in user_intents.items():
        if nid not in valid_nodes:
            continue
        for intent in (payload or {}).get("user_intents") or []:
            text = str(intent).strip()
            if text and nid not in intent_to_nodes[text]:
                intent_to_nodes[text].append(nid)
    items = [(t, ns) for t, ns in intent_to_nodes.items() if ns]
    items.sort(key=lambda x: (-len(x[1]), x[0]))
    if max_goals is not None and len(items) > max_goals:
        multi = [it for it in items if len(it[1]) > 1]
        singles = [it for it in items if len(it[1]) == 1]
        singles.sort(key=lambda it: hashlib.md5(f"{seed}:{it[0]}".encode()).hexdigest())
        items = (multi + singles)[:max_goals]
    goals = []
    for text, nodes in items:
        gid = f"{app_id}__{_slug(text)}__{hashlib.md5(text.encode()).hexdigest()[:8]}"
        goals.append(GoalDefinition(gid, text, list(nodes), app_id))
    return goals


def load_app_graph(app_dir: Path, max_goals: int = MAX_GOALS, seed: int = SEED) -> AppGraph:
    app_dir = Path(app_dir)
    app_id = app_dir.name
    graph_json = _load_json(app_dir / "graph.json", {"nodes": [], "links": []})
    node_level = _load_json(app_dir / "node_level_information.json")
    edge_level = _load_json(app_dir / "edge_level_information.json")
    user_intents = _load_json(app_dir / "user_intents.json")
    shots = app_dir / "screenshots"

    nodes: dict[str, GraphNode] = {}
    for n in graph_json.get("nodes") or []:
        rid = n.get("node_id") or n.get("id")
        if not rid:
            continue
        shot = shots / f"{rid}.jpg"
        if not shot.is_file():
            shot = shots / f"{rid}.png"
        nli = node_level.get(rid) or {}
        parts = []
        if isinstance(nli, dict):
            for k in ("high_level", "medium_level", "low_level"):
                v = (nli.get(k) or "").strip()
                if v:
                    parts.append(v)
        page_desc = "\n\n".join(parts) if parts else (n.get("page_purpose") or n.get("page_summary") or "")
        nodes[rid] = GraphNode(
            node_id=rid,
            screenshot_path=str(shot) if shot.is_file() else "",
            page_description=page_desc,
        )

    edges: list[GraphEdge] = []
    for link in graph_json.get("links") or []:
        src, tgt = link.get("source"), link.get("target")
        if src not in nodes or tgt not in nodes:
            continue
        key = int(link.get("key", 0))
        desc = str(link.get("description") or "")
        high, low = _edge_descriptions(edge_level, src, tgt, key, desc)
        edges.append(
            GraphEdge(
                source_id=src,
                target_id=tgt,
                action_type=str(link.get("type") or "nav").strip().lower() or "nav",
                element_text=desc or low or None,
                high_level=high,
                low_level=low,
                description=desc,
                key=key,
            )
        )

    goals = build_goals(app_id, user_intents, set(nodes), max_goals=max_goals, seed=seed)
    return AppGraph(app_id=app_id, app_dir=app_dir, nodes=nodes, edges=edges, goals=goals)


# ---------------------------------------------------------------------------
# Frozen SigLIP
# ---------------------------------------------------------------------------


def letterbox(image: Image.Image, size: int = 384) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = image.resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


class FrozenSigLIP:
    def __init__(self, checkpoint: str = SIGLIP_ID, image_size: int = 384, device: str | None = None):
        from transformers import AutoModel, AutoProcessor

        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        boxed = [letterbox(im, self.image_size) for im in images]
        inputs = self.processor(images=boxed, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        if hasattr(self.model, "get_image_features"):
            feats = self.model.get_image_features(**inputs)
        else:
            feats = self.model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        return F.normalize(feats.float(), dim=-1).cpu().numpy()

    @torch.no_grad()
    def encode_image_paths(self, paths: list[str], batch_size: int = 16) -> np.ndarray:
        embs = []
        for i in range(0, len(paths), batch_size):
            images = []
            for p in paths[i : i + batch_size]:
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0)))
            embs.append(self.encode_images(images))
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, 768), dtype=np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
            if hasattr(self.model, "get_text_features"):
                feats = self.model.get_text_features(**inputs)
            else:
                feats = self.model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                ).pooler_output
            embs.append(F.normalize(feats.float(), dim=-1).cpu().numpy())
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, 768), dtype=np.float32)


def embed_nodes(graph: AppGraph, encoder: FrozenSigLIP, cache: Path | None = None) -> dict[str, np.ndarray]:
    if cache and cache.is_file():
        data = np.load(cache)
        return {nid: data["embeddings"][i] for i, nid in enumerate(data["node_ids"])}
    ids, paths = [], []
    for nid, node in graph.nodes.items():
        if node.screenshot_path and Path(node.screenshot_path).is_file():
            ids.append(nid)
            paths.append(node.screenshot_path)
    if not paths:
        return {}
    emb = encoder.encode_image_paths(paths)
    out = {nid: emb[i] for i, nid in enumerate(ids)}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, node_ids=np.array(ids), embeddings=emb.astype(np.float32))
    return out


def embed_goals(graph: AppGraph, encoder: FrozenSigLIP, cache: Path | None = None) -> dict[str, np.ndarray]:
    if cache and cache.is_file():
        data = np.load(cache)
        return {gid: data["embeddings"][i] for i, gid in enumerate(data["goal_ids"])}
    ids = [g.goal_id for g in graph.goals]
    texts = [g.goal_text for g in graph.goals]
    if not texts:
        return {}
    emb = encoder.encode_texts(texts)
    out = {gid: emb[i] for i, gid in enumerate(ids)}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, goal_ids=np.array(ids), embeddings=emb.astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# Semantic target expansion (sentence-transformers)
# ---------------------------------------------------------------------------


class TextEmbedder:
    def __init__(self, model_id: str = TEXT_ENCODER_ID, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device=device)
        self._mem: dict[str, np.ndarray] = {}

    def encode_one(self, text: str) -> np.ndarray:
        if text not in self._mem:
            v = self.model.encode([text], normalize_embeddings=True)[0]
            self._mem[text] = np.asarray(v, dtype=np.float32)
        return self._mem[text]


def load_node_intents(app_dir: Path) -> dict[str, list[str]]:
    raw = _load_json(app_dir / "user_intents.json")
    out: dict[str, list[str]] = {}
    for nid, payload in raw.items():
        intents = [str(it).strip() for it in (payload or {}).get("user_intents") or [] if str(it).strip()]
        if intents:
            out[nid] = intents
    return out


def expand_targets(
    graph: AppGraph, app_dir: Path, embedder: TextEmbedder, tau: float = TAU_G
) -> dict[str, list[str]]:
    ni = load_node_intents(app_dir)
    out: dict[str, list[str]] = {}
    for g in graph.goals:
        zg = embedder.encode_one(g.goal_text)
        original = set(g.target_node_ids)
        added = []
        for nid, intents in ni.items():
            if nid not in graph.nodes or nid in original:
                continue
            best = max(float(np.dot(zg, embedder.encode_one(gp))) for gp in intents)
            if best >= tau:
                added.append(nid)
        out[g.goal_id] = list(original) + added
    return out


# ---------------------------------------------------------------------------
# Continuation teacher (GCFSim / manual9 semantic target)
# ---------------------------------------------------------------------------

_INTENT_PATTERNS = [
    ("scroll", re.compile(r"\b(scroll|swipe|page down|page up)\b", re.I)),
    ("dismiss", re.compile(r"\b(dismiss|close|cancel|back|exit)\b", re.I)),
    ("input", re.compile(r"\b(type|enter|search|input|edit text|fill)\b", re.I)),
    ("select", re.compile(r"\b(select|choose|pick|filter|toggle|switch)\b", re.I)),
    ("open", re.compile(r"\b(open|view|show|expand|reveal)\b", re.I)),
    ("navigate", re.compile(r"\b(navigate|go to|goto|visit|move to|proceed)\b", re.I)),
    ("play", re.compile(r"\b(play|pause|resume|shuffle|queue)\b", re.I)),
    ("purchase", re.compile(r"\b(buy|purchase|checkout|subscribe|premium|pay)\b", re.I)),
    ("share", re.compile(r"\b(share|send|invite|copy link)\b", re.I)),
]
_TYPE_TO_INTENT = {
    "scroll": "scroll",
    "swipe": "scroll",
    "input": "input",
    "button": "select",
    "icon": "select",
    "card": "open",
    "nav": "navigate",
}


def broad_action_intent(edge: GraphEdge) -> str:
    text = " ".join(
        filter(None, [edge.high_level or "", edge.low_level or "", edge.description or "", edge.element_text or "", edge.action_type or ""])
    )
    for intent, pat in _INTENT_PATTERNS:
        if pat.search(text):
            return intent
    return _TYPE_TO_INTENT.get((edge.action_type or "").lower(), "other")


def reverse_distances(graph: AppGraph, target_ids: list[str]) -> dict[str, float]:
    rev: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    for e in graph.edges:
        if e.target_id in rev and e.source_id in rev:
            rev[e.target_id].append(e.source_id)
    dist: dict[str, float] = {nid: float("inf") for nid in graph.nodes}
    q: deque[str] = deque()
    for tid in target_ids:
        if tid in dist:
            dist[tid] = 0
            q.append(tid)
    while q:
        cur = q.popleft()
        d = int(dist[cur])
        for src in rev.get(cur, []):
            if dist[src] > d + 1:
                dist[src] = d + 1
                q.append(src)
    return dist


def build_continuation_teacher(
    graph: AppGraph, goal: GoalDefinition, target_ids: list[str]
) -> dict[str, Any]:
    """Full recursive SET of (φ(e), successor_class) over optimal edges — GCFSim teacher."""
    distance = reverse_distances(graph, target_ids)
    planning_class: dict[str, int] = {tid: 0 for tid in target_ids if tid in graph.nodes}
    plan_sig_to_class: dict[tuple, int] = {("GOAL",): 0}
    next_id = 1
    finite = [int(d) for d in distance.values() if d != float("inf")]
    if not finite:
        for nid, d in distance.items():
            if d == float("inf"):
                planning_class[nid] = -1
        return {
            "goal_id": goal.goal_id,
            "goal_text": goal.goal_text,
            "class_id": planning_class,
            "distance": {k: (None if v == float("inf") else int(v)) for k, v in distance.items()},
            "target_ids": list(target_ids),
        }

    for depth in range(1, max(finite) + 1):
        for node_id in [n for n, d in distance.items() if d == depth]:
            d = distance[node_id]
            edges = [
                e
                for e in graph.outgoing_edges(node_id)
                if distance.get(e.target_id, float("inf")) == d - 1
            ]
            pairs = set()
            for e in edges:
                if e.target_id in planning_class:
                    pairs.add((broad_action_intent(e), planning_class[e.target_id]))
            sig = ("MANUAL9", tuple(sorted(pairs)))
            if sig not in plan_sig_to_class:
                plan_sig_to_class[sig] = next_id
                next_id += 1
            planning_class[node_id] = plan_sig_to_class[sig]

    for nid, d in distance.items():
        if d == float("inf"):
            planning_class[nid] = -1

    return {
        "goal_id": goal.goal_id,
        "goal_text": goal.goal_text,
        "class_id": planning_class,
        "distance": {k: (None if v == float("inf") else int(v)) for k, v in distance.items()},
        "target_ids": list(target_ids),
    }


# ---------------------------------------------------------------------------
# Pair sampling
# ---------------------------------------------------------------------------


@dataclass
class MetricPair:
    app_id: str
    goal_id: str
    goal_text: str
    state_a_id: str
    state_b_id: str
    class_a: int
    class_b: int
    label: int
    pair_type: str
    confidence: float = 1.0
    goal_id_neg: str | None = None


def _valid_pair(graph: AppGraph, a: str, b: str) -> bool:
    if a == b:
        return False
    sa, sb = graph.nodes[a].screenshot_path, graph.nodes[b].screenshot_path
    if not sa or not sb:
        return False
    return Path(sa).name != Path(sb).name


def _by_class(class_id: dict[str, int]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = defaultdict(list)
    for nid, cid in class_id.items():
        if cid >= 0:
            out[cid].append(nid)
    return out


def sample_pairs(
    graph: AppGraph,
    teachers: dict[str, dict],
    img: dict[str, np.ndarray],
    goal_emb: dict[str, np.ndarray],
    seed: int = SEED,
) -> list[MetricPair]:
    cfg = PAPER_SAMPLING
    rng = random.Random(seed)
    all_pairs: list[MetricPair] = []
    n_target = TARGET_PAIRS_PER_GOAL

    # Goal flips
    flips: list[MetricPair] = []
    goal_ids = list(teachers)
    node_ids = [n for n in graph.nodes if graph.nodes[n].screenshot_path]
    seen: set[tuple] = set()
    attempts = 0
    max_flips = int(cfg["max_goal_flips_per_app"])
    while len(flips) < max_flips and attempts < max_flips * 40 and len(goal_ids) >= 2 and len(node_ids) >= 2:
        attempts += 1
        a, b = rng.sample(node_ids, 2)
        if not _valid_pair(graph, a, b):
            continue
        labels = {}
        for gid in goal_ids:
            ca = teachers[gid]["class_id"].get(a, -1)
            cb = teachers[gid]["class_id"].get(b, -1)
            if ca < 0 and cb < 0:
                continue
            labels[gid] = int(ca == cb and ca >= 0)
        pos = [g for g, lab in labels.items() if lab == 1]
        neg = [g for g, lab in labels.items() if lab == 0]
        if not pos or not neg:
            continue
        g_pos, g_neg = rng.choice(pos), rng.choice(neg)
        key = tuple(sorted([a, b]) + [g_pos, g_neg])
        if key in seen:
            continue
        seen.add(key)
        cp = teachers[g_pos]["class_id"]
        flips.append(
            MetricPair(
                graph.app_id, g_pos, teachers[g_pos]["goal_text"], a, b,
                cp.get(a, -1), cp.get(b, -1), 1, "goal_flip", 1.0, g_neg,
            )
        )

    for goal_id, result in teachers.items():
        class_id = result["class_id"]
        distance = result.get("distance") or {}
        goal_text = result["goal_text"]
        by_c = _by_class(class_id)
        n = n_target
        n_pos = int(n * cfg["positive_fraction"])
        n_same = int(n * cfg["same_distance_negative_fraction"])
        n_vis = int(n * cfg["visual_hard_negative_fraction"])
        n_flip = int(n * cfg["goal_flip_fraction"])
        n_rand = max(0, n - n_pos - n_same - n_vis - n_flip)

        # Positives
        pos: list[MetricPair] = []
        for cid, nids in by_c.items():
            if len(nids) < 2:
                continue
            cands = [(a, b) for i, a in enumerate(nids) for b in nids[i + 1 :] if _valid_pair(graph, a, b)]
            rng.shuffle(cands)
            for a, b in cands[: int(cfg["max_positive_pairs_per_class"])]:
                pos.append(
                    MetricPair(graph.app_id, goal_id, goal_text, a, b, class_id[a], class_id[b], 1, "positive")
                )
        rng.shuffle(pos)
        pos = pos[:n_pos]

        # Same-distance hard negatives
        by_dist: dict[int, list[str]] = defaultdict(list)
        for nid, d in distance.items():
            if d is None or class_id.get(nid, -1) < 0:
                continue
            by_dist[int(d)].append(nid)
        same: list[MetricPair] = []
        depths = [d for d, ns in by_dist.items() if len(ns) >= 2]
        att = 0
        while len(same) < n_same and depths and att < n_same * 40:
            att += 1
            depth = rng.choice(depths)
            a, b = rng.sample(by_dist[depth], 2)
            if class_id[a] == class_id[b] or not _valid_pair(graph, a, b):
                continue
            same.append(
                MetricPair(graph.app_id, goal_id, goal_text, a, b, class_id[a], class_id[b], 0, "same_distance_hard_negative")
            )

        # Visual hard negatives
        vis: list[MetricPair] = []
        labeled = [nid for nid, cid in class_id.items() if cid >= 0 and nid in img]
        if len(labeled) >= 2:
            X = np.stack([img[n] for n in labeled])
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            sim = X @ X.T
            thr = float(cfg["visual_hard_sim_threshold"])
            cands = []
            for i, a in enumerate(labeled):
                for j in range(i + 1, len(labeled)):
                    b = labeled[j]
                    if class_id[a] != class_id[b] and sim[i, j] >= thr and _valid_pair(graph, a, b):
                        cands.append((float(sim[i, j]), a, b))
            cands.sort(reverse=True)
            for _, a, b in cands[:n_vis]:
                vis.append(
                    MetricPair(graph.app_id, goal_id, goal_text, a, b, class_id[a], class_id[b], 0, "visual_hard_negative")
                )

        # Random negatives
        rand: list[MetricPair] = []
        cids = [c for c, ns in by_c.items() if ns]
        att = 0
        while len(rand) < n_rand and len(cids) >= 2 and att < n_rand * 30:
            att += 1
            ca, cb = rng.sample(cids, 2)
            a, b = rng.choice(by_c[ca]), rng.choice(by_c[cb])
            if not _valid_pair(graph, a, b):
                continue
            rand.append(
                MetricPair(graph.app_id, goal_id, goal_text, a, b, class_id[a], class_id[b], 0, "random_negative", 0.9)
            )

        # Goal flips for this goal
        gflips = [p for p in flips if p.goal_id == goal_id][:n_flip]
        all_pairs.extend(pos + same + vis + rand + gflips)

    return all_pairs


# ---------------------------------------------------------------------------
# Model + training
# ---------------------------------------------------------------------------


class SimpleGoalMetric(nn.Module):
    """z(s,g) = F_θ(x_s, x_g); p = σ(H([|z1−z2|, z1⊙z2]))."""

    def __init__(self, image_dim: int = 768, goal_dim: int = 768, hidden_dim: int = 512, metric_dim: int = 256):
        super().__init__()
        self.image_projection = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, hidden_dim))
        self.goal_projection = nn.Sequential(nn.LayerNorm(goal_dim), nn.Linear(goal_dim, hidden_dim))
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, metric_dim)
        )
        self.pair_head = nn.Sequential(
            nn.LayerNorm(metric_dim * 2),
            nn.Linear(metric_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def encode(self, image: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        img = self.image_projection(image)
        g = self.goal_projection(goal)
        z = self.fusion(torch.cat([img, g, img * g], dim=-1))
        return F.normalize(z, dim=-1)

    def forward(self, image_a, image_b, goal):
        z1, z2 = self.encode(image_a, goal), self.encode(image_b, goal)
        return self.pair_head(torch.cat([torch.abs(z1 - z2), z1 * z2], dim=-1)).squeeze(-1)

    def similarity(self, image_a, image_b, goal):
        return torch.sigmoid(self.forward(image_a, image_b, goal))


class PairDataset(Dataset):
    def __init__(self, pairs: list[MetricPair], img: dict[str, np.ndarray], goal: dict[str, np.ndarray], app: str):
        self.img, self.goal, self.app = img, goal, app
        self.pairs = [
            p
            for p in pairs
            if p.state_a_id in img and p.state_b_id in img and p.goal_id in goal
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        item = {
            "image_a": torch.tensor(self.img[p.state_a_id], dtype=torch.float32),
            "image_b": torch.tensor(self.img[p.state_b_id], dtype=torch.float32),
            "goal": torch.tensor(self.goal[p.goal_id], dtype=torch.float32),
            "label": torch.tensor(float(p.label)),
            "confidence": torch.tensor(float(p.confidence)),
            "class_a": p.class_a,
            "class_b": p.class_b,
            "has_flip": torch.tensor(0.0),
            "goal_neg": torch.zeros(self.goal[p.goal_id].shape[0]),
        }
        if p.goal_id_neg and p.goal_id_neg in self.goal:
            item["goal_neg"] = torch.tensor(self.goal[p.goal_id_neg], dtype=torch.float32)
            item["has_flip"] = torch.tensor(1.0)
        return item


def _collate(batch):
    out = {}
    for k in ("image_a", "image_b", "goal", "goal_neg", "label", "confidence", "has_flip"):
        out[k] = torch.stack([b[k] for b in batch])
    out["class_a"] = [b["class_a"] for b in batch]
    out["class_b"] = [b["class_b"] for b in batch]
    return out


def _supcon(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    keep = y >= 0
    if keep.sum() < 2:
        return z.new_zeros(())
    z = F.normalize(z[keep], dim=-1)
    y = y[keep]
    n = z.size(0)
    sim = (z @ z.T) / max(temperature, 1e-4)
    mask = torch.ones((n, n), device=z.device) - torch.eye(n, device=z.device)
    pos = (y.view(-1, 1) == y.view(1, -1)).float() * mask
    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    log_prob = logits - torch.log((torch.exp(logits) * mask).sum(1, keepdim=True) + 1e-8)
    count = pos.sum(1)
    valid = count > 0
    if valid.sum() < 1:
        return z.new_zeros(())
    return -((pos * log_prob).sum(1) / count.clamp(min=1))[valid].mean()


def train_metric(
    model: SimpleGoalMetric,
    train_pairs: list[MetricPair],
    val_pairs: list[MetricPair],
    img: dict[str, np.ndarray],
    goal: dict[str, np.ndarray],
    app: str,
    device: str,
    epochs: int,
    ckpt_dir: Path,
    seed: int = SEED,
) -> dict[str, Any]:
    model = model.to(device)
    train_ds = PairDataset(train_pairs, img, goal, app)
    val_ds = PairDataset(val_pairs, img, goal, app)
    if len(train_ds) == 0:
        raise SystemExit("no training pairs")

    rng = random.Random(seed)
    history, best_val, patience = [], -1.0, 0
    best_state = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        # Sample balanced-ish batches of 128
        idxs = list(range(len(train_ds)))
        rng.shuffle(idxs)
        epoch_loss, n_seen, steps = 0.0, 0, 0
        for bi in range(0, min(len(idxs), 25 * 128), 128):
            batch_idx = idxs[bi : bi + 128]
            if len(batch_idx) < 4:
                break
            batch = _collate([train_ds[i] for i in batch_idx])
            ia = batch["image_a"].to(device)
            ib = batch["image_b"].to(device)
            g = batch["goal"].to(device)
            y = batch["label"].to(device)
            conf = batch["confidence"].to(device)
            logits = model(ia, ib, g)
            z_a, z_b = model.encode(ia, g), model.encode(ib, g)
            l_bce = F.binary_cross_entropy_with_logits(logits, y, weight=conf)
            ca = torch.tensor(batch["class_a"], device=device, dtype=torch.long)
            cb = torch.tensor(batch["class_b"], device=device, dtype=torch.long)
            l_sup = _supcon(torch.cat([z_a, z_b]), torch.cat([ca, cb]))
            l_flip = logits.new_zeros(())
            mask = batch["has_flip"].to(device)
            if mask.sum() > 0:
                p_g = torch.sigmoid(logits)
                p_alt = model.similarity(ia, ib, batch["goal_neg"].to(device))
                p_pos = torch.where(y > 0.5, p_g, p_alt)
                p_neg = torch.where(y > 0.5, p_alt, p_g)
                l_flip = (F.relu(0.2 + p_neg - p_pos) * mask).sum() / mask.sum().clamp(min=1)
            loss = l_bce + 1.0 * l_sup + 2.0 * l_flip
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * ia.size(0)
            n_seen += ia.size(0)
            steps += 1

        # Val AUROC (simple ranking AUC)
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for i in range(0, len(val_ds), 256):
                batch = _collate([val_ds[j] for j in range(i, min(i + 256, len(val_ds)))])
                s = torch.sigmoid(model(batch["image_a"].to(device), batch["image_b"].to(device), batch["goal"].to(device)))
                scores.append(s.cpu().numpy())
                labels.append(batch["label"].numpy())
        if scores:
            s = np.concatenate(scores)
            y = np.concatenate(labels)
            # Mann-Whitney AUROC without sklearn
            pos, neg = s[y >= 0.5], s[y < 0.5]
            if len(pos) and len(neg):
                # P(score_pos > score_neg)
                auroc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
            else:
                auroc = float("nan")
        else:
            auroc = float("nan")

        history.append({"epoch": epoch, "train_loss": epoch_loss / max(1, n_seen), "val_auroc": auroc})
        score = auroc if auroc == auroc else -epoch_loss
        print(f"  epoch {epoch}: loss={history[-1]['train_loss']:.4f} val_auroc={auroc:.4f}", file=sys.stderr)
        if score > best_val + 1e-4:
            best_val = score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_dir / "best.pt")
        else:
            patience += 1
            if patience >= 6:
                break

    model.load_state_dict(best_state)
    return {"model": model, "best_val_auroc": best_val, "history": history}


def split_by_goal(pairs: list[MetricPair], seed: int = SEED):
    rng = random.Random(seed)
    goals = sorted({p.goal_id for p in pairs})
    rng.shuffle(goals)
    n = len(goals)
    n_tr = max(1, int(0.6 * n))
    n_va = max(1, int(0.2 * n)) if n >= 3 else max(1, n - n_tr)
    train_g = set(goals[:n_tr])
    val_g = set(goals[n_tr : n_tr + n_va]) or train_g
    train = [p for p in pairs if p.goal_id in train_g]
    val = [p for p in pairs if p.goal_id in val_g] or train[: max(1, len(train) // 5)]
    return train, val


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------


@torch.no_grad()
def retrieve(
    model: SimpleGoalMetric,
    img_q: np.ndarray,
    goal_q: np.ndarray,
    img_nodes: dict[str, np.ndarray],
    device: str,
    k: int = 1,
) -> list[tuple[str, float]]:
    model.eval()
    g = torch.tensor(goal_q, device=device, dtype=torch.float32).unsqueeze(0)
    zq = model.encode(torch.tensor(img_q, device=device, dtype=torch.float32).unsqueeze(0), g)
    zq = zq.squeeze(0).cpu().numpy()
    ids = list(img_nodes)
    X = torch.tensor(np.stack([img_nodes[n] for n in ids]), device=device, dtype=torch.float32)
    Z = model.encode(X, g.expand(X.size(0), -1)).cpu().numpy()
    zq = zq / (np.linalg.norm(zq) + 1e-8)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    sims = Z @ zq
    order = np.argsort(-sims)
    return [(ids[i], float(sims[i])) for i in order[:k]]


def main():
    ap = argparse.ArgumentParser(description="Standalone GCFSim train + retrieve")
    ap.add_argument("--graph", required=True, type=Path, help="App graph directory")
    ap.add_argument("--screenshot", required=True, type=Path)
    ap.add_argument("--goal", required=True, type=str)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--ckpt-dir", type=Path, default=None)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--top-k", type=int, default=1)
    args = ap.parse_args()

    graph_dir = args.graph.resolve()
    app = graph_dir.name
    ckpt_dir = args.ckpt_dir or (Path(__file__).resolve().parent / "outputs" / "gcfsim_standalone" / app)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print(f"[1/6] load graph {graph_dir}", file=sys.stderr)
    graph = load_app_graph(graph_dir)
    if not graph.goals:
        raise SystemExit(f"no goals in {graph_dir}")
    print(f"  nodes={len(graph.nodes)} edges={len(graph.edges)} goals={len(graph.goals)}", file=sys.stderr)

    print("[2/6] SigLIP embeddings", file=sys.stderr)
    encoder = FrozenSigLIP(device=args.device)
    img = embed_nodes(graph, encoder, cache=ckpt_dir / "node_emb.npz")
    goal_emb = embed_goals(graph, encoder, cache=ckpt_dir / "goal_emb.npz")
    if not img:
        raise SystemExit("no node screenshots found")

    print("[3/6] semantic target expansion", file=sys.stderr)
    text = TextEmbedder(device=args.device)
    targets = expand_targets(graph, graph_dir, text)

    print("[4/6] continuation teacher", file=sys.stderr)
    teachers = {
        g.goal_id: build_continuation_teacher(graph, g, targets[g.goal_id])
        for g in graph.goals
        if targets.get(g.goal_id)
    }

    ckpt_path = ckpt_dir / "best.pt"
    image_dim = int(next(iter(img.values())).shape[-1])
    goal_dim = int(next(iter(goal_emb.values())).shape[-1])

    if args.reuse and ckpt_path.is_file():
        print(f"[5/6] load checkpoint {ckpt_path}", file=sys.stderr)
        model = SimpleGoalMetric(image_dim, goal_dim).to(args.device)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
        model.eval()
        trained = False
    else:
        print("[5/6] sample pairs + train", file=sys.stderr)
        pairs = sample_pairs(graph, teachers, img, goal_emb, seed=SEED)
        train_p, val_p = split_by_goal(pairs)
        print(f"  pairs={len(pairs)} train={len(train_p)} val={len(val_p)}", file=sys.stderr)
        model = SimpleGoalMetric(image_dim, goal_dim)
        info = train_metric(
            model, train_p, val_p, img, goal_emb, app, args.device, args.epochs, ckpt_dir, SEED
        )
        model = info["model"].eval()
        trained = True

    print("[6/6] retrieve", file=sys.stderr)
    q_img = encoder.encode_images([Image.open(args.screenshot).convert("RGB")])[0]
    q_goal = encoder.encode_texts([args.goal])[0]
    ranked = retrieve(model, q_img, q_goal, img, args.device, k=args.top_k)
    if not ranked:
        raise SystemExit("retrieval failed")

    print(ranked[0][0])
    if args.top_k > 1:
        for nid, s in ranked:
            print(f"{nid}\t{s:.4f}")
    print(f"# trained={trained} ckpt={ckpt_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
