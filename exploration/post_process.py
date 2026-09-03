import cv2
from dynaconf import Dynaconf
from PIL import Image

from Debugger import Debugger

from ImageUtils import ImageUtils
from VLM import VLM
from VLM_Yibu import VLM_Yibu

import json
import numpy as np
from networkx.readwrite import json_graph
import networkx as nx
import random
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import traceback
import dashscope
import copy
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
from collections import Counter


from networkx.readwrite import json_graph
from FlagEmbedding import BGEM3FlagModel

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache



for var in ["http_proxy", "https_proxy", "ftp_proxy", "socks_proxy", 
            "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "SOCKS_PROXY"]:
    os.environ.pop(var, None)

def visualize_edge_on_screenshot(screenshot, edge_data, thickness=4):
    """Draw all parallel edges from ``get_edge_data(u, v)`` on a BGR screenshot."""
    if screenshot is None:
        return None
    vis = screenshot.copy()
    if not edge_data:
        return vis

    # MultiDiGraph: {key: attrs}; single-edge attrs: {type, description, boundingBox, ...}
    if any(k in edge_data for k in ("type", "description", "boundingBox", "bbox")):
        edge_attrs_list = [edge_data]
    else:
        edge_attrs_list = [v for v in edge_data.values() if isinstance(v, dict)]

    h, w = vis.shape[:2]
    palette = [
        (0, 255, 0),
        (0, 165, 255),
        (255, 0, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
    ]

    for idx, attrs in enumerate(edge_attrs_list):
        color = palette[idx % len(palette)]
        el_type = str(attrs.get("type") or "").strip().lower()
        desc = str(attrs.get("description") or attrs.get("type") or "action").strip()
        if len(edge_attrs_list) > 1:
            desc = f"[{idx + 1}] {desc}"
        bbox = attrs.get("boundingBox") or attrs.get("bbox")

        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            pad = max(8, min(x2 - x1, y2 - y1) // 6)
            if el_type == "scroll":
                cv2.arrowedLine(
                    vis, (cx, y1 + pad), (cx, y2 - pad), color, thickness, tipLength=0.3
                )
            elif el_type == "swipe":
                cv2.arrowedLine(
                    vis, (x1 + pad, cy), (x2 - pad, cy), color, thickness, tipLength=0.3
                )

            continue

        if "scroll" in el_type:
            cx = w // 2
            cv2.arrowedLine(
                vis,
                (cx, int(h * 0.22)),
                (cx, int(h * 0.78)),
                color,
                max(thickness + 2, 6),
                tipLength=0.08,
            )
            label = desc or "SCROLL"
            ty = 28 + idx * 28
            cv2.putText(
                vis,
                label[:64],
                (10, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            continue

    return vis

def simplify_edge_actions(edge_data):
    if not edge_data:
        return []

    simplified = []
    for key, attrs in edge_data.items():
        simplified.append({
            "edge_key": key,
            "type": attrs.get("type", ""),
            "description": attrs.get("description", "")
        })

    return simplified


def resolve_enhanced_page_summary(path_node_id, node_intents, nx_graph):
    """Return enhanced_page_summary dict from node_intents, or a graph-metadata fallback."""
    node_intents = node_intents or {}
    enhanced = node_intents.get(path_node_id, {}).get("enhanced_page_summary")
    if isinstance(enhanced, dict) and enhanced:
        return enhanced

    node_data = nx_graph.nodes[path_node_id]
    compact = str(node_data.get("page_summary", "") or "").strip()
    page_purpose = str(node_data.get("page_purpose", "") or "").strip() or compact
    return {
        "tag": compact[:80] if compact else path_node_id,
        "active_tab": node_data.get("active_tab"),
        "active_subtab": node_data.get("active_subtab"),
        "page_purpose": page_purpose,
        "screen_type": "",
        "selected_navigation": "",
        "visual_landmarks": [],
        "regions": {"top_bar": "", "main_area": "", "bottom_nav": ""},
        "salient_controls": [],
    }


def build_paths_for_node(nx_graph, node_id, logs_root):
    root_node_ids = [node_id for node_id, node in nx_graph.nodes(data=True) if node.get("is_root", False)]
    paths = dict()
    for root_node_id in root_node_ids:
        try:
            node_path = nx.shortest_path(nx_graph, source=root_node_id, target=node_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        screenshots = []
        page_summaries = []
        actions = []
        for i, node in enumerate(node_path):
            screenshot = cv2.imread(os.path.join(logs_root, "screenshots", node + ".jpg"))
            screenshots.append(screenshot)
            page_summaries.append(nx_graph.nodes[node].get("page_summary", ""))
            if i < len(node_path) - 1:
                screenshots[i] = visualize_edge_on_screenshot(
                    screenshots[i],
                    nx_graph.get_edge_data(node_path[i], node_path[i + 1]),
                )
                actions.append(
                    simplify_edge_actions(
                        nx_graph.get_edge_data(node_path[i], node_path[i + 1])
                    )
                )

        paths[root_node_id] = {
            "screenshots": screenshots,
            "page_summaries": page_summaries,
            "actions": actions,
        }

    return paths


def _process_node_level_information(node_id, nx_graph, vlm_client, logs_root):
    
    img = cv2.imread(os.path.join(logs_root, "screenshots", node_id + ".jpg"))
    page_summary = nx_graph.nodes[node_id].get("page_summary", "")
    out = vlm_client.get_node_level_information(img, page_summary)
    return node_id, out["page_description"]


def save_node_level_information(nx_graph, vlm_client, configs, dbg):
    logs_root = configs.logs.root
    node_level_information_path = os.path.join(logs_root, "node_level_information.json")


    if os.path.exists(node_level_information_path):
        with open(node_level_information_path, "r", encoding="utf-8") as f:
            try:
                node_level_information = json.load(f)
            except json.JSONDecodeError:
                node_level_information = {}
    else:
        node_level_information = {}

    max_workers = int(getattr(configs.post_process, "max_workers", 4) or 4)
    write_lock = threading.Lock()
    node_ids = list(nx_graph.nodes())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_node_level_information,
                node_id,
                nx_graph,
                vlm_client,
                logs_root,
            ): node_id
            for node_id in node_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="post-process nodes"):
            node_id = futures[future]
            try:
                result = future.result()
            except Exception:
                dbg.log(f"Failed post-process for node {node_id}", color="red")
                traceback.print_exc()
                continue

            if result is None:
                continue

            _, entry = result
            with write_lock:
                node_level_information[node_id] = entry
                with open(node_level_information_path, "w", encoding="utf-8") as f:
                    json.dump(node_level_information, f, ensure_ascii=False, indent=4)

    return node_level_information

def _process_edge_level_information(edge_id, node_level_information, nx_graph, vlm_client, logs_root, dbg):

    source_id, target_id = edge_id
    source_id = str(source_id)
    target_id = str(target_id)

    edge_data = nx_graph.get_edge_data(source_id, target_id)
    source_node_level_information = node_level_information.get(source_id, {})
    target_node_level_information = node_level_information.get(target_id, {})

    source_screenshot = cv2.imread(os.path.join(logs_root, "screenshots", source_id + ".jpg"))
    action_crops = []
    for edge_key, edge_attrs in edge_data.items():
        if 'boundingBox' in edge_attrs.keys():
            x1, y1, x2, y2 = edge_attrs['boundingBox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            action = source_screenshot[y1:y2, x1:x2]
            action_crops.append(action)

    h, w = source_screenshot.shape[:2]
    # we copy edge data and create a normalized bounding box
    normalized_edge_data = copy.deepcopy(edge_data)
    for edge_key, edge_attrs in normalized_edge_data.items():
        if 'boundingBox' in edge_attrs.keys():
            x1, y1, x2, y2 = edge_attrs['boundingBox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            edge_attrs['boundingBox'] = round(float(x1) / w, 2), round(float(y1) / h, 2), round(float(x2) / w, 2), round(float(y2) / h, 2)

    transition_info = vlm_client.get_transition_info(action_crops, normalized_edge_data, source_node_level_information, target_node_level_information)

    return edge_id, transition_info['transition_info']



def save_edge_level_information(nx_graph, node_level_information, vlm_client, configs, dbg):
    logs_root = configs.logs.root
    edge_level_information_path = os.path.join(logs_root, "edge_level_information.json")

    if os.path.exists(edge_level_information_path):
        with open(edge_level_information_path, "r", encoding="utf-8") as f:
            edge_level_information = json.load(f)
    else:
        edge_level_information = {}

    max_workers = int(getattr(configs.post_process, "max_workers", 8) or 8)
    write_lock = threading.Lock()
    edge_ids = list(nx_graph.edges())


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_edge_level_information,
                edge_id,
                node_level_information,
                nx_graph,
                vlm_client,
                logs_root, dbg
            ): edge_id
            for edge_id in edge_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="post-process edges"):
            edge_id = futures[future]
            try:
                result = future.result()
            except Exception:
                dbg.log(f"Failed post-process for edge {edge_id}", color="red")
                traceback.print_exc()
                continue

            if result is None:
                continue

            _, entry = result
            with write_lock:
                key = f"{edge_id[0]}|{edge_id[1]}"
                edge_level_information[key] = entry
                with open(edge_level_information_path, "w", encoding="utf-8") as f:
                    json.dump(edge_level_information, f, ensure_ascii=False, indent=4)

    return edge_level_information

def _process_path_intents(node_id, node_level_information, edge_level_information, nx_graph, vlm_client, logs_root, dbg):

    def prepare_path_information(
        path,
        node_level_information,
        edge_level_information,
    ):

        trajectory = []

        for i, node_id in enumerate(path):
            node_info = node_level_information.get(node_id, {}) or {}

            trajectory.append({
                "high_level": node_info.get("high_level", ""),
                "medium_level": node_info.get("medium_level", ""),
                "low_level": node_info.get("low_level", ""),
            })

            if i < len(path) - 1:
                edge_key = f"{path[i]}|{path[i + 1]}"
                edge_info = edge_level_information.get(edge_key, [])

                trajectory.append([
                    {
                        "low_level_action_description": item.get(
                            "low_level_action_description",
                            "",
                        ),
                        "high_level_action_description": item.get(
                            "high_level_action_description",
                            "",
                        ),
                    }
                    for item in edge_info
                    if isinstance(item, dict)
                ])

        return trajectory


    root_node_ids = [node_id for node_id, node in nx_graph.nodes(data=True) if node.get("is_root", False)]

    paths = []
    for root in root_node_ids:
        try:
            temp_paths = list(nx.all_shortest_paths(nx_graph, source=root, target=node_id))
            paths.extend(temp_paths)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    path_intents = {}
    for i, path in enumerate(paths):
        trajectory = prepare_path_information(path, node_level_information, edge_level_information)
        output = vlm_client.get_path_intents(trajectory)
        path_intents[str(i)] = output
        path_intents[str(i)]['node_sequence'] = path

    # now we use all of these to get top 3
    top_paths = vlm_client.get_top_paths(path_intents)

    # now we only keep path_intents for the top paths ids
    path_intents = {path_id: path_intents[path_id] for path_id in top_paths}

    return node_id, path_intents

def save_path_intents(nx_graph, node_level_information, edge_level_information, vlm_client, configs, dbg):
    logs_root = configs.logs.root
    path_intents_path = os.path.join(logs_root, "path_intents.json")

    if os.path.exists(path_intents_path):
        with open(path_intents_path, "r", encoding="utf-8") as f:
            path_intents = json.load(f)
    else:
        path_intents = {}

    max_workers = int(getattr(configs.post_process, "max_workers", 8) or 8)
    write_lock = threading.Lock()
    node_ids = list(nx_graph.nodes())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_path_intents,
                node_id,
                node_level_information,
                edge_level_information,
                nx_graph,
                vlm_client,
                logs_root,
                dbg
            ): node_id
            for node_id in node_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="post-process nodes"):
            node_id = futures[future]
            try:
                result = future.result()
            except Exception:
                dbg.log(f"Failed post-process for node {node_id}", color="red")
                traceback.print_exc()
                continue

            if result is None:
                continue

            _, entry = result
            with write_lock:
                path_intents[node_id] = entry
                with open(path_intents_path, "w", encoding="utf-8") as f:
                    json.dump(path_intents, f, ensure_ascii=False, indent=4)

    return path_intents


def save_user_intents(nx_graph, node_level_information, path_intents, vlm_client, configs, dbg):
    logs_root = configs.logs.root
    user_intents_path = os.path.join(logs_root, "user_intents.json")

    if os.path.exists(user_intents_path):
        with open(user_intents_path, "r", encoding="utf-8") as f:
            user_intents = json.load(f)
    else:
        user_intents = {}

    max_workers = int(getattr(configs.post_process, "max_workers", 8) or 8)
    write_lock = threading.Lock()
    node_ids = list(nx_graph.nodes())



    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_user_intents,
                node_id,
                node_level_information,
                path_intents,
                vlm_client,
                logs_root,
                dbg
            ): node_id
            for node_id in node_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="post-process nodes"):
            node_id = futures[future]
            try:
                result = future.result()
            except Exception:
                dbg.log(f"Failed post-process for node {node_id}", color="red")
                traceback.print_exc()
                continue

            if result is None:
                continue

            _, entry = result
            with write_lock:
                user_intents[node_id] = entry
                with open(user_intents_path, "w", encoding="utf-8") as f:
                    json.dump(user_intents, f, ensure_ascii=False, indent=4)

    return user_intents

def _process_user_intents(node_id, node_level_information, path_intents, vlm_client, logs_root, dbg):

    node_info = node_level_information[node_id]
    # Make a shallow copy of the path intents for the given node, removing the 'node_sequence' key if present
    node_path_intents = path_intents[node_id].copy()
    node_path_intents.pop("node_sequence", None)
    path_intents = node_path_intents

    out_edges = list(nx_graph.out_edges(node_id, data=True))
    out_edges = [
        data["description"]
        for source, target, data in nx_graph.out_edges(node_id, data=True)
        if source != target
        and str(data.get("type", "")).lower() not in ("scroll", "swipe")
        and data.get("description")
    ]

    user_intents = vlm_client.get_node_user_intents(node_info, path_intents, out_edges)
    
    return node_id, user_intents




def save_agent_thoughts(nx_graph, node_level_information, edge_level_information, user_intents, vlm_client, configs, dbg):
    logs_root = configs.logs.root
    agent_thoughts_path = os.path.join(logs_root, "agent_data.json")

    if os.path.exists(agent_thoughts_path):
        with open(agent_thoughts_path, "r", encoding="utf-8") as f:
            agent_thoughts = json.load(f)
    else:
        agent_thoughts = {}

    max_workers = int(getattr(configs.post_process, "max_workers", 8) or 8)
    write_lock = threading.Lock()
    node_ids = list(nx_graph.nodes())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_agent_thoughts,
                node_id,
                node_level_information,
                edge_level_information,
                user_intents,
                vlm_client,
                logs_root,
                dbg
            ): node_id
            for node_id in node_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="post-process nodes"):
            node_id = futures[future]
            try:
                result = future.result()
            except Exception:
                dbg.log(f"Failed post-process for node {node_id}", color="red")
                traceback.print_exc()
                continue

            if result is None:
                continue

            _, entry = result
            with write_lock:
                agent_thoughts[node_id] = entry
                with open(agent_thoughts_path, "w", encoding="utf-8") as f:
                    json.dump(agent_thoughts, f, ensure_ascii=False, indent=4)

    return agent_thoughts

def _process_agent_thoughts(node_id, node_level_information, edge_level_information, user_intents, vlm_client, logs_root, dbg):
    def prepare_path_information(
        path,
        node_level_information,
        edge_level_information):

        trajectory = []
        for i, node_id in enumerate(path):
            node_info = node_level_information.get(node_id, {}) or {}
            trajectory.append({
                "high_level": node_info.get("high_level", ""),
                "medium_level": node_info.get("medium_level", ""),
                "low_level": node_info.get("low_level", ""),
            })
            if i < len(path) - 1:
                edge_key = f"{path[i]}|{path[i + 1]}"
                edge_info = edge_level_information.get(edge_key, [])

                trajectory.append([
                    {
                        "low_level_action_description": item.get(
                            "low_level_action_description",
                            "",
                        ),
                        "high_level_action_description": item.get(
                            "high_level_action_description",
                            "",
                        ),
                    }
                    for item in edge_info
                    if isinstance(item, dict)
                ])

        return trajectory

    def trajectory_to_pairs(trajectory, path):
        pairs = []
        num_nodes_in_trajectory = (len(trajectory) + 1) // 2
        i = 0
        node_idx = 0

        while i < len(trajectory) - 2:
            n1 = trajectory[i]
            edge_block = trajectory[i + 1]
            n2 = trajectory[i + 2]

            edge = edge_block[0]

            node_id_1 = path[node_idx]
            node_id_2 = path[node_idx + 1]

            pairs.append((n1, edge, n2, node_id_1, node_id_2))

            i += 2
            node_idx += 1

        return pairs

    def make_mai_ui_data(user_intents_for_the_node, model_output, edge_data, screen_w, screen_h, scale=999, seed=None):
        import json
        import random

        rng = random.Random(seed)

        thoughts = model_output["agent_thoughts"]

        if len(user_intents_for_the_node) != len(thoughts):
            raise ValueError(
                f"Length mismatch: intents={len(user_intents_for_the_node)}, thoughts={len(thoughts)}"
            )

        if not edge_data:
            raise ValueError("edge_data is empty")

        edges = list(edge_data.values())
        data = []

        for intent, thought in zip(user_intents_for_the_node, thoughts):
            edge = rng.choice(edges)

            action_type = str(edge.get("type", "")).lower()

            if action_type in {"nav", "click", "tap", "button"}:
                action = "click"
            elif action_type in {"scroll", "swipe", "swipe_up"}:
                action = "swipe_up"
            elif action_type == "back":
                action = "back"
            elif action_type == "wait":
                action = "wait"
            else:
                action = "click"

            args = {"action": action}

            if action in {"click", "swipe_up"}:
                x1, y1, x2, y2 = edge["boundingBox"]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                x = round(cx / screen_w * scale)
                y = round(cy / screen_h * scale)

                x = max(0, min(scale, x))
                y = max(0, min(scale, y))

                args["coordinate"] = [x, y]

            tool_call = {
                "name": "mobile_use",
                "arguments": args,
            }

            agent_output = (
                f"<thinking>{thought}</thinking>\n"
                "<tool_call>\n"
                f"{json.dumps(tool_call, ensure_ascii=False)}\n"
                "</tool_call>"
            )

            data.append(
                {
                    "user_instruction": intent,
                    "agent_output": agent_output,
                }
            )

        return data


    root_node_ids = [node_id for node_id, node in nx_graph.nodes(data=True) if node.get("is_root", False)]

    paths = []
    for root in root_node_ids:
        try:
            temp_paths = list(nx.all_shortest_paths(nx_graph, source=root, target=node_id))
            paths.extend(temp_paths)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    node_data = {}
    path_intents = {}
    user_intents_for_the_node = user_intents[node_id]['user_intents']
    sample_img_to_get_size = cv2.imread(os.path.join(logs_root, "screenshots", f"{node_id}.jpg"))
    screen_h, screen_w = sample_img_to_get_size.shape[:2]
    for i, path in enumerate(paths):
        trajectory = prepare_path_information(path, node_level_information, edge_level_information)
        pairs = trajectory_to_pairs(trajectory, path)
        for pair in pairs:
            node_1_info, edge_info, node2_info, node_id_1, node_id_2 = pair
            output = vlm_client.get_agent_thought(node_1_info, edge_info, node2_info, user_intents_for_the_node)
            edge_data = nx_graph.get_edge_data(node_id_1, node_id_2)

            node_data[f"{node_id_1}|{node_id}"] = make_mai_ui_data(user_intents_for_the_node, output, edge_data, screen_w, screen_h)


    return node_id, node_data



def save_navigation_plans(nx_graph, node_level_information, edge_level_information, path_intents, user_intents, vlm_client, configs, dbg):

    logs_root = configs.logs.root
    node_navigation_plans_path = os.path.join(logs_root, "node_navigation_plans.json")

    if os.path.exists(node_navigation_plans_path):
        with open(node_navigation_plans_path, "r", encoding="utf-8") as f:
            node_navigation_plans = json.load(f)
    else:
        node_navigation_plans = {}

    max_workers = int(getattr(configs.post_process, "max_workers", 8) or 8)
    write_lock = threading.Lock()
    node_ids = list(nx_graph.nodes())


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_node_navigation_plans,
                node_id,
                node_level_information,
                edge_level_information,
                path_intents,
                user_intents,
                vlm_client,
                logs_root
            ): node_id
            for node_id in node_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="post-process nodes"):
            node_id = futures[future]
            try:
                result = future.result()
            except Exception:
                dbg.log(f"Failed post-process for node {node_id}", color="red")
                traceback.print_exc()
                continue

            if result is None:
                continue

            _, entry = result
            with write_lock:
                node_navigation_plans[node_id] = entry
                with open(node_navigation_plans_path, "w", encoding="utf-8") as f:
                    json.dump(node_navigation_plans, f, ensure_ascii=False, indent=4)

    return node_navigation_plans


def _process_node_navigation_plans(node_id, node_level_information, edge_level_information, path_intents, user_intents, vlm_client, logs_root):
    def prepare_navigation_plan_input(
    node_info,
    path_intents,
    node_level_information,
    edge_level_information):
        navigation_input = {
            "target_page": {
                "high_level": node_info.get("high_level", ""),
                "medium_level": node_info.get("medium_level", ""),
                "low_level": node_info.get("low_level", ""),
            },
            "paths": [],
        }

        for path_id, path_data in path_intents.items():
            node_sequence = path_data.get("node_sequence", [])

            path_pages = []
            transitions = []

            for seq_node_id in node_sequence:
                n = node_level_information.get(seq_node_id, {}) or {}
                path_pages.append({
                    "high_level": n.get("high_level", ""),
                    "medium_level": n.get("medium_level", ""),
                })

            for i in range(len(node_sequence) - 1):
                edge_key = f"{node_sequence[i]}|{node_sequence[i + 1]}"
                edge_info = edge_level_information.get(edge_key, [])

                transitions.append({
                    "alternative_actions": [
                        {
                            "low_level_action_description": a.get(
                                "low_level_action_description",
                                "",
                            ),
                            "high_level_action_description": a.get(
                                "high_level_action_description",
                                "",
                            ),
                        }
                        for a in edge_info
                        if isinstance(a, dict)
                    ],
                })

            navigation_input["paths"].append({
                "path_id": str(path_id),
                "path_user_goals": path_data.get("path_user_goals", []),
                "path_reliability_score": path_data.get(
                    "path_reliability",
                    {},
                ).get("score", 0),
                "num_pages": len(node_sequence),
                "num_transitions": max(0, len(node_sequence) - 1),
                "pages": path_pages,
                "transitions": transitions,
            })

        return navigation_input
    
    
    node_info = node_level_information[node_id]
    node_path_intents = path_intents[node_id]
    node_user_intents = user_intents[node_id]

    navigation_input = prepare_navigation_plan_input(
        node_info,
        node_path_intents,
        node_level_information,
        edge_level_information
    )

    navigation_plan = vlm_client.get_navigation_plan(navigation_input)

    return node_id, navigation_plan['ui_navigation_memory']

def add_node_embeddings_to_user_intents(
    root_path,
    node_level_information_filename="node_level_information.json",
    user_intents_filename="user_intents.json",
    output_filename="user_intents.json",
    bge_model_name="BAAI/bge-m3",
    batch_size=4,
    max_length=8192,
):
    node_level_information_path = os.path.join(root_path, node_level_information_filename)
    user_intents_path = os.path.join(root_path, user_intents_filename)
    output_path = os.path.join(root_path, output_filename)

    with open(user_intents_path, "r", encoding="utf-8") as f:
        user_intents_data = json.load(f)

    with open(node_level_information_path, "r", encoding="utf-8") as f:
        node_level_information = json.load(f)

    node_ids = []
    embedding_texts = []

    for node_id, node_data in user_intents_data.items():
        node_info = node_level_information[node_id]

        user_intents = node_data.get("user_intents", [])

        if isinstance(user_intents, dict):
            user_intents = user_intents.get("user_intents", [])

        if not isinstance(user_intents, list):
            user_intents = []

        user_intents = [
            str(intent).strip()
            for intent in user_intents
            if str(intent).strip()
        ]

        page_description = "\n".join([
            str(node_info.get("high_level", "")).strip(),
            str(node_info.get("medium_level", "")).strip(),
            str(node_info.get("low_level", "")).strip(),
        ]).strip()

        embedding_text = f"""
PAGE_DESCRIPTION:
{page_description}

USER_INTENTS:
{json.dumps(user_intents, ensure_ascii=False)}
""".strip()

        if not embedding_text:
            continue

        node_ids.append(node_id)
        embedding_texts.append(embedding_text)

    model = BGEM3FlagModel(
        bge_model_name,
        use_fp16=True,
    )

    outputs = model.encode(
        embedding_texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    embeddings = np.asarray(outputs["dense_vecs"], dtype=np.float32)

    embeddings = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-12,
    )

    for i, node_id in enumerate(node_ids):
        user_intents_data[node_id]["embedding_text"] = embedding_texts[i]
        user_intents_data[node_id]["embedding"] = embeddings[i].tolist()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(user_intents_data, f, ensure_ascii=False, indent=4)

    return {
        "num_nodes": len(node_ids),
        "output_path": output_path,
        "embedding_dim": int(embeddings.shape[1]),
    }



if __name__ == "__main__":
    dbg = Debugger(palette="soft", indent_size=2, width=90)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/clock_android.yaml")
    args = parser.parse_args()
    configs = Dynaconf(settings_files=[args.config], merge_enabled=True)
    configs = configs.default

    if configs.post_process.use_yibu_api:
        vlm_client = VLM_Yibu(configs, dbg)
    else:
        vlm_client = VLM(configs, dbg)


    image_utils = ImageUtils()

    with open(os.path.join(configs.logs.root, "graph.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    nx_graph = json_graph.node_link_graph(data, edges="links")


    dbg.log("Stage 1 starting: node_level_information.json", color="yellow")
    node_level_information = save_node_level_information(nx_graph, vlm_client, configs, dbg)
    dbg.log("Stage 1 complete: node_level_information.json", color="green")
    edge_level_information = save_edge_level_information(nx_graph, node_level_information, vlm_client, configs, dbg)
    dbg.log("Stage 2 complete: edge_level_information.json", color="green")
    dbg.log("Stage 3 starting: path_intents.json", color="yellow")
    path_intents = save_path_intents(nx_graph, node_level_information, edge_level_information, vlm_client, configs, dbg)
    dbg.log("Stage 3 complete: path_intents.json", color="green")
    dbg.log("Stage 4 starting: user_intents.json", color="yellow")
    user_intents = save_user_intents(nx_graph, node_level_information, path_intents, vlm_client, configs, dbg)
    dbg.log("Stage 4 complete: user_intents.json", color="green")
    dbg.log("Stage 5 starting: node_navigation_plans.json", color="yellow")
    node_navigation_plans = save_navigation_plans(nx_graph, node_level_information, edge_level_information, path_intents, user_intents, vlm_client, configs, dbg)
    dbg.log("Stage 5 complete: node_navigation_plans.json", color="green")
    dbg.log("Stage 6 User Intents Embedding starting: user_intents.json", color="yellow")
    add_node_embeddings_to_user_intents(configs.logs.root, "node_level_information.json", "user_intents.json", "user_intents.json", "BAAI/bge-m3", 4, 8192)
    dbg.log("Stage 6 User Intents Embedding complete: user_intents.json", color="green")


