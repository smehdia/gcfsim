from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

import cv2
import networkx as nx

from Graph.AppGraph import AppGraph
from Graph.Global_Localization import Global_Localization
from ImageUtils import ImageUtils
from Debugger import Debugger
from Driver.BaseDriver import BaseDriver
from Agents.BaseAgent import BaseAgent
from Agents.MAI_UI.MAI_UI import ParsedAction
from VLM import VLM
from layout_distance import get_android_xml_structure_distance


class BackTrack:
    def __init__(self, app_graph: AppGraph, image_utils: ImageUtils, debugger: Debugger, driver: BaseDriver, agent: BaseAgent, vlm_client: VLM, global_localization: Global_Localization, configs: dict):
        self.app_graph = app_graph
        self.image_utils = image_utils
        self.debugger = debugger
        self.driver = driver
        self.agent = agent
        self.vlm_client = vlm_client
        self.global_localization = global_localization
        self.configs = configs


    def get_shortest_path_from_root_node(self, target_node):
        g = self.app_graph.nx_graph
        tid = str(getattr(target_node, "node_id", None) or target_node)
        node = self.app_graph.nodes.get(tid) or target_node
        last_page_summary = str(getattr(node, "page_summary", "") or "")
        if tid in g:
            last_page_summary = str(g.nodes[tid].get("page_summary", "") or last_page_summary)

        roots = [nid for nid, n in self.app_graph.nodes.items() if getattr(n, "is_root", False)]
        if not roots:
            roots = [nid for nid in g.nodes if g.in_degree(nid) == 0]

        best_path = None
        for r in roots:
            rs = str(r)
            if rs == tid:
                return last_page_summary, []
            try:
                p = nx.shortest_path(g, source=rs, target=tid)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if best_path is None or len(p) < len(best_path):
                best_path = p

        if not best_path or len(best_path) < 2:
            return last_page_summary, []

        actions = []
        for i in range(len(best_path) - 1):
            u, v = best_path[i], best_path[i + 1]
            ed = g.get_edge_data(u, v) or {}
            action_info = None
            for d in ed.values():
                if not isinstance(d, dict):
                    continue
                action_info = {
                    "type": str(d.get("type", "") or "").strip(),
                    "description": str(d.get("description", "") or "").strip(),
                    "boundingBox": d.get("boundingBox", None)
                }
                break  # Take the first matching
            if action_info is None:
                action_info = {
                    "type": "",
                    "description": "",
                    "boundingBox": None
                }
            actions.append(action_info)

        return last_page_summary, actions


    def get_back_to_previous_node(self, current_node, previous_node, forward_action_element, screenshot):
        def parsed_action_from_dict(d: dict) -> ParsedAction:
            def _coords_to_tuples(d):
                if not d:
                    return {}
                out = {}
                for k, v in d.items():
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        out[k] = (int(round(v[0])), int(round(v[1])))
                return out
            return ParsedAction(
                raw_text=str(d.get("raw_text") or ""),
                thought=str(d.get("thought") or ""),
                action_type=str(d.get("action_type") or "unknown"),
                params=dict(d.get("params") or {}),
                orig_coords=_coords_to_tuples(d.get("orig_coords")),
                sent_coords=_coords_to_tuples(d.get("sent_coords")),
            )

        parent_screenshot = cv2.imread(os.path.join(self.configs.logs.root, "screenshots", f"{previous_node.node_id}.jpg"))
        
        # if we are in BFS node we don't need to guess undo action
        if current_node.node_id == previous_node.node_id:
            return True
        elif current_node.backtracking_action is not None:
            undo = current_node.backtracking_action
            if isinstance(undo, dict):
                undo_action_standardized = parsed_action_from_dict(undo)
            elif not isinstance(undo, ParsedAction):
                undo_action_standardized = None
            else:
                undo_action_standardized = undo

        else:
            with self.debugger.time_block("Guess undo action with VLM", color="gray"):
                try:
                    vlm_out, undo_action_standardized = self.vlm_client.guess_undo_action(screenshot, parent_screenshot, forward_action_element)
                except Exception as e:
                    self.debugger.log(f"Error guessing undo action with VLM: {e}", color="red")
                    undo_action_standardized = None
                    return False


        if undo_action_standardized is not None:
            self.driver.execute_action(undo_action_standardized)
            current_screenshot = self.driver.get_non_loading_page()
            current_xml_layout = self.driver.get_xml_layout()
            find_result = self.global_localization.find_node(current_screenshot, current_xml_layout, self.configs, [current_node], [forward_action_element])
            clip_similarity_dict = find_result["query_clip_similarity_dict"]
            xml_structure_distance_dict = find_result["query_xml_structure_distance_dict"]
            
            if clip_similarity_dict[previous_node.node_id] >= self.configs.graph.exploration_modes.local_bfs.backtracking_with_undo_clip_similarity_threshold and xml_structure_distance_dict[previous_node.node_id] <= self.configs.graph.exploration_modes.local_bfs.backtracking_with_undo_xml_structure_distance_threshold:
                self.debugger.log(f"We are back in the parent node {previous_node.node_id}", color="green")
                current_node.backtracking_action = undo_action_standardized
                return True
            else:
                # in the fail case we remove backtracking action from the current node
                self.debugger.log(f"We are not back in the parent node {previous_node.node_id}", color="red")
                current_node.backtracking_action = None
                return False
        else:
            return False

    def get_action_history_and_node_history_from_root_node(self, target_node):
        g = self.app_graph.nx_graph
        tid = str(getattr(target_node, "node_id", None) or target_node)
        node = self.app_graph.nodes.get(tid) or target_node

        roots = [nid for nid, n in self.app_graph.nodes.items() if getattr(n, "is_root", False)]
        if not roots:
            roots = [nid for nid in g.nodes if g.in_degree(nid) == 0]

        best_path = None
        for r in roots:
            rs = str(r)
            if rs == tid:
                return [node], []
            try:
                p = nx.shortest_path(g, source=rs, target=tid)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if best_path is None or len(p) < len(best_path):
                best_path = p

        if not best_path or len(best_path) < 2:
            return [node], []

        path_node_history = [self.app_graph.nodes[str(nid)] for nid in best_path]
        path_action_history = []
        for i in range(len(best_path) - 1):
            u, v = best_path[i], best_path[i + 1]
            ed = g.get_edge_data(u, v) or {}
            action_info = None
            for d in ed.values():
                if not isinstance(d, dict):
                    continue
                action_info = {
                    "type": str(d.get("type", "") or "").strip(),
                    "description": str(d.get("description", "") or "").strip(),
                    "boundingBox": d.get("boundingBox", None),
                }
                break
            if action_info is None:
                action_info = {"type": "", "description": "", "boundingBox": None}
            path_action_history.append(action_info)

        return path_node_history, path_action_history

    def backtrack_using_agent(self, target_node):
        """
        Replay the shortest path from the root node to the given target_node using the agent.

        Args:
            target_node: The node to backtrack to (rewind to).

        Returns:
            True if backtracking with the agent returns to the target node,
            False otherwise.
        """
        self.driver.reset_to_start_page()
        _page_summary, path_actions = self.get_shortest_path_from_root_node(target_node)
        if not path_actions:
            return False

        self.agent.clear_history()
        w, h = self.driver.get_screen_size()
        center = (w // 2, h // 2)

        for action_info in path_actions:
            if not isinstance(action_info, dict):
                continue
            el_type = str(action_info.get("type", "") or "").strip().lower()
            desc_str = str(action_info.get("description", "") or "").strip()
            bb = action_info.get("boundingBox") or action_info.get("bbox")
            if bb and len(bb) >= 4:
                x1, y1, x2, y2 = (int(round(float(v))) for v in bb[:4])
                point = ((x1 + x2) // 2, (y1 + y2) // 2)
            else:
                point = center

            if "swipe" in el_type:
                parsed = ParsedAction(
                    raw_text=desc_str,
                    thought=desc_str,
                    action_type="scroll",
                    params={"direction": "left"},
                    orig_coords={"point": point},
                )
            elif "scroll" in el_type:
                parsed = ParsedAction(
                    raw_text=desc_str,
                    thought=desc_str,
                    action_type="scroll",
                    params={"direction": "down"},
                    orig_coords={"point": point},
                )
            else:
                label = desc_str or el_type or "element"
                prompt = (
                    f"Locate element with the following description:\n{label}\n\n"
                    "If it is not found return Click(0, 0)."
                )
                agent_out = self.agent.grounding_action(label, self.driver.take_screenshot())
                parsed = agent_out[0] if isinstance(agent_out, tuple) else agent_out
                if isinstance(parsed, tuple):
                    parsed = parsed[0]
                if not hasattr(parsed, "action_type"):
                    return False
                pt = (getattr(parsed, "orig_coords", None) or {}).get("point")
                if pt is not None and len(pt) >= 2 and int(round(pt[0])) == 0 and int(round(pt[1])) == 0:
                    return False
                if str(getattr(parsed, "action_type", "") or "").strip().lower() == "unknown":
                    return False

            self.driver.execute_action(parsed)
            self.driver.wait()

        # After replaying the actions, check if we have arrived at the target node.
        current_screenshot = self.driver.get_non_loading_page()
        current_xml_layout = self.driver.get_xml_layout()
        find_result = self.global_localization.find_node(
            current_screenshot,
            current_xml_layout,
            self.configs,
            [target_node],    # path_node_history: [target_node]
            [],              # path_action_history: []
        )
        clip_similarity_dict = find_result["query_clip_similarity_dict"]
        xml_structure_distance_dict = find_result["query_xml_structure_distance_dict"]
        pid = target_node.node_id
        clip_thr = self.configs.graph.exploration_modes.local_bfs.backtracking_clip_similarity_threshold_with_agent
        xml_thr = self.configs.graph.exploration_modes.local_bfs.backtracking_xml_structure_distance_threshold_with_agent
        if (
            clip_similarity_dict.get(pid, 0.0) >= clip_thr
            and xml_structure_distance_dict.get(pid, float("inf")) <= xml_thr
        ):
            self.debugger.log(f"We are back in the target node {pid} (agent replay)", color="green")
            return True
        self.debugger.log(f"We are not back in the target node {pid} (agent replay)", color="red")
        return False
