from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from Graph.AppGraph import AppGraph
from Graph.Node import Node
from ImageUtils import ImageUtils
from Debugger import Debugger
from VLM import VLM
from layout_distance import get_android_xml_structure_distance


class Global_Localization:
    def __init__(self, app_graph: AppGraph, image_utils: ImageUtils, vlm_client: VLM, debugger: Debugger, configs: dict):
        self.app_graph = app_graph
        self.image_utils = image_utils
        self.debugger = debugger
        self.configs = configs
        self.vlm_client = vlm_client

    def get_clip_similarity_with_all_nodes(self, query_clip_embedding: np.ndarray) -> dict:
        """
        Returns a dict mapping node_id to similarity score.
        """
        node_ids = []
        clip_embeddings = []
        for node_id, node in self.app_graph.nodes.items():
            if node.clip_embedding is not None:
                node_ids.append(node.node_id)
                clip_embeddings.append(node.clip_embedding)

        if not clip_embeddings:
            return {}

        clip_embedding_stack = np.vstack(clip_embeddings)
        if query_clip_embedding.ndim == 1:
            query_clip_embedding = query_clip_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_clip_embedding, clip_embedding_stack)[0]  # shape: (N,)

        clip_similarity_dict = {node_id: sim for node_id, sim in zip(node_ids, similarities)}
        return clip_similarity_dict

    def get_page_purpose_embedding_similarity_with_all_nodes(self, query_page_purpose_embedding: np.ndarray) -> dict:
        """
        Returns a dict mapping node_id to similarity score.
        """
        node_ids = []
        page_purpose_embeddings = []
        for node_id, node in self.app_graph.nodes.items():
            if node.page_purpose_embedding is not None:
                node_ids.append(node.node_id)
                page_purpose_embeddings.append(node.page_purpose_embedding)

        if not page_purpose_embeddings:
            return {}

        page_purpose_embedding_stack = np.vstack(page_purpose_embeddings)
        if query_page_purpose_embedding.ndim == 1:
            query_page_purpose_embedding = query_page_purpose_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_page_purpose_embedding, page_purpose_embedding_stack)[0]  # shape: (N,)

        page_purpose_embedding_similarity_dict = {node_id: sim for node_id, sim in zip(node_ids, similarities)}
        return page_purpose_embedding_similarity_dict
   

    def get_xml_structure_distance_with_all_nodes(self, query_xml_layout: str) -> dict:
        xml_structure_distance_dict = {}
        for node_id, node in self.app_graph.nodes.items():
            if node.xml_layout is not None:
                xml_structure_distance = get_android_xml_structure_distance(node.xml_layout, query_xml_layout)['distance']
                xml_structure_distance_dict[node_id] = xml_structure_distance
        return xml_structure_distance_dict


    def find_node_visual_emphasize(self, clip_similarity_dict, xml_similarity_dict, clip_similarity_threshold, xml_dist_threshold) -> tuple[str, float]:

        candidates = [
            (nid, csim)
            for nid, csim in clip_similarity_dict.items()
            if csim >= clip_similarity_threshold and xml_similarity_dict.get(nid, 1) <= xml_dist_threshold
        ]
        if candidates:
            best_id, best_sim = max(candidates, key=lambda x: x[1])
            return best_id, best_sim
        return None, 0.0


    def find_node_structure_emphasize(self, clip_similarity_dict, xml_similarity_dict, clip_similarity_threshold, xml_dist_threshold, exclusion_list=None) -> tuple[str, float, float]:
        candidates = [
            (nid, csim, xml_similarity_dict.get(nid, 1))
            for nid, csim in clip_similarity_dict.items()
            if csim >= clip_similarity_threshold and xml_similarity_dict.get(nid, 1) <= xml_dist_threshold
        ]
        if exclusion_list:
            candidates = [
                (nid, csim, xml_similarity_dict.get(nid, 1))
                for nid, csim, _xml in candidates
                if nid not in exclusion_list
            ]

        if candidates:
            # Maximize by clip similarity
            best_id, best_sim, best_xml_dist = max(candidates, key=lambda x: x[1])
            return best_id, best_sim, best_xml_dist
        return None, 0.0, 1.0
 
    def find_node_path_emphasize(
        self,
        clip_similarity_dict: dict,
        xml_similarity_dict: dict,
        clip_similarity_threshold: float,
        xml_dist_threshold: float,
        path_node_history: list[Node],
        path_action_history: list[dict],
    ) -> tuple[str | None, float]:
        nodes = list(path_node_history)
        actions = list(path_action_history)
        i = 1
        while i < len(nodes):
            if nodes[i].node_id == nodes[i - 1].node_id:
                nodes.pop(i)
                if i - 1 < len(actions):
                    actions.pop(i - 1)
                continue
            i += 1
        if len(nodes) >= 2:
            tail = nodes[-1].node_id
            for j in range(len(nodes) - 2, -1, -1):
                if nodes[j].node_id == tail:
                    nodes = nodes[j + 1 :]
                    actions = actions[j + 1 :]
                    if len(nodes) and len(actions) >= len(nodes):
                        actions = actions[: len(nodes) - 1]
                    break

        want = [str(a.get("description") or "") for a in actions]
        if not want:
            return None, 0.0

        reachable = {nid for nid, n in self.app_graph.nodes.items() if n.is_root} or set(self.app_graph.nx_graph.nodes)
        for desc in want:
            nxt = set()
            for u in reachable:
                for _, v, d in self.app_graph.nx_graph.edges(u, data=True):
                    if str(d.get("description") or "") == desc:
                        nxt.add(v)
            reachable = nxt

        if not reachable:
            return None, 0.0

        candidates = [
            (nid, clip_similarity_dict.get(nid, 0.0), xml_similarity_dict.get(nid, 1))
            for nid in reachable
            if clip_similarity_dict.get(nid, 0.0) >= clip_similarity_threshold
            and xml_similarity_dict.get(nid, 1) <= xml_dist_threshold
        ]
        if not candidates:
            self.debugger.log(
                f"Path-matched {reachable} but none pass clip>={clip_similarity_threshold} xml<={xml_dist_threshold}",
                color="gray",
            )
            return None, 0.0

        best_id, best_sim, best_xml = max(candidates, key=lambda x: x[1])
        self.debugger.log(
            f"Path-matched {reachable} -> {best_id} (clip {best_sim}, xml {best_xml})",
            color="gray",
        )
        return best_id, float(best_sim)

    def create_exclusion_list_for_siblings_targets(self, parent_node: Node, last_action: dict) -> list[str]:
        last_desc = str(last_action.get("description") or "")
        siblings = []
        for _, v, data in self.app_graph.nx_graph.edges(parent_node.node_id, data=True):
            if str(data.get("description") or "") != last_desc:
                siblings.append(v)
        return siblings

    def get_shortest_path_from_roots(self, target_node_id: str) -> dict:
        """
        Shortest simple path from any root to ``target_node_id``.
        Returns edge action descriptions along the path and the target node's page_purpose.
        """
        root_ids = [nid for nid, n in self.app_graph.nodes.items() if n.is_root]
        if not root_ids:
            root_ids = [
                n
                for n in self.app_graph.nx_graph.nodes
                if self.app_graph.nx_graph.in_degree(n) == 0
            ]

        best_path: list[str] | None = None
        for root_id in root_ids:
            if root_id == target_node_id:
                best_path = [root_id]
                break
            try:
                path = nx.shortest_path(self.app_graph.nx_graph, root_id, target_node_id)
            except nx.NetworkXNoPath:
                continue
            if best_path is None or len(path) < len(best_path):
                best_path = path

        if best_path is None:
            return {"action_descriptions": [], "page_purpose": None, "path_node_ids": []}

        action_descriptions: list[str] = []
        for i in range(len(best_path) - 1):
            u, v = best_path[i], best_path[i + 1]
            edge_data = self.app_graph.nx_graph.get_edge_data(u, v)
            if not edge_data:
                action_descriptions.append("")
                continue
            first_edge = next(iter(edge_data.values()))
            action_descriptions.append(str(first_edge.get("description") or ""))

        node = self.app_graph.nodes.get(target_node_id)
        page_purpose = node.page_purpose if node is not None else None

        return {
            "action_descriptions": action_descriptions,
            "page_purpose": page_purpose,
            "path_node_ids": best_path,
        }

    def find_node_page_purpose_emphasize(
        self,
        query_page_summary_payload: dict,
        path_action_history: list[dict], path_node_history: list[Node]
    ) -> tuple[str | None, dict]:
        cfg = self.configs.graph.global_localization.find_node_page_purpose_emphasize
        sim_thresh = cfg.page_purpose_similarity_threshold
        max_k = cfg.max_num_candidates_for_comparison

        page_purpose_embedding = self.image_utils.text_model_embedder.encode(
            query_page_summary_payload.get("page_purpose", ""), max_length=1024
        )["dense_vecs"]
        active_tab = query_page_summary_payload.get("active_tab", "")
        active_subtab = query_page_summary_payload.get("active_subtab", "")
        page_purpose_embedding_similarity_dict = self.get_page_purpose_embedding_similarity_with_all_nodes(
            page_purpose_embedding
        )

        exclude_id = path_node_history[-1].node_id if path_node_history else None  # we exclude the parent node (to prevent merging parent child node because of similarity of their path the LLM most of the times wants to merge them)


        same_tab_subtab = {
            node_id: sim
            for node_id, sim in page_purpose_embedding_similarity_dict.items()
            if (sim >= sim_thresh) and (node_id != exclude_id)
            and active_tab == self.app_graph.nodes[node_id].active_tab
            and active_subtab == self.app_graph.nodes[node_id].active_subtab
        }
        top_k = sorted(same_tab_subtab.items(), key=lambda x: x[1], reverse=True)[:max_k]

        


        candidates = {}
        for node_id, _purpose_sim in top_k:
            path_info = self.get_shortest_path_from_roots(node_id)
            candidates[node_id] = {
                "page_purpose": self.app_graph.nodes[node_id].page_purpose,
                "action_path": path_info["action_descriptions"],
            }

        if not candidates:
            return None, {}

        query_action_history_descriptions = [
            str(action.get("description") or "") for action in path_action_history
        ]
        out = self.vlm_client.compare_navigtional_purpose(
            query_page_summary_payload.get("page_purpose", ""),
            query_action_history_descriptions,
            candidates,
        )
        node_id = out.get("matched_node_id") if out.get("same_purpose") else None
        return node_id, out


    def find_node(self, query_screenshot: np.ndarray, query_xml_layout, configs: dict, path_node_history: list[Node], path_action_history: list[dict], include_path_emphasize=True) -> tuple[str, float]:
        

        query_clip_embedding = self.image_utils.extract_clip_embedding(query_screenshot)

        query_clip_similarity_dict = self.get_clip_similarity_with_all_nodes(query_clip_embedding)
        query_xml_structure_distance_dict = self.get_xml_structure_distance_with_all_nodes(query_xml_layout)
        
        with self.debugger.time_block("Find node visual emphasize", color="gray"):
            cthresh = configs.graph.global_localization.find_node_visual_emphasize.clip_embedding_similarity_threshold
            xthresh = configs.graph.global_localization.find_node_visual_emphasize.xml_distance_threshold
            node_id, similarity = self.find_node_visual_emphasize(query_clip_similarity_dict, query_xml_structure_distance_dict, cthresh, xthresh)
            if node_id is not None:
                self.debugger.log(f"Found node (visual_emphasize) {node_id} with similarity {similarity}", color="green")
                return {"node_id": node_id, "query_clip_similarity_dict": query_clip_similarity_dict, "query_xml_structure_distance_dict": query_xml_structure_distance_dict} 
        

        if path_node_history and path_action_history:
            exclusion_list = self.create_exclusion_list_for_siblings_targets(
                path_node_history[-1], path_action_history[-1]
            )
        else:
            exclusion_list = []
        # in second stage we use structure similarity if the last action is swipe or scroll we use higher clip similarity threshold
        cthresh = configs.graph.global_localization.find_node_structure_emphasize.clip_embedding_similarity_threshold
        xthresh = configs.graph.global_localization.find_node_structure_emphasize.xml_distance_threshold
        if path_action_history and (path_action_history[-1]['type'] in ('swipe', 'scroll')):
            cthresh = configs.graph.global_localization.find_node_structure_emphasize.clip_similarity_threshold_if_previous_action_is_swipe_or_scroll
           

        with self.debugger.time_block("Find node structure emphasize", color="gray"):
            node_id, clip_similarity, xml_distance = self.find_node_structure_emphasize(query_clip_similarity_dict, query_xml_structure_distance_dict, cthresh, xthresh, exclusion_list)
            if node_id is not None:
                self.debugger.log(f"Found node (structure_emphasize) {node_id} with clip similarity {clip_similarity} and xml distance {xml_distance}", color="green")
                return {"node_id": node_id, "query_clip_similarity_dict": query_clip_similarity_dict, "query_xml_structure_distance_dict": query_xml_structure_distance_dict} 

        if include_path_emphasize:
            with self.debugger.time_block("Find node path emphasize", color="cyan"):
                pthresh = configs.graph.global_localization.find_node_path_emphasize.clip_embedding_similarity_threshold
                pxthresh = configs.graph.global_localization.find_node_path_emphasize.xml_distance_threshold
                node_id, similarity = self.find_node_path_emphasize(
                    query_clip_similarity_dict,
                    query_xml_structure_distance_dict,
                    pthresh,
                    pxthresh,
                    path_node_history,
                    path_action_history,
                )
                if node_id is not None:
                    self.debugger.log(
                        f"Found node (path_emphasize) {node_id} with clip similarity {similarity}",
                        color="green",
                    )
                    return {
                        "node_id": node_id,
                        "query_clip_similarity_dict": query_clip_similarity_dict,
                        "query_xml_structure_distance_dict": query_xml_structure_distance_dict,
                    }

        with self.debugger.time_block("Extracting Page Summary", color="gray"):
            page_summary_payload = self.vlm_client.extract_page_summary(
                query_screenshot,
                application_description=self.configs.app.description,
                action_history=path_action_history,
            )

        with self.debugger.time_block("Find node page purpose emphasize", color="gray"):
            node_id, purpose_match = self.find_node_page_purpose_emphasize(
                page_summary_payload, path_action_history, path_node_history
            )
            if node_id is not None:
                self.debugger.log(
                    f"Found node (page_purpose_emphasize) {node_id} {purpose_match}",
                    color="purple",
                )
                return {
                    "node_id": node_id,
                    "query_clip_similarity_dict": query_clip_similarity_dict,
                    "query_xml_structure_distance_dict": query_xml_structure_distance_dict,
                    "page_purpose_match": purpose_match,
                    "page_summary_payload": page_summary_payload
                }


        return {
            "node_id": None,
            "page_summary_payload": page_summary_payload,
            "query_clip_similarity_dict": query_clip_similarity_dict,
            "query_xml_structure_distance_dict": query_xml_structure_distance_dict,
        }
