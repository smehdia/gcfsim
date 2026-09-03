from __future__ import annotations

import random
import cv2
import os

from Graph.AppGraph import AppGraph
from ImageUtils import ImageUtils
from Debugger import Debugger
from Driver.BaseDriver import BaseDriver
from VLM import VLM
from Agents.BaseAgent import BaseAgent

from Graph.Global_Localization import Global_Localization
from Graph.Node import Node
from Agents.MAI_UI.MAI_UI import ParsedAction
from sklearn.metrics.pairwise import cosine_similarity
from Graph.BackTrack import BackTrack

class Walker:
    def __init__(self, app_graph: AppGraph, image_utils: ImageUtils, debugger: Debugger, driver: BaseDriver, vlm_client: VLM, agent: BaseAgent, configs: dict, walk_counter: int):
        self.app_graph = app_graph
        self.image_utils = image_utils
        self.debugger = debugger
        self.driver = driver
        self.vlm_client = vlm_client
        self.agent = agent
        self.global_localization = Global_Localization(app_graph, image_utils, vlm_client, debugger, configs)
        self.configs = configs
        self.walk_counter = walk_counter
        self.backtracker = BackTrack(app_graph, image_utils, debugger, driver, agent, vlm_client, self.global_localization, configs)

    @staticmethod
    def ui_element_to_parsed_action(ui_element: dict) -> ParsedAction:
        bb = ui_element.get("boundingBox") or ui_element.get("bbox")
        if not bb or len(bb) < 4:
            return ParsedAction(raw_text="", thought="", action_type="unknown")
        x1, y1, x2, y2 = (int(round(v)) for v in bb[:4])
        point = ((x1 + x2) // 2, (y1 + y2) // 2)
        desc = str(ui_element.get("description") or "")
        el_type = str(ui_element.get("type") or "").strip().lower()
        if el_type == "scroll":
            return ParsedAction(
                raw_text=desc, thought=desc, action_type="scroll",
                params={"direction": "down"}, orig_coords={"point": point},
            )
        if el_type == "swipe":
            return ParsedAction(
                raw_text=desc, thought=desc, action_type="scroll",
                params={"direction": "left"}, orig_coords={"point": point},
            )
        return ParsedAction(
            raw_text=desc, thought=desc, action_type="click", orig_coords={"point": point},
        )

    def _log_selected_action(self, ui_element: dict, parsed_action: ParsedAction) -> None:
        self.debugger.log(
            f"Selected action: type={ui_element.get('type')} | "
            f"{ui_element.get('description')} | bbox={ui_element.get('boundingBox')} | "
            f"exec={parsed_action.action_type} {parsed_action.params or {}} {parsed_action.orig_coords or {}}",
            color="cyan",
        )


    def resolve_current_screen(self, screenshot, xml_layout, path_node_history, path_action_history, include_path_emphasize=True):
        with self.debugger.time_block("Find node in the graph", color="gray"):
            find_node_result = self.global_localization.find_node(screenshot, xml_layout, self.configs, path_node_history, path_action_history, include_path_emphasize)
            node_id = find_node_result["node_id"]
            page_summary_payload = find_node_result.get("page_summary_payload", None)

        if node_id is None:
            current_node_id = self.app_graph.generate_new_id()
            root_flag = len(path_node_history) == 0
            

            with self.debugger.time_block(f"Create new node {current_node_id}", color="green"):
                clip_embedding = self.image_utils.extract_clip_embedding(screenshot)
                current_node = Node(
                    clip_embedding=clip_embedding,
                    screenshot=screenshot,
                    node_id=current_node_id,
                    is_root=root_flag,
                    node_history=path_node_history,
                    action_history=path_action_history,
                    screen_type="normal",
                    xml_layout=xml_layout,
                    configs=self.configs,
                    debugger=self.debugger,
                    vlm_client=self.vlm_client,
                    image_utils=self.image_utils,
                    page_summary_payload=page_summary_payload,
                )
                self.app_graph.add_node(current_node)

        else:
            self.app_graph.nodes[node_id].num_visits += 1
            if (self.app_graph.nodes[node_id].num_refreshed_elements < self.configs.graph.max_refresh_elements_on_each_node_visit) and (self.app_graph.nodes[node_id].num_visits >= self.configs.graph.refresh_elements_after_at_least_nodes_for_k_times):
                with self.debugger.time_block(f"Refresh elements on node {node_id}", color="blue"):
                    self.app_graph.nodes[node_id].refresh_elements(cv2.imread(os.path.join(self.configs.logs.root, "screenshots", f"{node_id}.jpg")), self.vlm_client, self.configs, self.image_utils, action_history=path_action_history)
                    self.app_graph.nodes[node_id].num_refreshed_elements += 1

            self.app_graph.sync_node(self.app_graph.nodes[node_id])
            current_node = self.app_graph.nodes[node_id]

        if (len(path_node_history) > 0) and (len(path_action_history) > 0):
            self.app_graph.add_edge(path_node_history[-1], current_node, action=path_action_history[-1])


        return current_node


    def select_action_random_walk_policy(self, current_node: Node, screenshot, last_step=False):
        # note that we use screenshot to make sure the selected action is present in the screenshot
        # currenet node only retains one screenshot so in case where selected action does not match the screenshot action we should not perform it
        # to verify this we use CLIP similarity between selected action and that region in the screenshot (excluding swipe or scroll actions)

        if not last_step:
            # select a random action from the node
            if len(current_node.ui_elements) == 0:
                return None, None
            ui_element = random.choice(current_node.ui_elements)

            if ui_element["type"] not in ("swipe", "scroll"):
                # verify if the action is present in the screenshot
                bbox_element = ui_element["boundingBox"]
                x1, y1, x2, y2 = bbox_element
                clip_embedding = self.image_utils.extract_clip_embedding(screenshot[y1:y2, x1:x2])
                clip_similarity = cosine_similarity(ui_element["clip_embedding"].reshape(1, -1), clip_embedding.reshape(1, -1))[0][0]

                if clip_similarity < self.configs.graph.action_matching_clip_similarity_threshold:
                    self.debugger.log(f"Action {ui_element['description']} not present in the screenshot — terminating walk", color="red")
                    return None, None


            ui_element["explored"] = True
            parsed_action = self.ui_element_to_parsed_action(ui_element)
            return ui_element, parsed_action
        else:
            # we first extract unexplored actions and select among them
            unexplored_actions = [action for action in current_node.ui_elements if not action.get("explored")]
            if not unexplored_actions:
                return None, None

            ui_element = random.choice(unexplored_actions)


            if ui_element["type"] not in ("swipe", "scroll"):
                # verify if the action is present in the screenshot
                bbox_element = ui_element["boundingBox"]
                x1, y1, x2, y2 = bbox_element
                clip_embedding = self.image_utils.extract_clip_embedding(screenshot[y1:y2, x1:x2])
                clip_similarity = cosine_similarity(ui_element["clip_embedding"].reshape(1, -1), clip_embedding.reshape(1, -1))[0][0]

                if clip_similarity < self.configs.graph.action_matching_clip_similarity_threshold:
                    self.debugger.log(f"Action {ui_element['description']} not present in the screenshot — terminating walk", color="red")
                    return None, None

            ui_element["explored"] = True
            parsed_action = self.ui_element_to_parsed_action(ui_element)
            return ui_element, parsed_action


    def select_action_local_bfs_policy(self, current_node: Node, screenshot):

        # note that we use screenshot to make sure the selected action is present in the screenshot
        # currenet node only retains one screenshot so in case where selected action does not match the screenshot action we should not perform it
        # to verify this we use CLIP similarity between selected action and that region in the screenshot (excluding swipe or scroll actions)

        unexplored_actions = [action for action in current_node.ui_elements if not action.get("explored")]
        if not unexplored_actions:
            return None, None

        # First, prioritize unexplored scroll or swipe actions
        prioritized_actions = [
            action for action in unexplored_actions 
            if action.get("type", "").lower() in ("scroll", "swipe")
        ]


        if prioritized_actions:
            ui_element = random.choice(prioritized_actions)
        else:
            ui_element = random.choice(unexplored_actions)
        
        if ui_element["type"] not in ("swipe", "scroll"):
            # verify if the action is present in the screenshot
            bbox_element = ui_element["boundingBox"]
            x1, y1, x2, y2 = bbox_element
            clip_embedding = self.image_utils.extract_clip_embedding(screenshot[y1:y2, x1:x2])
            clip_similarity = cosine_similarity(ui_element["clip_embedding"].reshape(1, -1), clip_embedding.reshape(1, -1))[0][0]

            if clip_similarity < self.configs.graph.action_matching_clip_similarity_threshold:
                self.debugger.log(f"Action {ui_element['description']} not present in the screenshot — terminating walk", color="red")
                return None, None


        ui_element["explored"] = True
        parsed_action = self.ui_element_to_parsed_action(ui_element)
        return ui_element, parsed_action
 

    def save_debug_path(self, path_action_history, debug_path_screenshot_history, walk_counter, mode, step=None):
        if debug_path_screenshot_history:
            os.makedirs(os.path.join(self.configs.logs.root, 'debug_paths'), exist_ok=True)
            actions = path_action_history
            imgs = debug_path_screenshot_history
            vis = [
                self.image_utils.visualize_action_on_screenshot(imgs[i], actions[i] if i < len(actions) else None)
                for i in range(len(imgs))
            ]
            minw = min(img.shape[1] for img in vis)
            vis = [cv2.resize(img, (minw, int(img.shape[0] * minw / img.shape[1]))) for img in vis]
            if step is None:
                out_path = os.path.join(self.configs.logs.root, 'debug_paths',f"debug_path_screenshot_history_{self.walk_counter}_{mode}.jpg")
            else:
                out_path = os.path.join(self.configs.logs.root, 'debug_paths',f"debug_path_screenshot_history_{self.walk_counter}_{mode}_{step}.jpg")
            cv2.imwrite(out_path, cv2.hconcat(vis))


    def random_walk(self, max_num_actions, bypass_last_step_action_policy_for_bfs=False, by_pass_increase_walk_counter=False, save_debug_path=False):

        # pipeline:
        # loop max_num_actions times:
            # 1. reset to start page
            # 2. get non loading page
            # 3. find node based in graph 
            # 4. create new node if not found
            # 5. select a random UI element
            # 6. execute the action
            # NOTE THAT on the last action since depth is controlled we want to select an unexplored action (previous actions could be repetetive though)
        # in BFS mode we don't want to select unexplored action to do BFS (in BFS N-1 actions can be repetetive)


        if not by_pass_increase_walk_counter:
            self.app_graph.num_walks += 1
            self.walk_counter += 1
        

        walk_completed = True
        with self.debugger.time_block("Random walk", color="gray"):
            with self.debugger.time_block("Reset to start page", color="gray"):
                self.driver.reset_to_start_page()

            path_node_history = []
            path_action_history = []
            debug_path_screenshot_history = []
            for step in range(max_num_actions):
                self.debugger.log(f"Step {step}", color="yellow")

                screenshot = self.driver.get_non_loading_page()
                xml_layout = self.driver.get_xml_layout()
                debug_path_screenshot_history.append(screenshot)

                current_node = self.resolve_current_screen(screenshot, xml_layout, path_node_history, path_action_history)
                # add current node to path
                path_node_history.append(current_node)

                # Skip node if no UI elements, or no available action (for last step)
                if bypass_last_step_action_policy_for_bfs:
                    last_step = False
                else:
                    last_step = step == max_num_actions - 1
                ui_element, parsed_action = self.select_action_random_walk_policy(current_node, screenshot, last_step=last_step)

                # if no action is selected
                if not ui_element:
                    if len(current_node.ui_elements) == 0:
                        reason = f"No UI elements on node {current_node.node_id}"
                    else:
                        reason = f"No unexplored action for last step {current_node.node_id}"

                    self.debugger.log(f"{reason} — terminating walk", color="red")
                    walk_completed = False
                    break

                self._log_selected_action(ui_element, parsed_action)
                self.driver.execute_action(parsed_action)
                path_action_history.append(ui_element)

                # check if we are in the app package
                if self.configs.driver.skip_exploration_for_no_app_package:
                    fg = self.driver.get_foreground_package()
                    if fg != self.configs.driver.appPackage:
                        self.debugger.log(
                            f"Not in app package {self.configs.driver.appPackage} (foreground={fg!r}) — terminating walk",
                            color="red",
                        )
                        walk_completed = False
                        break

            # resolve landing after last executed action
            if len(path_action_history) > 0 and len(path_action_history) == len(path_node_history):
                # after last action we should resolve the current screen again
                screenshot = self.driver.get_non_loading_page()
                xml_layout = self.driver.get_xml_layout()
                current_node = self.resolve_current_screen(screenshot, xml_layout, path_node_history, path_action_history)
                path_node_history.append(current_node)
                debug_path_screenshot_history.append(screenshot)

            if save_debug_path:
                self.save_debug_path(path_action_history, debug_path_screenshot_history, self.walk_counter, "random_walk")
         
        self.app_graph.export_app_graph_state(self.configs.logs.root)
        self.app_graph.export_interactive_node_graph(os.path.join(self.configs.logs.root, "screenshots"), os.path.join(self.configs.logs.root, "interactive_node_graph.html"))
        self.app_graph.export_nx_graph_json(os.path.join(self.configs.logs.root, "graph.json"))


        
        return path_node_history, path_action_history, debug_path_screenshot_history, current_node, walk_completed



    def local_bfs(self, max_consetive_actions_before_bfs, max_num_actions_on_bfs_node):

        # max_consetive_actions_before_bfs is to control depth of graph, max_num_actions_on_bfs_node is to control branching factor of the node that we are doing BFS on
        
        
        self.debugger.rule("Local BFS", color="gray")
        if max_consetive_actions_before_bfs > 0:
            path_node_history_bfs, path_action_history_bfs, debug_path_screenshot_history_bfs, bfs_node, walk_completed = self.random_walk(max_consetive_actions_before_bfs, bypass_last_step_action_policy_for_bfs=True, by_pass_increase_walk_counter=True, save_debug_path=False)
            if walk_completed is False:
                return
        else:
            self.driver.reset_to_start_page()
            screenshot = self.driver.get_non_loading_page()
            xml_layout = self.driver.get_xml_layout()
            path_node_history_bfs = []
            path_action_history_bfs = []
            bfs_node = self.resolve_current_screen(screenshot, xml_layout, path_node_history_bfs, path_action_history_bfs)
            # add current node to path
            path_node_history_bfs.append(bfs_node)
            debug_path_screenshot_history_bfs = [screenshot]

        self.debugger.log(f"BFS node: {bfs_node.node_id}", color="blue")

        if self.configs.driver.skip_exploration_for_no_app_package:
            fg = self.driver.get_foreground_package()
            if fg != self.configs.driver.appPackage:
                self.debugger.log(
                    f"Not in app package {self.configs.driver.appPackage} (foreground={fg!r}) — terminating walk",
                    color="red",
                )
                return


        self.app_graph.num_walks += 1
        self.walk_counter += 1


        for step in range(max_num_actions_on_bfs_node):
            # Select an unexplored action from the node
            # Execute the action from BFS node --> See if we land in a new node (analyze that node)
            # Then we see if we can go back to parent node to do so we see if the back action is in the current node if node we use VLM
            # If we could go back we are in parent node so we go for next loop
            ui_element, parsed_action = self.select_action_local_bfs_policy(bfs_node, self.driver.get_non_loading_page())

            if not ui_element:
                if len(bfs_node.ui_elements) == 0:
                    reason = f"No UI elements on node {bfs_node.node_id}"
                else:
                    reason = f"No unexplored action for last step {bfs_node.node_id}"

                self.debugger.log(f"{reason} — terminating walk", color="red")
                break

            self._log_selected_action(ui_element, parsed_action)
            self.driver.execute_action(parsed_action)
            screenshot = self.driver.get_non_loading_page()
            xml_layout = self.driver.get_xml_layout()
            current_node = self.resolve_current_screen(screenshot, xml_layout, path_node_history_bfs, path_action_history_bfs + [ui_element], include_path_emphasize=False)

            back_to_previous_node_flag = self.backtracker.get_back_to_previous_node(current_node, bfs_node, ui_element, screenshot)

            if not back_to_previous_node_flag:
                with self.debugger.time_block("Backtrack using agent", color="gray"):
                    backtrack_using_agent_flag = self.backtracker.backtrack_using_agent(bfs_node)
                    if backtrack_using_agent_flag:
                        self.save_debug_path(path_action_history_bfs + [ui_element], debug_path_screenshot_history_bfs + [screenshot], self.walk_counter, "local_bfs", step)
                        continue
                    else:
                        self.debugger.log(f"Failed to backtrack using agent — terminating walk", color="red")
                        break

            # Write debug_path_screenshot_history as one jpg image
            self.save_debug_path(path_action_history_bfs + [ui_element], debug_path_screenshot_history_bfs + [screenshot], self.walk_counter, "local_bfs", step)


            if step % 5 == 0:
                self.app_graph.export_app_graph_state(self.configs.logs.root)
                self.app_graph.export_interactive_node_graph(os.path.join(self.configs.logs.root, "screenshots"), os.path.join(self.configs.logs.root, "interactive_node_graph.html"))
                self.app_graph.export_nx_graph_json(os.path.join(self.configs.logs.root, "graph.json"))



    def llm_based_bfs(self, max_num_actions_on_bfs_node):

        path_node_history_bfs = []
        path_action_history_bfs = []
        
        with self.debugger.time_block("LLM based BFS", color="gray"):
            prioritized_nodes = self.vlm_client.prioritize_node_for_bfs(self.app_graph, top_k=self.configs.graph.exploration_modes.model_llm_bfs.top_k_nodes_for_llm)
            ranked = prioritized_nodes.get("ranked_nodes", [])
            if ranked:
                selected_node = random.choice(ranked)
                selected_node_id = selected_node["node_id"]
                selected_reason = selected_node["reason"]
            else:
                selected_node_id = None
                selected_reason = None
 
        if selected_node_id is None:
            self.debugger.log(f"No node selected for LLM BFS — terminating walk", color="red")
            return

        self.debugger.log(f"Selected node for BFS: {selected_node_id} with reason: {selected_reason}", color="green")

        # LLM can return some key that is not in graph
        if selected_node_id not in self.app_graph.nodes:
            self.debugger.log(f"Node {selected_node_id} not in graph — terminating walk", color="red")
            return

        bfs_node = self.app_graph.nodes[selected_node_id]
        backtrack_using_agent_flag = self.backtracker.backtrack_using_agent(bfs_node)

        path_node_history_bfs, path_action_history_bfs = self.backtracker.get_action_history_and_node_history_from_root_node(bfs_node)

        debug_path_screenshot_history_bfs = []
        for node in path_node_history_bfs:
            screenshot = cv2.imread(os.path.join(self.configs.logs.root, "screenshots", f"{node.node_id}.jpg"))
            debug_path_screenshot_history_bfs.append(screenshot)

        if self.configs.driver.skip_exploration_for_no_app_package:
            fg = self.driver.get_foreground_package()
            if fg != self.configs.driver.appPackage:
                self.debugger.log(
                    f"Not in app package {self.configs.driver.appPackage} (foreground={fg!r}) — terminating walk",
                    color="red",
                )
                return


        self.app_graph.num_walks += 1
        self.walk_counter += 1


        if backtrack_using_agent_flag:
            for step in range(max_num_actions_on_bfs_node):
                # Select an unexplored action from the node
                # Execute the action from BFS node --> See if we land in a new node (analyze that node)
                # Then we see if we can go back to parent node to do so we see if the back action is in the current node if node we use VLM
                # If we could go back we are in parent node so we go for next loop
                ui_element, parsed_action = self.select_action_local_bfs_policy(bfs_node, self.driver.get_non_loading_page())

                if not ui_element:
                    if len(bfs_node.ui_elements) == 0:
                        reason = f"No UI elements on node {bfs_node.node_id}"
                    else:
                        reason = f"No unexplored action for last step {bfs_node.node_id}"

                    self.debugger.log(f"{reason} — terminating walk", color="red")
                    break

                self._log_selected_action(ui_element, parsed_action)
                self.driver.execute_action(parsed_action)
                screenshot = self.driver.get_non_loading_page()
                xml_layout = self.driver.get_xml_layout()
                current_node = self.resolve_current_screen(screenshot, xml_layout, path_node_history_bfs, path_action_history_bfs + [ui_element], include_path_emphasize=False)
                back_to_previous_node_flag = self.backtracker.get_back_to_previous_node(current_node, bfs_node, ui_element, screenshot)

                if not back_to_previous_node_flag:
                    with self.debugger.time_block("Backtrack using agent", color="gray"):
                        backtrack_using_agent_flag = self.backtracker.backtrack_using_agent(bfs_node)
                        if backtrack_using_agent_flag:
                            self.save_debug_path(path_action_history_bfs + [ui_element], debug_path_screenshot_history_bfs + [screenshot], self.walk_counter, "llm_based_bfs", step)
                            continue
                            
                        else:
                            self.debugger.log(f"Failed to backtrack using agent — terminating walk", color="red")
                            break

                self.save_debug_path(path_action_history_bfs + [ui_element], debug_path_screenshot_history_bfs + [screenshot], self.walk_counter, "llm_based_bfs", step)


                if step % 5 == 0:
                    self.app_graph.export_app_graph_state(self.configs.logs.root)
                    self.app_graph.export_interactive_node_graph(os.path.join(self.configs.logs.root, "screenshots"), os.path.join(self.configs.logs.root, "interactive_node_graph.html"))
                    self.app_graph.export_nx_graph_json(os.path.join(self.configs.logs.root, "graph.json"))

        else:
            self.debugger.log(f"Failed to backtrack using agent — terminating walk", color="red")


   
    