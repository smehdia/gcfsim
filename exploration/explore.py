import cv2
from dynaconf import Dynaconf
from PIL import Image

from Debugger import Debugger
from Driver.factory import build_driver

from ImageUtils import ImageUtils
from Graph.Node import Node
from VLM import VLM
from VLM_Yibu import VLM_Yibu
from Graph.AppGraph import AppGraph
from Graph.Walker import Walker

from Agents.factory import build_agent

import json
from networkx.readwrite import json_graph
import networkx as nx
import random
import os
import pickle
import argparse
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import traceback

for var in ["http_proxy", "https_proxy", "ftp_proxy", "socks_proxy", 
            "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "SOCKS_PROXY"]:
    os.environ.pop(var, None)


def select_exploration_mode(configs):
    mode_probs = configs.graph.exploration_modes["probabilities"]
    modes = list(mode_probs.keys())
    weights = list(mode_probs.values())
    selected_mode = random.choices(modes, weights=weights, k=1)[0]
    return selected_mode

def load_or_create_app_graph(configs, debugger):
    if not os.path.exists(configs.logs.root):
        os.makedirs(configs.logs.root, exist_ok=True)
    if configs.logs.resume_from_checkpoint and "app_graph.pkl" in os.listdir(configs.logs.root):
        debugger.log(f"Loading AppGraph from {configs.logs.root}/app_graph.pkl", color="cyan")
        with open(os.path.join(configs.logs.root, "app_graph.pkl"), "rb") as f:
            app_graph = pickle.load(f)
        app_graph.save_path = configs.logs.root
        return app_graph
    debugger.log("Creating new AppGraph", color="yellow")
    return AppGraph(save_path=configs.logs.root, device_id=configs.driver.device_id)


def get_max_consecutive_actions_based_on_schedule(schedule_dict, i):
    """
    Given a schedule_dict (e.g., {1:20, 2:100, 3:200,...}) and a step i,
    returns the key from the schedule dict corresponding to i.
    For example: schedule_dict = {1: 20, 2: 100, 3: 200, 4: 300, 5: 300, 6: 100}
    If i = 10 -> returns 1 (10 < 20)
    If i = 350 -> returns 4 (first 20 + 100 + 200 = 320, next 300 covers 320-620)
    """
    cumulative = 0
    for key in sorted(schedule_dict.keys()):
        cumulative += schedule_dict[key]
        if i < cumulative:
            return key
    # If i exceeds total schedule, return the last key
    return max(schedule_dict.keys())


def plot_nodes_vs_walk(logs_num_walks_nodes, save_path: str):
    walks, num_nodes, modes = [], [], []
    for row in logs_num_walks_nodes:
        if not row or len(row) < 2:
            continue
        try:
            walks.append(int(row[0]))
            num_nodes.append(int(row[1]))
            modes.append(str(row[2]) if len(row) > 2 else "unknown")
        except (TypeError, ValueError):
            continue
    if not walks:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(walks, num_nodes, color="gray", linewidth=1, zorder=1, label="_nolegend_")

    mode_color_map = {
        "local_bfs": "dodgerblue",
        "random_walk": "orange",
        "model_llm_bfs": "green",
    }
    mode_names_pretty = {
        "local_bfs": "Local BFS",
        "random_walk": "Random Walk",
        "model_llm_bfs": "LLM BFS",
    }
    for mode in sorted(set(modes)):
        inds = [i for i, m in enumerate(modes) if m == mode]
        plt.scatter(
            [walks[i] for i in inds],
            [num_nodes[i] for i in inds],
            color=mode_color_map.get(mode, "k"),
            s=40,
            alpha=0.85,
            label=mode_names_pretty.get(mode, mode),
            zorder=2,
        )

    plt.xlabel("Walk")
    plt.ylabel("Number of nodes")
    plt.title("Number of nodes per walk")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    dbg = Debugger(palette="soft", indent_size=2, width=90)

    # i want to get config file from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/clock_android.yaml")
    args = parser.parse_args()

    configs = Dynaconf(settings_files=[args.config], merge_enabled=True)
    configs = configs.default

    if configs.vlm.use_yibu_api:
        vlm_client = VLM_Yibu(configs, dbg)
    else:
        vlm_client = VLM(configs, dbg)
    agent = build_agent(model_name=configs.agent.model_name, url=configs.agent.url, agent_settings=configs.agent.settings, debugger=dbg)
    driver = build_driver(settings=configs.driver, agent=agent)

    app_meta_info = driver.get_app_version()
    dbg.log(f"App Meta info: {app_meta_info}", color="green")


    # Ensure the logs directory exists before writing meta_info.json
    os.makedirs(configs.logs.root, exist_ok=True)
    with open(os.path.join(configs.logs.root, "meta_info.json"), "w") as f:
        json.dump({"app_meta_info": app_meta_info, "configs": configs.to_dict()}, f)

    app_graph = load_or_create_app_graph(configs, dbg)

    image_utils = ImageUtils()

    walker = Walker(app_graph, image_utils, dbg, driver, vlm_client, agent, configs, walk_counter=0)
    
    for i in range(app_graph.num_walks, configs.graph.exploration_walks):

        try:
            max_consecutive_actions_in_each_iteration = get_max_consecutive_actions_based_on_schedule(configs.graph.exploration_schedule_depth_iterations, i)
            # if max depth < 2 we don't LLM BFS to save budget
            if max_consecutive_actions_in_each_iteration <= 2:
                while True:
                    selected_mode = select_exploration_mode(configs)
                    if selected_mode != "model_llm_bfs":
                        break
            else:
                selected_mode = select_exploration_mode(configs)
            
            configs.graph.max_consecutive_exploration_actions_in_each_iteration = max_consecutive_actions_in_each_iteration
            
            dbg.rule(f"Exploration walk {i} - exploration mode: {selected_mode}, max Depth: {max_consecutive_actions_in_each_iteration}", color="purple")

            if selected_mode == "local_bfs":
                walker.local_bfs(max_consetive_actions_before_bfs=configs.graph.max_consecutive_exploration_actions_in_each_iteration-1, max_num_actions_on_bfs_node=configs.graph.exploration_modes.local_bfs.max_branching_factor)
                app_graph.logs_num_walks_nodes.append([int(app_graph.num_walks), len(app_graph.nodes), "local_bfs"])
                app_graph.export_app_graph_state(configs.logs.root)
                app_graph.export_interactive_node_graph(os.path.join(configs.logs.root, "screenshots"), os.path.join(configs.logs.root, "interactive_node_graph.html"))
                app_graph.export_nx_graph_json(os.path.join(configs.logs.root, "graph.json"))

            elif selected_mode == "random_walk":
                walker.random_walk(max_num_actions=configs.graph.max_consecutive_exploration_actions_in_each_iteration, save_debug_path=True)
                app_graph.logs_num_walks_nodes.append([int(app_graph.num_walks), len(app_graph.nodes), "random_walk"])
                app_graph.export_app_graph_state(configs.logs.root)
                app_graph.export_interactive_node_graph(os.path.join(configs.logs.root, "screenshots"), os.path.join(configs.logs.root, "interactive_node_graph.html"))
                app_graph.export_nx_graph_json(os.path.join(configs.logs.root, "graph.json"))
            elif selected_mode == "model_llm_bfs":
                walker.llm_based_bfs(max_num_actions_on_bfs_node=configs.graph.exploration_modes.local_bfs.max_branching_factor)
                app_graph.logs_num_walks_nodes.append([int(app_graph.num_walks), len(app_graph.nodes), "model_llm_bfs"])
                app_graph.export_app_graph_state(configs.logs.root)
                app_graph.export_interactive_node_graph(os.path.join(configs.logs.root, "screenshots"), os.path.join(configs.logs.root, "interactive_node_graph.html"))
                app_graph.export_nx_graph_json(os.path.join(configs.logs.root, "graph.json"))

            else:
                raise ValueError(f"Invalid exploration mode: {selected_mode}")

            dbg.log("Token usage details", value=vlm_client.token_usage_details, color="green")
            plot_nodes_vs_walk(
                app_graph.logs_num_walks_nodes,
                os.path.join(configs.logs.root, "num_nodes_vs_walk.png"),
            )
        except Exception as e:
            dbg.log(f"Error in exploration walk {i} - exploration mode: {selected_mode} — terminating walk. Exception: {e}", color="red")
            traceback.print_exc()
       
