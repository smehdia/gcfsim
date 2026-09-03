
import os
import re
import secrets
import json
from datetime import datetime, timezone

import pickle
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from dataclasses import asdict, is_dataclass



def build_graph_id_session_prefix(device_id) -> str:
    """
    One prefix per ``AppGraph`` instance: device + UTC calendar date + short random.
    Parallel exploration sessions get distinct prefixes even on the same device/day.
    """
    def _sanitize_device_id_for_graph_ids(device_id) -> str:
        """Alphanumeric + hyphen/period only; safe for filenames and NetworkX labels."""
        s = re.sub(r"[^a-zA-Z0-9.-]", "-", str(device_id or "unknown"))
        s = re.sub(r"-+", "-", s).strip("-")
        return s[:48] if s else "unknown"
    dev = _sanitize_device_id_for_graph_ids(device_id)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    short = secrets.token_hex(4)
    return f"{dev}-{day}-{short}"

class AppGraph:
    def __init__(self, save_path, device_id=None):
        self.nx_graph = nx.MultiDiGraph()  # Internal NetworkX Directed Graph
        self.nodes = {}
        self._graph_id_session_prefix = build_graph_id_session_prefix(device_id)
        self.save_path = save_path
        self.num_walks = 0
        self.logs_num_walks_nodes = []  # useful to see the explored number of nodes per walk

    def generate_new_id(self):
        while True:
            tail = secrets.token_hex(5)
            candidate_id = f"{self._graph_id_session_prefix}-{tail}"
            if candidate_id not in self.nx_graph.nodes:
                return candidate_id

    def add_edge(self, from_node, to_node, action):
        """
        Add an edge from from_node to to_node to self.nx_graph with attributes
        'type', 'description', and 'boundingBox' taken from the action dict.
        """
        # Check if the edge already exists with the same src/dst, type, description, and boundingBox.
        from_id = from_node.node_id
        to_id = to_node.node_id
        el_type = action.get("type", "")
        el_desc = action.get("description", "")
        el_bbox = action.get("boundingBox")
        for _, v, edata in self.nx_graph.edges(from_id, data=True):
            if v != to_id:
                continue
            if (
                edata.get("type") == el_type
                and edata.get("description") == el_desc
                and edata.get("boundingBox") == el_bbox
            ):
                return


        self.nx_graph.add_edge(
            from_node.node_id,
            to_node.node_id,
            type=action.get("type", ""),
            description=action.get("description", ""),
            boundingBox=action.get("boundingBox")
        )

    def export_app_graph_state(self, root_path):
        self.export_nx_graph_json(os.path.join(root_path, 'graph.json'))
        with open(os.path.join(root_path, 'app_graph.pkl'), 'wb') as f:
            pickle.dump(self, f)
        print(f"AppGraph exported to {os.path.join(root_path, 'app_graph.pkl')}")
   

    def export_nx_graph_json(self, path):
        try:
            data = json_graph.node_link_data(self.nx_graph, edges="links")
        except TypeError:
            data = json_graph.node_link_data(self.nx_graph)
        data = self._to_json_safe(data)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _to_json_safe(obj):
        """Recursively convert ndarrays (e.g. ui_elements[].clip_embedding) for JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "cpu"):
            return obj.cpu().numpy().tolist()
        if isinstance(obj, dict):
            return {k: AppGraph._to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [AppGraph._to_json_safe(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if is_dataclass(obj):
            return AppGraph._to_json_safe(asdict(obj))
        raise TypeError(f"Cannot JSON-serialize {type(obj).__name__}")

    def _node_attrs(self, node):
        return {k: v for k, v in node.__dict__.items() if not k.startswith("_")}
        
    def sync_node(self, node):
        """Copy Node.__dict__ onto the NetworkX node (for JSON export)."""
        node_id = node.node_id
        if node_id not in self.nx_graph:
            self.nx_graph.add_node(node_id, **self._node_attrs(node))
        else:
            self.nx_graph.nodes[node_id].update(self._node_attrs(node))

    def add_node(self, node):
        self.nodes[node.node_id] = node
        self.sync_node(node)
    

    def visualize_actions_in_node(self, node_id, color=(0, 255, 0), thickness=2, show_titles=True):
        """
        Visualize available actions (UI elements) for a node by drawing bounding boxes on the screenshot.
        Also overlays the node ID in the top left corner.

        Args:
            node_id (str): ID of the node whose actions to visualize.
            color (tuple): BGR color tuple for bounding boxes.
            thickness (int): Thickness of bounding box lines.
            show_titles (bool): Whether to put action descriptions near boxes.
            out_path (str|None): File path to save visualization; if None, just show.
        Returns:
            output_img: np.ndarray (BGR visualization).
        """
        import cv2

        screenshot_path = os.path.join(self.save_path, "screenshots", f"{node_id}.jpg")
        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            raise ValueError(f"Could not read screenshot for node_id={node_id} at {screenshot_path}")

        # Retrieve node and its UI elements/actions
        node = self.nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in graph.")

        img_vis = screenshot.copy()
        ui_elements = getattr(node, "ui_elements", [])
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (color[0], color[1], color[2])
        line_type = 2

        # Draw the node ID in the top left corner
        node_id_text = f"Node ID: {node_id}"
        cv2.putText(
            img_vis, node_id_text, (10, 50),
            font, 1.0, (255, 0, 255), 3, cv2.LINE_AA  # Magenta for good contrast
        )

        for idx, el in enumerate(ui_elements):
            bbox = el.get("boundingBox") or el.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
            if ui_elements[idx].get("explored"):
                color = (0, 255, 0)
                thickness = 2
            else:
                color = (0, 0, 255)
                thickness = 2
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, thickness)
            if show_titles:
                desc = str(el.get("description") or el.get("type") or f"action{idx}")
                y_text = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.putText(
                    img_vis, desc, (x1, y_text),
                    font, font_scale, font_color, line_type, cv2.LINE_AA
                )

        return img_vis



    def export_interactive_node_graph(self, screenshots_path, output_html: str):
        import io
        import json
        import base64
        import cv2
        from PIL import Image

        def safe_base64_img(node_id):
            img_vis = self.visualize_actions_in_node(node_id)
            rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        nodes = []
        for node_id, data in self.nx_graph.nodes(data=True):
            page_summary = str(data.get("page_summary", "") or "")
            img_data_uri = safe_base64_img(node_id)
            nodes.append({
                "data": {
                    "id": str(node_id),
                    "label": page_summary,
                    "page_summary": page_summary,
                    "image": img_data_uri,
                }
            })

        edges = []
        edge_counter = 0
        nxg = self.nx_graph
        if nxg.is_multigraph():
            edge_iter = nxg.edges(keys=True, data=True)
        else:
            edge_iter = ((u, v, None, d) for u, v, d in nxg.edges(data=True))
        for u, v, _k, edata in edge_iter:
            label = str(edata.get("description", "")) or str(edata.get("type", "")) or "action"
            edges.append({
                "data": {
                    "id": f"ne{edge_counter}",
                    "source": str(u),
                    "target": str(v),
                    "label": label,
                }
            })
            edge_counter += 1

        elements = nodes + edges

        html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>AppGraph Node Graph</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
html, body, #cy {{
  height: 100%;
  width: 100%;
  margin: 0;
  padding: 0;
  overflow: hidden;
  background: #ffffff;
}}
.legend {{
  position: absolute;
  top: 8px; left: 8px;
  background: rgba(255,255,255,0.9);
  border-radius: 8px;
  padding: 8px 12px;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  font-size: 10px;
  z-index: 20;
}}
</style>
</head>
<body>
<div id="cy"></div>
<div class="legend">
<b>Controls</b>: drag to pan, wheel to zoom, click node/edge to inspect.
</div>
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<script>
const elements = {json.dumps(elements, separators=(',', ':'))};
const cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: elements,
  layout: {{
    name: 'cose',
    animate: false,
    padding: 50
  }},
  style: [
    {{
      selector: 'node',
      style: {{
        'background-image': 'data(image)',
        'background-fit': 'contain',
        'background-color': '#ffffff',
        'border-color': '#94a3b8',
        'border-width': 2,
        'shape': 'round-rectangle',
        'width': 280,
        'height': 180,
        'label': 'data(label)',
        'text-valign': 'bottom',
        'text-wrap': 'wrap',
        'text-max-width': 260,
        'font-size': 10,
        'color': '#e2e8f0',
        'padding': '8px',
        'text-margin-y': 6,
        'shadow-blur': 10,
        'shadow-color': 'rgba(0,0,0,0.35)',
        'shadow-offset-x': 0,
        'shadow-offset-y': 2
      }}
    }},
    {{
      selector: 'edge',
      style: {{
        'curve-style': 'bezier',
        'width': 2,
        'line-color': '#94a3b8',
        'target-arrow-color': '#94a3b8',
        'target-arrow-shape': 'triangle',
        'arrow-scale': 1.2,
        'label': 'data(label)',
        'font-size': 9,
        'color': '#cbd5e1',
        'text-wrap': 'wrap',
        'text-max-width': 240,
        'text-background-color': '#ffffff',
        'text-background-opacity': 0.7,
        'text-background-shape': 'round-rectangle',
        'text-rotation': 'autorotate'
      }}
    }},
    {{
      selector: ':selected',
      style: {{
        'border-color': '#22c55e',
        'line-color': '#22c55e',
        'target-arrow-color': '#22c55e',
        'background-color': '#22c55e'
      }}
    }}
  ]
}});
cy.on('tap', 'node, edge', (evt) => {{
  const d = evt.target.data();
  console.log('Selected:', d);
}});
cy.ready(() => {{
  cy.fit(undefined, 50);
}});
</script>
</body>
</html>"""
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html)

    def convert_nodes_to_json_safe(self, data, ignore_keys=("embedding", "screenshot")):
        for node in data.get("nodes", []):
            for k in ignore_keys:
                node.pop(k, None)
        for edge in data.get("links", []):
            for k in ignore_keys:
                edge.pop(k, None)
        return AppGraph._to_json_safe(data)

    def split_long_text(self, text, max_length=60):
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) <= max_length:
                current_line += " " + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return "\n".join(lines)