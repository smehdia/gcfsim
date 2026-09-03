import re
import html
from typing import Dict, List
from collections import Counter
import xml.etree.ElementTree as ET



def get_android_xml_structure_distance(xml1: str, xml2: str) -> dict:
    """
    Structure-only normalized distance between two Android XML layouts.

    Designed to ignore:
      - text/resource-id/content-desc semantics
      - clickable / long-clickable changes
      - enabled / disabled changes
      - checked / selected / focused changes
      - toggle/textbox temporary actionability changes
      - status bar / navigation bar / system UI

    It compares only a normalized structural class sequence.
    """

    def _unusable_xml(s: str) -> bool:
        t = (s or "").strip()
        return (not t) or ("</hierarchy>" not in t)

    s1 = "" if xml1 is None else str(xml1)
    s2 = "" if xml2 is None else str(xml2)

    if _unusable_xml(s1) or _unusable_xml(s2):
        return {
            "distance": 1.0,
            "preorder_distance": 1.0,
            "bag_distance": 1.0,
            "preorder_similarity": 0.0,
            "bag_similarity": 0.0,
            "num_nodes_1": 0,
            "num_nodes_2": 0,
            "valid": False,
            "reason": "unusable_xml",
        }

    def parse_bounds(bounds: str):
        """
        Parse Android bounds like: [0,111][1216,2688]
        """
        m = re.match(r"\[(\-?\d+),(\-?\d+)\]\[(\-?\d+),(\-?\d+)\]", bounds or "")
        if not m:
            return None
        return tuple(map(int, m.groups()))

    def coarse_class(cls: str) -> str:
        if not cls:
            return "unknown"

        c = cls.split(".")[-1].lower()

        mapping = {
            "textview": "text",
            "text": "text",
            "edittext": "input",

            "button": "button",
            "imagebutton": "icon_button",

            "imageview": "image",
            "image": "image",

            # Ignore state, but keep structural role.
            "checkbox": "toggle",
            "radiobutton": "toggle",
            "switch": "toggle",
            "switchcompat": "toggle",

            "recyclerview": "list",
            "listview": "list",

            "scrollview": "scroll",
            "nestedscrollview": "scroll",

            "linearlayout": "container",
            "relativelayout": "container",
            "framelayout": "container",
            "constraintlayout": "container",
            "viewgroup": "container",

            "viewpager": "pager",
            "viewpager2": "pager",
            "tablayout": "tabs",
            "toolbar": "toolbar",

            # Harmony / system-like structural primitives.
            "stack": "container",
            "row": "container",
            "column": "container",
            "relativecontainer": "container",
            "__common__": "container",
        }

        return mapping.get(c, c)

    def is_visible(node) -> bool:
        v = node.attrib.get("visible-to-user")
        return not (v is not None and str(v).lower() == "false")

    def is_system_ui(node) -> bool:
        raw = " ".join([
            node.attrib.get("class", ""),
            node.attrib.get("resource-id", ""),
            node.attrib.get("package", ""),
        ]).lower()

        bad = [
            "statusbar",
            "status_bar",
            "statusbarbackground",
            "navigationbar",
            "navigation_bar",
            "navigationbarbackground",
            "windowscene",
            "sceneboard",
            "batterycomponent",
            "wificomponent",
            "signalcomponent",
            "clockstatusview",
            "timeview",
            "livecapsule",
            "statusbaricon",
            "status_bar_clock",
            "status_bar_wifi",
            "status_bar_signal",
            "status_bar_battery",
            "status_bar_notification",
        ]

        return any(x in raw for x in bad)

    def is_tiny_artifact(node) -> bool:
        """
        Removes separators or degenerate layout artifacts.
        Does not use clickable/enabled/checkable state.
        """
        box = parse_bounds(node.attrib.get("bounds", ""))
        if box is None:
            return False

        x1, y1, x2, y2 = box
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        # Extremely thin/tiny nodes are usually dividers, bottom gesture bars,
        # or measurement artifacts.
        return w <= 2 or h <= 2

    def should_skip(node) -> bool:
        if is_system_ui(node):
            return True

        if not is_visible(node):
            return True

        if is_tiny_artifact(node):
            return True

        cls = coarse_class(node.attrib.get("class", ""))

        # IMPORTANT:
        # Do not use clickable, long-clickable, enabled, checkable, checked,
        # focused, or selected here. Those are state/action attributes and can
        # change between dumps of the same page.
        #
        # Only skip trivial one-child containers. We still traverse children.
        if cls == "container" and len(list(node)) <= 1:
            return True

        return False

    def token(node, depth):
        """
        Pure structure token.
        No exact depth.
        No clickable/enabled/checkable/checked.
        No text/resource-id/content-desc.
        """
        cls = coarse_class(node.attrib.get("class", ""))
        return cls

    def traverse(root):
        preorder = []
        bag = []

        def dfs(node, depth):
            if should_skip(node):
                for child in list(node):
                    dfs(child, depth)
                return

            t = token(node, depth)
            preorder.append(t)
            bag.append(t)

            for child in list(node):
                dfs(child, depth + 1)

        dfs(root, 0)
        return preorder, bag

    def levenshtein(seq1, seq2):
        n, m = len(seq1), len(seq2)

        if n == 0:
            return m
        if m == 0:
            return n

        dp = list(range(m + 1))

        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i

            for j in range(1, m + 1):
                tmp = dp[j]
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1

                dp[j] = min(
                    dp[j] + 1,      # deletion
                    dp[j - 1] + 1,  # insertion
                    prev + cost     # substitution
                )

                prev = tmp

        return dp[m]

    def normalized_edit_distance(seq1, seq2):
        return levenshtein(seq1, seq2) / max(len(seq1), len(seq2), 1)

    def multiset_jaccard_distance(items1, items2):
        c1, c2 = Counter(items1), Counter(items2)
        keys = set(c1) | set(c2)

        inter = sum(min(c1[k], c2[k]) for k in keys)
        union = sum(max(c1[k], c2[k]) for k in keys)

        if union == 0:
            return 0.0

        return 1.0 - inter / union

    try:
        root1 = ET.fromstring(s1)
        root2 = ET.fromstring(s2)
    except ET.ParseError:
        return {
            "distance": 1.0,
            "preorder_distance": 1.0,
            "bag_distance": 1.0,
            "preorder_similarity": 0.0,
            "bag_similarity": 0.0,
            "num_nodes_1": 0,
            "num_nodes_2": 0,
            "valid": False,
            "reason": "parse_error",
        }

    preorder1, bag1 = traverse(root1)
    preorder2, bag2 = traverse(root2)

    n1, n2 = len(preorder1), len(preorder2)

    d_preorder = normalized_edit_distance(preorder1, preorder2)
    d_bag = multiset_jaccard_distance(bag1, bag2)

    # Less preorder dominance because Android/Compose XML ordering can shift.
    distance = 0.4 * d_preorder + 0.6 * d_bag
    distance = max(0.0, min(1.0, distance))

    count_ratio = min(n1, n2) / max(n1, n2, 1)

    return {
        "distance": distance,
        "preorder_distance": d_preorder,
        "bag_distance": d_bag,
        "preorder_similarity": 1.0 - d_preorder,
        "bag_similarity": 1.0 - d_bag,
        "num_nodes_1": n1,
        "num_nodes_2": n2,
        "count_ratio": count_ratio,
        "valid": True,
        "reason": "ok",
    }



def xml_to_semantic_text_signature(
    xml: str,
    include_content_desc: bool = True,
    include_hint: bool = False,
    max_items_per_group: int = 30,
    max_content_examples: int = 8,
) -> str:
    """
    Convert Android XML into a compact semantic text signature.

    Intended use:
        - second-stage semantic verification after XML structure distance already passed
        - Qwen/LLM input
        - exact navigational page matching

    Returns a text block like:

        NAV:
        - search
        - home tab
        - cart tab

        SUBNAV:
        - same-day delivery
        - haul

        CONTROLS:
        - your orders
        - shop by category

        CONTENT_EXAMPLES:
        - deals based on your lists
        - product title ...

    Notes:
        - Does not include structural counts/classes.
        - Tries to separate stable UI labels from dynamic/feed content.
        - Keeps heuristics minimal and conservative.
    """

    def normalize_text(s: str) -> str:
        s = html.unescape(s or "")
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)

        # Accessibility suffix cleanup.
        s = re.sub(r"\.?\s*list item \d+ of \d+", "", s)
        s = re.sub(r"\.?\s*tab \d+ of \d+", " tab", s)
        s = re.sub(r"\.?\s*button \d+ of \d+", " button", s)

        # Light normalization of dynamic quantities.
        s = re.sub(r"\b\d+\s+items?\b", "items", s)
        s = re.sub(r"\b\d+\s+notifications?\b", "notifications", s)
        s = re.sub(r"\b\d+%\s*off\b", "discount", s)
        s = re.sub(r"\b\d+\s*viewed\b", "viewed", s)

        # Normalize common deal terms.
        s = re.sub(r"\blimited-time deal\b", "deal", s)
        s = re.sub(r"\bdeal selling fast\b", "deal", s)

        # Fix common fused accessibility delivery strings.
        # Example: "deliver tol 5 n 0 hbutton"
        s = re.sub(
            r"deliver to\s*[a-z]\s*\d\s*[a-z]\s*\d\s*[a-z]\s*\d\s*button",
            "deliver to",
            s,
        )
        s = re.sub(
            r"deliver to[a-z]\s*\d\s*[a-z]\s*\d\s*[a-z]\s*\dbutton",
            "deliver to",
            s,
        )

        s = re.sub(r"\s+", " ", s)
        s = s.strip(" .,;:-")
        return s

    def is_noise_text(s: str) -> bool:
        if not s:
            return True

        # Pure numeric/internal accessibility IDs.
        if re.fullmatch(r"[\d\s:.,/%$€£¥+\-#()]+", s):
            return True

        if s in {"•", "|", "-", "–", "—"}:
            return True

        return False

    def bool_attr(node: ET.Element, name: str) -> bool:
        return node.attrib.get(name, "false").lower() == "true"

    def dedup_preserve_order(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def classify_label(text: str, clickable: bool, focusable: bool, selected: bool) -> str:
        """
        Minimal semantic role classification:
            nav | subnav | control | content
        """

        # Real tab label. Avoid matching words like "tablet".
        if re.search(r"\btab\b", text):
            return "nav"

        # Persistent search/header controls.
        if text in {
            "search",
            "search or ask a question",
            "scan it",
            "voice search",
            "camera search",
            "google search",
            "deliver to",
        }:
            return "nav"

        if text.startswith("deliver to"):
            return "nav"

        # Long text is likely product/feed/body content.
        if len(text) > 80 or len(text.split()) >= 10:
            return "content"

        # Common dynamic/feed labels.
        if re.search(
            r"\b("
            r"sponsored|product image|viewed|recommended|"
            r"keep shopping|continue shopping|based on your|"
            r"mother's day|father's day|last-minute|"
            r"limited-time deal|deal selling fast"
            r")\b",
            text,
        ):
            return "content"

        # Page-level pills/categories/section links.
        if re.search(
            r"\b("
            r"deals|same-day delivery|haul|alexa lists|"
            r"see more|shop|category|categories|prime|"
            r"gifting|savings|home & kitchen"
            r")\b",
            text,
        ):
            return "subnav"

        # Clickable/focusable short labels are likely page controls/subnav.
        if clickable or focusable or selected:
            return "subnav"

        return "control"

    if not xml or not str(xml).strip():
        return ""

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return ""

    fields = ["text"]
    if include_content_desc:
        fields.append("content-desc")
    if include_hint:
        fields.append("hint")

    nav = []
    subnav = []
    control = []
    content = []
    selected = []
    clickable_labels = []

    seen_raw = set()

    for node in root.iter():
        clickable = bool_attr(node, "clickable") or bool_attr(node, "long-clickable")
        focusable = bool_attr(node, "focusable")
        is_selected = bool_attr(node, "selected")

        for field in fields:
            text = normalize_text(node.attrib.get(field, ""))

            if is_noise_text(text):
                continue

            # Dedup by normalized text globally.
            if text in seen_raw:
                continue
            seen_raw.add(text)

            role = classify_label(
                text=text,
                clickable=clickable,
                focusable=focusable,
                selected=is_selected,
            )

            if role == "nav":
                nav.append(text)
            elif role == "subnav":
                subnav.append(text)
            elif role == "control":
                control.append(text)
            else:
                content.append(text)

            if is_selected:
                selected.append(text)

            if clickable or focusable:
                clickable_labels.append(text)

    nav = dedup_preserve_order(nav)[:max_items_per_group]
    subnav = dedup_preserve_order(subnav)[:max_items_per_group]
    control = dedup_preserve_order(control)[:max_items_per_group]
    content = dedup_preserve_order(content)[:max_content_examples]
    selected = dedup_preserve_order(selected)[:max_items_per_group]
    clickable_labels = dedup_preserve_order(clickable_labels)[:max_items_per_group]

    sections = []

    if nav:
        sections.append("NAV:\n" + "\n".join(f"- {x}" for x in nav))

    if subnav:
        sections.append("SUBNAV_OR_PAGE_LINKS:\n" + "\n".join(f"- {x}" for x in subnav))

    if control:
        sections.append("CONTROLS_OR_STABLE_LABELS:\n" + "\n".join(f"- {x}" for x in control))

    if selected:
        sections.append("SELECTED:\n" + "\n".join(f"- {x}" for x in selected))

    if clickable_labels:
        sections.append("CLICKABLE_LABELS:\n" + "\n".join(f"- {x}" for x in clickable_labels))

    if content:
        sections.append(
            "CONTENT_EXAMPLES_WEAK_EVIDENCE:\n"
            + "\n".join(f"- {x}" for x in content)
        )

    return "\n\n".join(sections)