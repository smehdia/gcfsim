from contextlib import contextmanager
from datetime import datetime
from time import perf_counter

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text


class Debugger:
    PALETTES = {
        "default": {
            "blue": "cyan",
            "green": "green",
            "yellow": "yellow",
            "red": "red",
            "purple": "magenta",
            "gray": "dim white",
            "white": "white",
        },
        "soft": {
            "blue": "sky_blue1",
            "green": "spring_green3",
            "yellow": "khaki1",
            "red": "salmon1",
            "purple": "plum1",
            "gray": "grey70",
            "white": "white",
        },
        "bold": {
            "blue": "bold blue",
            "green": "bold green",
            "yellow": "bold yellow",
            "red": "bold red",
            "purple": "bold magenta",
            "gray": "bold grey70",
            "white": "bold white",
        },
    }

    def __init__(
        self,
        enabled: bool = True,
        palette: str = "default",
        indent_size: int = 2,
        width: int = 100,
    ):
        self.console = Console(width=width)
        self.enabled = enabled
        self.indent_size = indent_size
        self.width = width
        self.depth = 0
        self.palette = self.PALETTES.get(palette, self.PALETTES["default"])

    def _style(self, color: str) -> str:
        return self.palette.get(color, color)

    def _depth(self, depth):
        return self.depth if depth is None else depth

    def _indent(self, depth) -> str:
        return " " * (self._depth(depth) * self.indent_size)

    def _get_field(self, obj, name, default="-"):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _format_resolution(self, w, h) -> str:
        if w in (None, "-") or h in (None, "-"):
            return "-"
        return f"{w} × {h}"

    def _format_coords(self, coords) -> str:
        if not coords:
            return "-"

        if isinstance(coords, dict) and "point" in coords:
            x, y = coords["point"]
            return f"({x}, {y})"

        return str(coords)

    def log(self, message, value=None, *, depth=None, color="blue"):
        if not self.enabled:
            return

        depth = self._depth(depth)
        style = self._style(color)
        indent = self._indent(depth)

        self.console.print(Text(indent + str(message), style=style))

        if value is not None:
            value_indent = self._indent(depth + 1)
            self.console.print(value_indent, Pretty(value, indent_guides=True))

    def rule(self, title="", *, color="blue"):
        """
        Major section divider.
        Usually not indented.
        """
        if not self.enabled:
            return

        style = self._style(color)
        self.console.rule(f"[{style}]{title}[/{style}]", style=style)

    def subrule(self, title="", *, depth=None, color="gray", symbol="─"):
        """
        Smaller subsection divider with optional automatic indentation.

        Useful symbols:
            ─  light line
            ━  heavy line
            ·  dotted
            •  bullet-like
            -  ASCII-safe
            =  strong ASCII
        """
        if not self.enabled:
            return

        depth = self._depth(depth)
        indent = self._indent(depth)
        style = self._style(color)

        if title:
            prefix = f"{indent}{symbol * 4} {title} "
            remaining = max(0, self.width - len(prefix))
            text = prefix + symbol * remaining
        else:
            text = indent + symbol * max(0, self.width - len(indent))

        self.console.print(Text(text, style=style))

    @contextmanager
    def time_block(
        self,
        name,
        *,
        depth=None,
        color="gray",
        show_start=True,
        show_end=True,
    ):
        """
        Timed debug block with automatic nesting.

        Example:
            with dbg.time_block("main"):
                dbg.log("inside main")

                with dbg.time_block("sub step"):
                    dbg.log("inside sub step")
        """
        if not self.enabled:
            yield
            return

        depth = self._depth(depth)

        start_clock = datetime.now()
        start = perf_counter()

        if show_start:
            self.log(
                f"▶ START {name} | {start_clock.strftime('%H:%M:%S')}",
                depth=depth,
                color=color,
            )

        previous_depth = self.depth
        self.depth = depth + 1

        try:
            yield
        finally:
            elapsed = perf_counter() - start
            end_clock = datetime.now()

            self.depth = previous_depth

            if show_end:
                self.log(
                    f"✓ END   {name} | {end_clock.strftime('%H:%M:%S')} | took {self.format_seconds(elapsed)}",
                    depth=depth,
                    color=color,
                )

    def format_seconds(self, seconds: float) -> str:
        if seconds < 1:
            return f"{seconds * 1000:.1f} ms"

        if seconds < 60:
            return f"{seconds:.2f} s"

        minutes, sec = divmod(seconds, 60)
        if minutes < 60:
            return f"{int(minutes)} min {sec:.1f} s"

        hours, minutes = divmod(minutes, 60)
        return f"{int(hours)} h {int(minutes)} min {sec:.1f} s"

    def elements(self, elements, *, title="UI Elements", depth=None):
        if not self.enabled:
            return

        depth = self._depth(depth)

        table = Table(
            title=title,
            show_header=True,
            show_lines=False,
            box=box.ROUNDED,
            padding=(0, 1),
        )

        table.add_column("id", style=self._style("blue"), justify="right")
        table.add_column("type", style=self._style("green"))
        table.add_column("description", style=self._style("white"), overflow="fold")

        for elem in elements:
            table.add_row(
                str(elem.get("id", "")),
                str(elem.get("type", "")),
                str(elem.get("description", "")),
            )

        self.console.print(self._indent(depth), table)

    def node_report(
        self,
        node_id,
        page_summary,
        elements,
        *,
        element_tokens=None,
        summary_tokens=None,
        depth=None,
        title="Node Report",
        color="blue",
    ):
        """
        Print one node with:
          - node_id
          - page_summary
          - token usage for element extraction and page summary extraction
          - UI elements table

        Expected token dict format:
            {
                "input_text": 1200,
                "input_image": 2300,
                "output": 400,
            }
        """
        if not self.enabled:
            return

        depth = self._depth(depth)
        element_tokens = element_tokens or {}
        summary_tokens = summary_tokens or {}

        def token_total(tokens):
            return (
                int(tokens.get("input_text", 0))
                + int(tokens.get("input_image", 0))
                + int(tokens.get("output", 0))
            )

        metadata = Table.grid(padding=(0, 1))
        metadata.add_column(style=self._style("purple"), justify="right")
        metadata.add_column(style=self._style("white"))

        metadata.add_row("node_id", str(node_id))
        metadata.add_row("summary", str(page_summary))

        token_table = Table(
            title="Token Usage",
            show_header=True,
            show_lines=False,
            box=box.ROUNDED,
            padding=(0, 1),
        )

        token_table.add_column("stage", style=self._style("purple"))
        token_table.add_column("text in", justify="right", style=self._style("blue"))
        token_table.add_column("image in", justify="right", style=self._style("yellow"))
        token_table.add_column("output", justify="right", style=self._style("green"))
        token_table.add_column("total", justify="right", style=self._style("white"))

        token_table.add_row(
            "element extraction",
            str(element_tokens.get("input_text", 0)),
            str(element_tokens.get("input_image", 0)),
            str(element_tokens.get("output", 0)),
            str(token_total(element_tokens)),
        )

        token_table.add_row(
            "page summary extraction",
            str(summary_tokens.get("input_text", 0)),
            str(summary_tokens.get("input_image", 0)),
            str(summary_tokens.get("output", 0)),
            str(token_total(summary_tokens)),
        )

        element_table = Table(
            title="UI Elements",
            show_header=True,
            show_lines=False,
            box=box.ROUNDED,
            padding=(0, 1),
        )

        element_table.add_column("id", style=self._style("blue"), justify="right")
        element_table.add_column("type", style=self._style("green"))
        element_table.add_column("description", style=self._style("white"), overflow="fold")

        for elem in elements:
            element_table.add_row(
                str(elem.get("id", "")),
                str(elem.get("type", "")),
                str(elem.get("description", "")),
            )

        content = Group(metadata, token_table, element_table)

        self.console.print(
            self._indent(depth),
            Panel(
                content,
                title=f"{title} | node {node_id}",
                border_style=self._style(color),
                box=box.ROUNDED,
                padding=(1, 2),
            ),
        )

    def agent_result(
        self,
        *,
        agent_name,
        instruction,
        result,
        meta=None,
        depth=None,
        title="Agent Result",
        color="blue",
        show_raw=False,
    ):
        """
        Print compact agent output.

        Expected result format:
            (parsed_action, sent_w, sent_h)

        Expected meta format:
            ResizeMeta(orig_w, orig_h, sent_w, sent_h)
        """
        if not self.enabled:
            return

        depth = self._depth(depth)

        parsed_action = None
        result_w = "-"
        result_h = "-"

        if isinstance(result, tuple):
            if len(result) >= 1:
                parsed_action = result[0]
            if len(result) >= 3:
                result_w = result[1]
                result_h = result[2]
        else:
            parsed_action = result

        orig_w = self._get_field(meta, "orig_w")
        orig_h = self._get_field(meta, "orig_h")
        sent_w = self._get_field(meta, "sent_w", result_w)
        sent_h = self._get_field(meta, "sent_h", result_h)

        metadata = Table(
            title="Metadata",
            show_header=False,
            box=box.ROUNDED,
            padding=(0, 1),
        )
        metadata.add_column("field", style=self._style("purple"))
        metadata.add_column("value", style=self._style("white"))

        metadata.add_row("agent", str(agent_name))
        metadata.add_row("image original", self._format_resolution(orig_w, orig_h))
        metadata.add_row("image resized/sent", self._format_resolution(sent_w, sent_h))
        metadata.add_row("result image size", self._format_resolution(result_w, result_h))

        instruction_panel = Panel(
            Text(str(instruction), style=self._style("white")),
            title="Instruction",
            border_style=self._style("gray"),
            box=box.ROUNDED,
            padding=(0, 1),
        )

        output = Table(
            title="Model Output",
            show_header=False,
            box=box.ROUNDED,
            padding=(0, 1),
        )
        output.add_column("field", style=self._style("purple"))
        output.add_column("value", style=self._style("white"), overflow="fold")

        thought = self._get_field(parsed_action, "thought")
        action_type = self._get_field(parsed_action, "action_type")
        params = self._get_field(parsed_action, "params")
        sent_coords = self._get_field(parsed_action, "sent_coords")
        orig_coords = self._get_field(parsed_action, "orig_coords")

        output.add_row("thought", str(thought))
        output.add_row("action_type", str(action_type))
        output.add_row("params", str(params))
        output.add_row("sent_coords", self._format_coords(sent_coords))
        output.add_row("orig_coords", self._format_coords(orig_coords))

        blocks = [metadata, instruction_panel, output]

        if show_raw:
            raw_text = self._get_field(parsed_action, "raw_text")
            raw_panel = Panel(
                Text(str(raw_text), style=self._style("gray")),
                title="Raw Model Output",
                border_style=self._style("gray"),
                box=box.ROUNDED,
                padding=(0, 1),
            )
            blocks.append(raw_panel)

        self.console.print(
            self._indent(depth),
            Panel(
                Group(*blocks),
                title=title,
                border_style=self._style(color),
                box=box.ROUNDED,
                padding=(1, 2),
            ),
        )