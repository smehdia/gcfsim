import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple

from Agents.BaseAgent import BaseAgent
from Agents.MAI_UI.MAI_UI import ParsedAction


class BaseDriver(ABC):
    def __init__(
        self,
        settings: dict,
        agent: BaseAgent = None,
    ) -> None:
        self.settings = settings
        self.device_id = settings.get("device_id", None)
        self.agent = agent

        if 'os_name' not in self.settings:
            raise ValueError("os_name is required")
        if self.settings['os_name'] not in ("android", "harmony"):
            raise ValueError("os_name must be 'android' or 'harmony'")
        if self.settings['os_name'] == "android":
            if 'device_id' not in self.settings:
                raise ValueError("device_id is required")
            if 'appPackage' not in self.settings:
                raise ValueError("appPackage is required")
            if 'appActivity' not in self.settings:
                raise ValueError("appActivity is required")
        
    @abstractmethod
    def run_application(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def check_device(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def take_screenshot(self):
        """Return a BGR numpy array (OpenCV format)."""
        raise NotImplementedError

    @abstractmethod
    def get_xml_layout(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_current_app_id(self) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def click(self, x: int, y: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 500,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def type_text(self, text: str) -> None:
        raise NotImplementedError



    @abstractmethod
    def back(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def home(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_screen_size(self) -> Tuple[int, int]:
        """Return (width, height) in device pixels."""
        raise NotImplementedError

    @abstractmethod
    def close_application(self) -> None:
        """Force-stop / kill the configured app process."""
        raise NotImplementedError

    @abstractmethod
    def is_keyboard_open(self) -> bool:
        raise NotImplementedError

    def wait(self, seconds: float | None = None) -> None:
        """Settle after an action. Default 1.0s; override via settings `action_wait_s`."""
        if seconds is None:
            seconds = float(self.settings.get("action_wait_s", 1.0))
        time.sleep(max(0.0, float(seconds)))

    @staticmethod
    def _best_xy(parsed: ParsedAction) -> Optional[Tuple[int, int]]:
        oc = parsed.orig_coords or {}
        for key in ("point", "start_point", "start_box"):
            xy = oc.get(key)
            if xy is not None and len(xy) >= 2:
                return int(round(xy[0])), int(round(xy[1]))
        return None

    def _scroll_from_parsed(self, parsed: ParsedAction) -> bool:
        direction = str((parsed.params or {}).get("direction", "down")).strip().lower()
        if direction not in ("up", "down", "left", "right"):
            direction = "down"
        w, h = self.get_screen_size()
        xy = self._best_xy(parsed)
        cx, cy = (xy if xy else (w // 2, h // 2))
        dx, dy = int(w * 0.35), int(h * 0.35)
        if direction == "down":
            x1, y1, x2, y2 = cx, cy, cx, max(1, cy - dy)
        elif direction == "up":
            x1, y1, x2, y2 = cx, cy, cx, min(h - 1, cy + dy)
        elif direction == "left":
            x1, y1, x2, y2 = cx, cy, max(1, cx - dx), cy
        else:
            x1, y1, x2, y2 = cx, cy, min(w - 1, cx + dx), cy
        self.swipe(x1, y1, x2, y2, duration_ms=400)
        self.wait()
        return True

    def execute_action(self, parsed: ParsedAction) -> bool:
        """Execute a ParsedAction from MAI-UI / UI-TARS. Returns False for finished/unknown."""
        at = str(parsed.action_type or "").strip().lower()
        params = parsed.params or {}

        if at in ("finished", "finish"):
            return False

        if at in ("press_back", "back"):
            self.back()
            self.wait()
            return True

        if at in ("press_home", "home"):
            self.home()
            self.wait()
            return True

        if at == "wait":
            self.wait()
            return True

        if at == 'open_app':
            self.run_application()
            self.wait()
            return True

        if at in ("click", "left_single", "left_double"):
            xy = self._best_xy(parsed)
            if not xy:
                return False
            self.click(*xy)
            self.wait()
            return True

        if at == "long_press":
            xy = self._best_xy(parsed)
            if not xy:
                return False
            x, y = xy
            self.swipe(x, y, x, y, duration_ms=600)
            self.wait()
            return True

        if at in ("scroll", "swipe"):
            return self._scroll_from_parsed(parsed)

        if at == "type":
            text = str(params.get("content", "") or "")
            if not text:
                return False
            self.type_text(text)
            self.wait()
            return True

        if at == "drag":
            oc = parsed.orig_coords or {}
            sp = oc.get("start_point") or oc.get("point")
            ep = oc.get("end_point")
            if not sp or not ep:
                return False
            self.swipe(
                int(round(sp[0])), int(round(sp[1])),
                int(round(ep[0])), int(round(ep[1])),
                duration_ms=450,
            )
            self.wait()
            return True

        return False


    def reset_to_start_page(self) -> None:
        pkg = self.settings.get("appPackage")
        for _ in range(5):
            # Don't Back from desktop / another app — on Harmony that opens
            # recents/control-center and leaves exploration on a weird screen.
            if pkg and self.get_foreground_package() != pkg:
                break
            self.back()
            self.wait()

        self.close_application()
        self.wait()
        self.run_application()
        self.wait()
        # Extra settle after launch (splash / cold start); default 0s.
        startup_s = float(self.settings.get("app_startup_time", 0) or 0)
        if startup_s > 0:
            self.wait(startup_s)
        # we do one scroll up to make sure we are on top of the page
        if not self.settings.get("skip_scroll_up_on_reset", False):
            w, h = self.get_screen_size()
            cx, cy = w // 2, h // 2
            dy = int(h * 0.35)
            self.swipe(cx, cy, cx, min(h - 1, cy + dy), duration_ms=400)
            self.wait()

    
        reset_instruction = self.settings.get("reset_instruction")
        if not self.agent or not reset_instruction:
            return

        finish_flag = False
        max_step_reset = 5
        # reset memory of agent
        self.agent.clear_history()
        for _ in range(max_step_reset):
            step_result, _ = self.agent.step(reset_instruction, self.take_screenshot())
            parsed = step_result[0] if isinstance(step_result, tuple) else step_result
            if str(getattr(parsed, "action_type", "") or "").strip().lower() in ("finished", "finish"):
                finish_flag = True
                break
            else:
                dbg = getattr(self.agent, "debugger", None)
                if dbg:
                    dbg.log(
                        f"Reset agent action: {getattr(parsed, 'action_type', None)} "
                        f"{getattr(parsed, 'params', None)} {getattr(parsed, 'orig_coords', None)}",
                        color="yellow",
                    )
                self.execute_action(parsed)
            self.wait()

        if not finish_flag:
            self.run_application()
            self.wait()


    def get_non_loading_page(self) -> bool:
        # this method requires agent to be set
        if not self.agent:
            raise ValueError("Agent is not set")

        # At most one extra capture. Extra Harmony snapshots re-open in-app
        # screenshot-share sheets (Tonghuashun 去分享吧 / 保存图片).
        self.agent.clear_history()
        screenshot = self.take_screenshot()
        step_result, _ = self.agent.step(
            instruction="If the app UI is visible (not a loading/splash spinner), terminate. Do not click, swipe, or press back.",
            screenshot=screenshot,
        )
        parsed = step_result[0] if isinstance(step_result, tuple) else step_result
        if str(getattr(parsed, "action_type", "") or "").strip().lower() in ("finished", "finish"):
            return screenshot
        self.wait(1.0)
        return self.take_screenshot()
        
