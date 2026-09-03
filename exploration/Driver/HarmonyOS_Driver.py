import json
import os
import re
import subprocess
import tempfile
import uuid
from typing import Optional, Tuple

import cv2
import numpy as np
import xml.etree.ElementTree as ET

import logging
from Driver.BaseDriver import BaseDriver

_IME_WINDOW_RE = re.compile(
    r"(?i)(?:softKeyboard|KeyboardPanel|KeyboardDialog|inputmethod|input.?method)"
)
_IME_BOUNDS_RE = re.compile(
    r"\[\s*\d+\s+\d+\s+(\d+)\s+(\d+)\s*\]"
)
_BUNDLE_NAME_RE = re.compile(r"bundle name\s*\[([^\]]+)\]")
_FOCUS_WINDOW_RE = re.compile(r"Focus window:\s*(\d+)")
_WMS_SKIP_WIN_RE = re.compile(
    r"^(?:SCB|BackgroundBlur|TransparentView|ARK_APP_SUBWINDOW)",
    re.I,
)
# System / shell surfaces that can also report FOREGROUND alongside the user app.
_SYSTEM_FOREGROUND_BUNDLE_RE = re.compile(
    r"^(?:com\.ohos\.sceneboard|com\.ohos\.systemui|com\.ohos\.launcher|"
    r"com\.ohos\.medialibrary(?:\..*)?)$"
)
# dumpDisplayInfo has no WxH on Harmony NEXT; prefer RenderService / Display -a -a.
_RENDER_RES_RE = re.compile(
    r"(?i)(?:render resolution|physical resolution|activeMode)\s*=\s*(\d+)\s*x\s*(\d+)"
)
_DISPLAY_WIDTH_RE = re.compile(r"(?im)^Width:\s*(\d+)\s*$")
_DISPLAY_HEIGHT_RE = re.compile(r"(?im)^Height:\s*(\d+)\s*$")
_LEGACY_WH_RE = re.compile(r"width\s*=\s*(\d+).*height\s*=\s*(\d+)", re.DOTALL | re.IGNORECASE)
_WMS_PANEL_RE = re.compile(
    r"SCBScenePanel\S*.*?\[\s*0\s+0\s+(\d+)\s+(\d+)\s*\]"
)
# In-app screenshot-share sheets (同花顺/微博/小红书 etc.) after snapshot_display.
_SCREENSHOT_SHARE_RE = re.compile(
    r"(保存图片|去分享吧|分享到|长按识别二维码|分享图片|微信好友)"
)
_BOUNDS_RE = re.compile(r'"bounds"\s*:\s*"\[(\d+),(\d+)\]\[(\d+),(\d+)\]"')


class HarmonyDriver(BaseDriver):
    def __init__(self, settings: dict, agent=None) -> None:
        super().__init__(settings, agent)
        self._screen_size_cache: Optional[Tuple[int, int]] = None

    def wait(self, seconds: float | None = None) -> None:
        # Shorter default settle than Android (1.0s); override with settings action_wait_s.
        if seconds is None:
            seconds = float(self.settings.get("action_wait_s", 0.5))
        super().wait(seconds)

    def _hdc_prefix(self):
        if not self.device_id:
            raise ValueError("HarmonyDriver requires device_id for hdc.")
        return ["hdc", "-t", self.device_id]

    def _hdc_out(self, args, timeout=20) -> bytes:
        return subprocess.check_output(self._hdc_prefix() + args, stderr=subprocess.STDOUT, timeout=timeout)

    def _hdc_run(self, args, timeout=20) -> None:
        subprocess.run(self._hdc_prefix() + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)

    def check_device(self) -> bool:
        try:
            out = subprocess.check_output(["hdc", "list", "targets"], stderr=subprocess.STDOUT, timeout=5).decode(
                "utf-8", "ignore"
            )
            return bool(self.device_id and self.device_id in out)
        except Exception:
            return False

    def is_keyboard_open(self) -> bool:
        """
        Fast keyboard check via WindowManagerService (~0.1–0.3s).

        Avoids uitest dumpLayout (~1s+), which previously ran on every screenshot.
        A keyboard window is treated as open when ZOrd >= 0 and bounds are non-trivial.
        """
        try:
            out = self._hdc_out(
                ["shell", "hidumper", "-s", "WindowManagerService", "-a", "-a"],
                timeout=5,
            ).decode("utf-8", "ignore")
        except Exception:
            return False

        for line in out.splitlines():
            if not _IME_WINDOW_RE.search(line):
                continue
            bracket = line.find("[")
            if bracket == -1:
                continue
            head = line[:bracket].split()
            if len(head) < 2:
                continue
            try:
                zord = int(head[-2])
            except ValueError:
                continue
            m = _IME_BOUNDS_RE.search(line)
            if not m:
                continue
            width, height = int(m.group(1)), int(m.group(2))
            if zord < 0 or width <= 50 or height <= 80:
                continue
            # Hidden IME surfaces report fullscreen bounds even when closed.
            try:
                _w, sh = self.get_screen_size()
            except Exception:
                sh = 2688
            if height >= int(sh * 0.85):
                continue
            return True
        return False

    def _dump_layout_raw(self) -> str:
        remote = "/data/local/tmp/window_dump.xml"
        try:
            return self._hdc_out(
                ["shell", f"uitest dumpLayout -p {remote} >/dev/null && cat {remote}"],
                timeout=30,
            ).decode("utf-8", "ignore")
        except Exception:
            return ""

    @staticmethod
    def _share_cancel_xy(raw: str) -> Optional[Tuple[int, int]]:
        """Center of the fat bottom 取消 button on a screenshot-share sheet."""
        if not raw:
            return None
        for token in ('"text":"取消"', '"originalText":"取消"'):
            start = 0
            while True:
                idx = raw.find(token, start)
                if idx < 0:
                    break
                window = raw[max(0, idx - 700) : idx + 80]
                m = _BOUNDS_RE.search(window)
                if m:
                    x1, y1, x2, y2 = (int(m.group(i)) for i in range(1, 5))
                    if (x2 - x1) >= 400 and y1 >= 1800:
                        return (x1 + x2) // 2, (y1 + y2) // 2
                start = idx + len(token)
        return None

    def _dismiss_screenshot_share_if_present(self) -> None:
        """
        snapshot_display is treated as a user screenshot by many CN apps (Tonghuashun,
        Weibo, XHS). They then open a share poster + 微信/保存图片 sheet. The *next*
        capture would graph that overlay as the current page. Dismiss it in-app only.
        """
        if not self.settings.get("dismiss_screenshot_share", True):
            return
        # Overlay is not in the a11y tree immediately after snapshot_display.
        self.wait(0.8)
        pkg = self.settings.get("appPackage")
        dbg = getattr(self.agent, "debugger", None) if self.agent else None
        for _ in range(3):
            if pkg and self.get_foreground_package() != pkg:
                break
            raw = self._dump_layout_raw()
            if not raw or not _SCREENSHOT_SHARE_RE.search(raw):
                break
            xy = self._share_cancel_xy(raw)
            msg = f"Harmony screenshot-share overlay detected; dismissing via {'取消 click' if xy else 'Back'}"
            logging.info(msg)
            if dbg:
                dbg.log(msg, color="yellow")
            if xy:
                self.click(*xy)
            else:
                self.back()
            self.wait(0.45)

    def take_screenshot(self):
        # Match Android: optional IME dismiss before capture. Harmony previously always
        # paid for a full layout dump here; that alone made reset ~5x slower.
        if self.settings.get("dismiss_keyboard_on_screenshot", True) and self.is_keyboard_open():
            self.back()

        tag = uuid.uuid4().hex[:8]
        remote = f"/data/local/tmp/__agentnav_{tag}.jpeg"
        local = os.path.join(tempfile.gettempdir(), f"__agentnav_{tag}.jpeg")
        try:
            self._hdc_run(["shell", "snapshot_display", "-f", remote], timeout=20)
            self._hdc_run(["file", "recv", remote, local], timeout=30)
            img = cv2.imread(local, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                raise RuntimeError("Failed to read screenshot from Harmony device.")
            self._dismiss_screenshot_share_if_present()
            return img
        finally:
            try:
                os.remove(local)
            except OSError:
                pass
            # Unique remote names already avoid collisions; skip sync rm (extra hdc RTT).

    def get_foreground_package(self) -> str | None:
        """
        Foreground bundle via `aa dump -l` (~0.2s). Some Harmony apps (e.g. Qunar)
        stay focused in WMS but never report `app state #FOREGROUND`, so fall back
        to the focused window pid → process name.
        """
        pkg = self._foreground_from_aa_dump()
        if pkg:
            return pkg
        return self._foreground_from_wms()

    def _foreground_from_aa_dump(self) -> Optional[str]:
        try:
            out = self._hdc_out(["shell", "aa", "dump", "-l"], timeout=10).decode("utf-8", "ignore")
        except Exception:
            return None

        current_bundle: Optional[str] = None
        for line in out.splitlines():
            m = _BUNDLE_NAME_RE.search(line)
            if m:
                current_bundle = m.group(1).strip()
                continue
            if "app state #FOREGROUND" not in line or not current_bundle:
                continue
            if _SYSTEM_FOREGROUND_BUNDLE_RE.match(current_bundle):
                continue
            return current_bundle
        return None

    def _foreground_from_wms(self) -> Optional[str]:
        try:
            wms = self._hdc_out(
                ["shell", "hidumper", "-s", "WindowManagerService", "-a", "-a"],
                timeout=5,
            ).decode("utf-8", "ignore")
        except Exception:
            return None

        focus_id: Optional[str] = None
        for line in wms.splitlines():
            m = _FOCUS_WINDOW_RE.search(line)
            if m:
                focus_id = m.group(1)
                break
        if not focus_id:
            return None

        pid: Optional[int] = None
        for line in wms.splitlines():
            parts = line.split()
            if len(parts) < 8:
                continue
            if parts[3] != focus_id:
                continue
            if _WMS_SKIP_WIN_RE.match(parts[0]):
                continue
            try:
                pid = int(parts[2])
            except ValueError:
                continue
            break
        if pid is None:
            return None

        try:
            ps = self._hdc_out(["shell", "ps", "-A", "-o", "pid,args"], timeout=5).decode("utf-8", "ignore")
        except Exception:
            return None
        pid_s = str(pid)
        for line in ps.splitlines():
            bits = line.strip().split(None, 1)
            if len(bits) != 2 or bits[0] != pid_s:
                continue
            proc = bits[1].strip().split(":")[0]
            if not proc or _SYSTEM_FOREGROUND_BUNDLE_RE.match(proc):
                return None
            return proc
        return None

    def close_application(self) -> None:
        bundle = self.settings["appPackage"]  
        self._hdc_run(["shell", "aa", "force-stop", bundle], timeout=10)

    def get_xml_layout(self) -> str:
        """
        Dump HarmonyOS layout via uitest and normalize to Android-style hierarchy XML.
        """
        raw = self._dump_layout_raw()

        xml_ok = self._normalize_hierarchy_xml(raw)
        if xml_ok.strip():
            return xml_ok

        converted = self._harmony_json_to_android_xml(raw)
        if converted.strip():
            return converted
        return ""

    @staticmethod
    def _normalize_hierarchy_xml(raw: str) -> str:
        s = (raw or "").strip()
        return s if s.startswith("<?xml") else ""

    @staticmethod
    def _harmony_json_to_android_xml(raw: str) -> str:
        """
        Convert HarmonyOS uitest JSON tree into Android-like XML (`hierarchy` + `node` elements).
        """
        if not raw:
            return ""
        try:
            data = json.loads(raw)
        except Exception:
            return ""

        def _as_bool_str(v) -> str:
            s = str(v if v is not None else "").strip().lower()
            return "true" if s in ("1", "true", "yes") else "false"

        def _to_android_node(node) -> ET.Element:
            if isinstance(node, dict):
                attrs = node.get("attributes", {}) if isinstance(node.get("attributes", {}), dict) else {}
                children = node.get("children", []) if isinstance(node.get("children", []), list) else []
            else:
                attrs, children = {}, []

            klass = str(attrs.get("class") or attrs.get("type") or "node")
            out = ET.Element("node")
            out.set("class", klass)
            out.set("text", str(attrs.get("text") or attrs.get("originalText") or ""))
            out.set("content-desc", str(attrs.get("description") or attrs.get("accessibilityId") or ""))
            out.set("resource-id", str(attrs.get("id") or attrs.get("key") or ""))
            out.set("package", str(attrs.get("bundleName") or ""))
            out.set("bounds", str(attrs.get("bounds") or ""))
            out.set("clickable", _as_bool_str(attrs.get("clickable")))
            out.set("long-clickable", _as_bool_str(attrs.get("longClickable")))
            out.set("enabled", _as_bool_str(attrs.get("enabled")))
            out.set("focusable", _as_bool_str(attrs.get("focused")))
            out.set("focused", _as_bool_str(attrs.get("focused")))
            out.set("checkable", _as_bool_str(attrs.get("checkable")))
            out.set("checked", _as_bool_str(attrs.get("checked")))
            out.set("selected", _as_bool_str(attrs.get("selected")))
            out.set("scrollable", _as_bool_str(attrs.get("scrollable")))

            for ch in children:
                out.append(_to_android_node(ch))
            return out

        roots = data if isinstance(data, list) else [data]
        hierarchy = ET.Element("hierarchy")
        for r in roots:
            hierarchy.append(_to_android_node(r))
        return ET.tostring(hierarchy, encoding="unicode")

    def get_current_app_id(self) -> Optional[str]:
        return self.get_foreground_package()

    def click(self, x: int, y: int) -> None:
        self._hdc_run(["shell", "uitest", "uiInput", "click", str(int(x)), str(int(y))], timeout=10)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None:
        # uitest uiInput swipe expects swipeVelocityPps_ (px/s, 200–40000), not duration_ms.
        distance = max(abs(int(x2) - int(x1)), abs(int(y2) - int(y1)))
        ms = max(int(duration_ms), 1)
        if distance > 0:
            velocity_pps = int(round(distance / (ms / 1000.0)))
        else:
            velocity_pps = 600  # Harmony default for same-point gestures (e.g. long press)
        velocity_pps = max(200, min(40000, velocity_pps))
        self._hdc_run(
            ["shell", "uitest", "uiInput", "swipe", str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), str(velocity_pps)],
            timeout=15,
        )

    def type_text(self, text: str) -> None:
        self._hdc_run(["shell", "uitest", "uiInput", "text", str(text)], timeout=15)

    def back(self) -> None:
        self._hdc_run(["shell", "uitest", "uiInput", "keyEvent", "Back"], timeout=10)

    def home(self) -> None:
        self._hdc_run(["shell", "uitest", "uiInput", "keyEvent", "Home"], timeout=10)

    @staticmethod
    def _parse_screen_size(text: str) -> Optional[Tuple[int, int]]:
        if not text:
            return None
        m = _RENDER_RES_RE.search(text)
        if m:
            return int(m.group(1)), int(m.group(2))
        mw, mh = _DISPLAY_WIDTH_RE.search(text), _DISPLAY_HEIGHT_RE.search(text)
        if mw and mh:
            return int(mw.group(1)), int(mh.group(1))
        m = _LEGACY_WH_RE.search(text)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = _WMS_PANEL_RE.search(text)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    def get_screen_size(self) -> Tuple[int, int]:
        if self._screen_size_cache is not None:
            return self._screen_size_cache
        dumps = (
            ["shell", "hidumper", "-s", "RenderService", "-a", "screen"],
            ["shell", "hidumper", "-s", "DisplayManagerService", "-a", "-a"],
            ["shell", "hidumper", "-s", "WindowManagerService", "-a", "-a"],
            ["shell", "hidumper", "-s", "DisplayManagerService", "-a", "dumpDisplayInfo"],
        )
        for args in dumps:
            try:
                out = self._hdc_out(args, timeout=10).decode("utf-8", "ignore")
            except Exception:
                continue
            parsed = self._parse_screen_size(out)
            if parsed and parsed[0] > 0 and parsed[1] > 0:
                self._screen_size_cache = parsed
                return self._screen_size_cache
        logging.warning("Harmony get_screen_size failed to parse display size; using 1080x2400 fallback")
        self._screen_size_cache = (1080, 2400)
        return self._screen_size_cache

    def run_application(self) -> None:
        pkg = self.settings["appPackage"]
        ability = self.settings.get("appActivity", "EntryAbility")
        module = self.settings.get("appModule") or "entry"
        # Prefer explicit module (-m). Some Harmony apps fail or land wrong without it.
        cmd = ["shell", "aa", "start", "-a", str(ability), "-b", str(pkg), "-m", str(module)]
        try:
            out = self._hdc_out(cmd, timeout=15).decode("utf-8", "ignore")
        except subprocess.CalledProcessError as exc:
            out = (exc.output or b"").decode("utf-8", "ignore")
            # Fallback without -m for older bundles
            try:
                out2 = self._hdc_out(
                    ["shell", "aa", "start", "-a", str(ability), "-b", str(pkg)],
                    timeout=15,
                ).decode("utf-8", "ignore")
                out = out2
            except subprocess.CalledProcessError as exc2:
                detail = (exc2.output or b"").decode("utf-8", "ignore") or out
                raise RuntimeError(f"Failed to start Harmony app {pkg}/{ability}: {detail}") from exc2
        if "error" in out.lower() and "no error" not in out.lower() and "successfully" not in out.lower():
            raise RuntimeError(f"Failed to start Harmony app {pkg}/{ability}: {out.strip()}")


    def get_app_version(self):
        pkg = self.settings["appPackage"]

        def extract_all_fields(text: str, field: str):
            pattern = rf'"{re.escape(field)}"\s*:\s*("([^"]*)"|-?\d+|true|false|null)'
            values = []

            for m in re.finditer(pattern, text):
                raw = m.group(1)

                if raw.startswith('"') and raw.endswith('"'):
                    value = raw[1:-1]
                elif raw == "true":
                    value = True
                elif raw == "false":
                    value = False
                elif raw == "null":
                    value = None
                elif re.fullmatch(r"-?\d+", raw):
                    value = int(raw)
                else:
                    value = raw

                values.append(value)

            return values

        def last_valid(text: str, field: str):
            values = extract_all_fields(text, field)

            for v in reversed(values):
                if v not in (None, "", 0):
                    return v

            return values[-1] if values else None

        try:
            dump = self._hdc_out(
                ["shell", "bm", "dump", "-n", pkg],
                timeout=20,
            ).decode("utf-8", "ignore")

            version_name = last_valid(dump, "versionName")
            version_code = last_valid(dump, "versionCode")
            main_ability = last_valid(dump, "mainAbility")
            main_element_name = last_valid(dump, "mainElementName")

            module_names = sorted(set(
                m for m in extract_all_fields(dump, "moduleName")
                if m not in (None, "", 0)
            ))

            ability_name = main_element_name or main_ability
            module_name = "entry" if "entry" in module_names else (
                module_names[0] if module_names else None
            )

            return {
                "package_name": pkg,
                "version_name": version_name,
                "version_code": version_code,
                "main_ability": main_ability,
                "main_element_name": main_element_name,
                "module_names": module_names,
                "entry": {
                    "bundle_name": pkg,
                    "module_name": module_name,
                    "ability_name": ability_name,
                    "launch_target": (
                        f"{pkg}/{module_name}/{ability_name}"
                        if module_name and ability_name
                        else None
                    ),
                    "launch_command": (
                        f"hdc -t {self.device_id} shell aa start -b {pkg} -m {module_name} -a {ability_name}"
                        if module_name and ability_name
                        else None
                    ),
                },
            }

        except Exception as e:
            logging.exception("Failed to get Harmony app version/info")
            return {
                "package_name": pkg,
                "version_name": None,
                "version_code": None,
                "main_ability": None,
                "main_element_name": None,
                "module_names": [],
                "entry": None,
                "error": str(e),
            }
