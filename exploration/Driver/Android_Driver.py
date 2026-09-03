import re
import subprocess
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import uiautomator2 as u2

from Driver.BaseDriver import BaseDriver

import xml.etree.ElementTree as ET
from typing import Iterable

_IME_PKG_RE = re.compile(
    r'package="[^"]*(?:inputmethod|\.ime\.|keyboard|honeyboard|input-method)',
    re.I,
)
_IME_CLASS_RE = re.compile(
    r'class="[^"]*(?:keyboard|inputmethod|softinput|keyboardpanel)',
    re.I,
)


def keyboard_visible_in_xml(xml: str) -> bool:
    if not xml or "</hierarchy>" not in xml:
        return False
    return bool(_IME_PKG_RE.search(xml) or _IME_CLASS_RE.search(xml))


class AndroidDriver(BaseDriver):
    def _adb_prefix(self):
        return ["adb"] + (["-s", self.device_id] if self.device_id else [])

    def _adb_out(self, args, timeout=15) -> bytes:
        return subprocess.check_output(self._adb_prefix() + args, stderr=subprocess.STDOUT, timeout=timeout)

    def _adb_run(self, args, timeout=15) -> None:
        subprocess.run(self._adb_prefix() + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)


    def run_application(self) -> None:
        pkg = self.settings["appPackage"]
        activity = self.settings.get("appActivity") or ""
        use_launcher = self.settings.get("use_launcher_intent") or ("$" in activity)
        if use_launcher:
            self._adb_run([
                "shell", "am", "start", "-a", "android.intent.action.MAIN",
                "-c", "android.intent.category.LAUNCHER", "-p", pkg,
            ], timeout=10)
        else:
            self._adb_run(["shell", "am", "start", "-W", f"{pkg}/{activity}"], timeout=10)        



    def check_device(self) -> bool:
        try:
            out = self._adb_out(["get-state"], timeout=5).decode("utf-8", "ignore").strip()
            return out == "device"
        except Exception:
            return False

    def take_screenshot(self):
        # Exploration historically dismissed the IME before capture. On Amazon (and
        # some other apps) BACK from an open search keyboard exits to the launcher,
        # so GRPO / navigation must disable this via settings.
        if self.settings.get("dismiss_keyboard_on_screenshot", True) and self.is_keyboard_open():
            self.back()
        raw = self._adb_out(["exec-out", "screencap", "-p"], timeout=20)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode screenshot from adb screencap.")
        return img


    def get_foreground_package(self) -> str | None:
        out = self._adb_out([
            "shell",
            "dumpsys window | grep -E 'mCurrentFocus|mFocusedApp|topResumedActivity'"
        ], timeout=5).decode("utf-8", "ignore")
        m = re.search(r"\b([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)+)/", out)
        if m:
            return m.group(1)
        return None


    def get_xml_layout(self) -> str:
        """
        Minimal robust XML fetcher:
        - tries uiautomator2
        - falls back to raw ADB /dev/tty
        - falls back to raw ADB file dump
        - filters to foreground/app package
        - rejects very thin WebView shells when possible
        - never returns stale ADB XML
        """

        max_attempts = int(self.settings.get("xml_layout_max_attempts", 3))
        retry_sleep_s = float(self.settings.get("xml_layout_retry_sleep_s", 0.5))

        min_nodes_if_webview = int(self.settings.get("xml_layout_min_nodes_if_webview", 80))
        min_webview_descendants = int(self.settings.get("xml_layout_min_webview_descendants", 10))

        app_pkg = str(self.settings.get("appPackage") or "").strip()

        def cut_xml(raw: str) -> str:
            raw = raw or ""
            start = raw.find("<?xml")
            if start == -1:
                start = raw.find("<hierarchy")
            end = raw.rfind("</hierarchy>")
            if start != -1 and end != -1:
                return raw[start : end + len("</hierarchy>")].strip()
            return ""

        def filter_xml_by_packages(xml: str, keep_packages: Iterable[str]) -> str:
            keep_packages = {p for p in keep_packages if p}
            if not keep_packages:
                return xml

            root = ET.fromstring(xml)
            new_root = ET.Element(root.tag, root.attrib)

            def clone_if_relevant(node: ET.Element):
                children = []
                for child in list(node):
                    cloned = clone_if_relevant(child)
                    if cloned is not None:
                        children.append(cloned)

                node_pkg = node.attrib.get("package", "")
                if node_pkg not in keep_packages and not children:
                    return None

                new_node = ET.Element(node.tag, node.attrib)
                for child in children:
                    new_node.append(child)
                return new_node

            for child in list(root):
                cloned = clone_if_relevant(child)
                if cloned is not None:
                    new_root.append(cloned)

            return ET.tostring(new_root, encoding="unicode")

        def layout_stats(xml: str) -> tuple[int, int, bool]:
            root = ET.fromstring(xml)
            nodes = [n for n in root.iter("node")]

            has_webview = False
            max_webview_desc = 0

            for node in nodes:
                cls = node.attrib.get("class", "").lower()
                if "webview" in cls:
                    has_webview = True
                    descendants = sum(1 for _ in node.iter("node")) - 1
                    max_webview_desc = max(max_webview_desc, descendants)

            return len(nodes), max_webview_desc, has_webview

        def is_good_xml(xml: str) -> bool:
            if not xml or "<hierarchy" not in xml or "</hierarchy>" not in xml:
                return False

            try:
                node_count, webview_desc, has_webview = layout_stats(xml)
            except ET.ParseError:
                return False

            if node_count == 0:
                return False

            # Reject common WebView shell: only outer app containers, no real inner tree.
            if has_webview:
                if node_count < min_nodes_if_webview:
                    return False
                if webview_desc < min_webview_descendants:
                    return False

            return True

        def score_xml(xml: str) -> tuple[int, int]:
            try:
                node_count, webview_desc, _ = layout_stats(xml)
                return node_count, webview_desc
            except ET.ParseError:
                return 0, 0

        def adb_direct_tty() -> str:
            out = self._adb_run(
                ["exec-out", "uiautomator", "dump", "--compressed", "/dev/tty"],
                timeout=8,
                capture=True,
                text=True,
            ) or ""
            return cut_xml(out)

        def adb_file_dump() -> str:
            remote = "/sdcard/__agentnav_window.xml"

            self._adb_run(["shell", "rm", "-f", remote], timeout=5, capture=False)

            self._adb_run(
                ["shell", "uiautomator", "dump", "--compressed", remote],
                timeout=12,
                capture=False,
            )

            exists = self._adb_run(
                ["shell", "test", "-f", remote, "&&", "echo", "OK", "||", "echo", "MISSING"],
                timeout=5,
                capture=True,
                text=True,
            ) or ""

            if "OK" not in exists:
                return ""

            out = self._adb_run(
                ["exec-out", "cat", remote],
                timeout=8,
                capture=True,
                text=True,
            ) or ""

            return cut_xml(out)

        if not getattr(self, "_u2_device", None):
            try:
                self._u2_device = u2.connect(self.device_id) if self.device_id else u2.connect()
            except Exception:
                self._u2_device = None

        best_xml = ""
        best_score = (0, 0)

        for _ in range(max_attempts):
            fg_pkg = self.get_foreground_package()
            keep_pkgs = {app_pkg, fg_pkg}

            candidates = []

            if getattr(self, "_u2_device", None):
                try:
                    candidates.append(self._u2_device.dump_hierarchy(compressed=False))
                except Exception:
                    pass

                try:
                    candidates.append(self._u2_device.dump_hierarchy(compressed=True))
                except Exception:
                    pass

            try:
                candidates.append(adb_direct_tty())
            except Exception:
                pass

            try:
                candidates.append(adb_file_dump())
            except Exception:
                pass

            for raw in candidates:
                xml = cut_xml(raw) if raw else ""
                if not xml:
                    continue

                try:
                    xml = filter_xml_by_packages(xml, keep_pkgs)
                except ET.ParseError:
                    continue

                if not xml or "<hierarchy" not in xml:
                    continue

                current_score = score_xml(xml)
                if current_score > best_score:
                    best_xml = xml
                    best_score = current_score

                if is_good_xml(xml):
                    return xml

            time.sleep(retry_sleep_s)

        # Last resort: return the best parseable XML instead of crashing,
        # but only if it is real XML. Your downstream code can treat it as weak XML.
        if best_xml and "<hierarchy" in best_xml and "</hierarchy>" in best_xml:
            return best_xml

        raise ValueError("XML can not be obtained")

    def get_current_app_id(self) -> Optional[str]:
        try:
            out = self._adb_out(["shell", "dumpsys", "window", "windows"], timeout=10).decode("utf-8", "ignore")
            m = re.search(r"mCurrentFocus=.*?\s([\w.]+)/(?:[\w.$]+)", out)
            return m.group(1) if m else None
        except Exception:
            return None

    def click(self, x: int, y: int) -> None:
        self._adb_run(["shell", "input", "tap", str(int(x)), str(int(y))], timeout=10)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None:
        self._adb_run(
            ["shell", "input", "swipe", str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), str(int(duration_ms))],
            timeout=15,
        )

    def type_text(self, text: str) -> None:
        # adb input text requires escaping spaces and some symbols; keep minimal.
        safe = str(text).replace(" ", "%s")
        self._adb_run(["shell", "input", "text", safe], timeout=15)

    def back(self) -> None:
        self._adb_run(["shell", "input", "keyevent", "4"], timeout=10)

    def home(self) -> None:
        self._adb_run(["shell", "input", "keyevent", "3"], timeout=10)

    def get_screen_size(self) -> Tuple[int, int]:
        out = self._adb_out(["shell", "wm", "size"], timeout=5).decode("utf-8", "ignore")
        m = re.search(r"(\d+)x(\d+)", out)
        if not m:
            raise RuntimeError(f"Could not parse screen size from: {out!r}")
        return int(m.group(1)), int(m.group(2))

    def close_application(self) -> None:
        pkg = self.settings["appPackage"]
        self._adb_run(["shell", "am", "force-stop", pkg], timeout=10)

    def is_keyboard_open(self) -> bool:
        try:
            out = self._adb_out(["shell", "dumpsys", "input_method"], timeout=5).decode(
                "utf-8", "ignore"
            )
        except Exception:
            return False
        if re.search(r"mInputShown\s*=\s*true", out, re.I):
            return True
        if re.search(r"mShowIme\s*=\s*true", out, re.I):
            return True
        if re.search(r"mVisibleState\s*=\s*STATE_VISIBLE", out, re.I):
            return True
        try:
            return keyboard_visible_in_xml(self.get_xml_layout())
        except Exception:
            return False        


    def get_app_version(self):
        """
        Returns meta info like:
        {
            'android_version': '16', 
            'device_name': 'SM-S928W', 
            'package': 'com.google.android.deskclock', 
            'app_version_output': '7.4.3'  # for example
        }
        """
        # Android version
        android_version = self._adb_out(
            ["shell", "getprop", "ro.build.version.release"], timeout=5
        ).decode("utf-8", "ignore").strip()
        # Device name
        device_name = self._adb_out(
            ["shell", "getprop", "ro.product.model"], timeout=5
        ).decode("utf-8", "ignore").strip()
        # Package name (try both app_package and appPackage for compatibility)
        app_package = (
            getattr(self.settings, "app_package", None)
            or getattr(self.settings, "appPackage", None)
            or self.settings.get("app_package")
            or self.settings.get("appPackage")
        )

        # Get app version output using adb shell dumpsys
        app_version_output = None
        if app_package:
            try:
                dumpsys_out = self._adb_out(
                    ["shell", "dumpsys", "package", app_package], timeout=5
                ).decode("utf-8", "ignore")
                # Extract versionName value using regex
                m = re.search(r"versionName=([^\s]+)", dumpsys_out)
                if m:
                    app_version_output = m.group(1)
                else:
                    # Try alternative matching if needed
                    m2 = re.search(r"versionName\s*:\s*([^\s]+)", dumpsys_out)
                    if m2:
                        app_version_output = m2.group(1)
            except Exception:
                app_version_output = None

        return {
            "android_version": android_version,
            "device_name": device_name,
            "package": app_package,
            "app_version_output": app_version_output,
        }