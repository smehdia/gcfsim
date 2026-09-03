from typing import Optional

from Driver.BaseDriver import BaseDriver
from Driver.Android_Driver import AndroidDriver
from Driver.HarmonyOS_Driver import HarmonyDriver
from Agents.BaseAgent import BaseAgent

def build_driver(
    settings: dict = None,
    agent: BaseAgent = None,
) -> BaseDriver:

    if 'os_name' not in settings:
        raise ValueError("os_name is required")
    if settings['os_name'] not in ("android", "harmony"):
        raise ValueError("os_name must be 'android' or 'harmony'")
    
    platform = settings['os_name']

    if platform == "android":
        return AndroidDriver(
            settings=settings,
            agent=agent
        )

    if platform in {"harmony", "harmonyos"}:
        return HarmonyDriver(
            settings=settings,
            agent=agent
        )

    raise ValueError("platform must be 'android' or 'harmony'")