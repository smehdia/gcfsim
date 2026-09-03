from typing import Optional

from Agents.BaseAgent import BaseAgent
from Agents.MAI_UI_Agent import MAI_UI_Agent
from Agents.UI_TARS_1_5_Agent import UITars_1_5_Agent

def build_agent(
    model_name: str,
    url: str,
    agent_settings: Optional[dict] = None,
    debugger=None,
) -> BaseAgent:
    model_name = str(model_name).strip().lower()

    if model_name == "ui_tars":
        return UITars_1_5_Agent(
            url=url,
            agent_settings=agent_settings,
            debugger=debugger,
        )

    if model_name == "mai_ui":
        return MAI_UI_Agent(
            url=url,
            agent_settings=agent_settings,
            debugger=debugger,
        )

    raise ValueError("model_name must be 'ui_tars' or 'mai_ui'")