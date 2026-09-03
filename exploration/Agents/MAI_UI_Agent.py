from Agents.BaseAgent import BaseAgent
from Agents.MAI_UI.MAI_UI import MAI_UI


class MAI_UI_Agent(BaseAgent):
    """Adapter for MAI-UI."""

    def _build_client(self):
        return MAI_UI(
            llm_base_url=self.agent_url,
            model_name=self.model_id or "MAI-UI-8B",
            runtime_conf=self.runtime_conf,
            debugger=self.debugger,
        )