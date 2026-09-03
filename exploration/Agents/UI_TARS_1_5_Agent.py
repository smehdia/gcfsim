from Agents.BaseAgent import BaseAgent
from Agents.UI_TARS_1_5.UI_TARS_1_5 import UITARS_1_5


class UITars_1_5_Agent(BaseAgent):
    """Adapter for UI-TARS."""

    def _build_client(self):
        return UITARS_1_5(
            base_url=self.agent_url,
            model=self.model_id,
            runtime_conf=self.runtime_conf,
            debugger=self.debugger,
        )

    @property
    def assistant_history(self):
        return self.agent_client.assistant_history

    @assistant_history.setter
    def assistant_history(self, value):
        self.agent_client.assistant_history = value