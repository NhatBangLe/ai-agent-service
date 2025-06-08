import logging
from logging import Logger
from typing import Literal, Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from src.config.configurer.agent import AgentConfigurer
from src.util.main import Progress


class Agent:
    _status: Literal["ON", "OFF", "RESTART"]
    _configurer: AgentConfigurer
    _graph: CompiledStateGraph | None
    _checkpointer: BaseCheckpointSaver[Any] | None
    _is_configured: bool
    _logger: Logger

    def __init__(self, configurer: AgentConfigurer):
        self._status = "ON"
        self._configurer = configurer
        self._graph = None
        self._is_configured = False
        self._logger = logging.getLogger(__name__)

    def configure(self, force: bool = False):
        if self._is_configured and not force:
            self._logger.debug("Not forcefully configuring the agent. Skipping...")
            return
        self._logger.info("Configuring agent...")
        self._configurer.configure()
        self._is_configured = True
        self._logger.info("Agent configured successfully!")

    def restart(self):
        """
        Triggers the process of restarting the agent, updates its status, reconfigures,
        and rebuilds its internal graph. The function yields progress updates
        throughout the restart process.

        Returns:
            A string representing the progress of the restart operation.
            `{"status": "RESTARTING", "percentage": 0.0}`, use a new line character to separate lines.
        """
        statuses: list[Progress] = [
            {
                "status": "RESTARTING",
                "percentage": 0.0
            },
            {
                "status": "RESTARTING",
                "percentage": 0.6
            },
            {
                "status": "RESTARTED",
                "percentage": 1.0
            }
        ]
        self._logger.info("Restarting agent...")
        yield str(f'{statuses[0]}\n')

        self._status = "RESTART"
        self._configurer.configure()
        yield str(f'{statuses[1]}\n')
        self._status = "ON"
        yield str(f'{statuses[2]}\n')

        self._logger.info("Agent restarted successfully!")

    @property
    def configurer(self):
        return self._configurer

    @property
    def status(self):
        return self._status

    @property
    def is_configured(self):
        return self._is_configured
