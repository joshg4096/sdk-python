"""Repository session manager implementation."""

import logging
from typing import TYPE_CHECKING, Any, Optional

from ..agent.state import AgentState
from ..tools._tool_helpers import generate_missing_tool_result_content
from ..types.content import Message
from ..types.exceptions import SessionException
from ..types.session import (
    Session,
    SessionAgent,
    SessionMessage,
    SessionType,
)
from .session_manager import SessionManager
from .session_repository import SessionRepository

if TYPE_CHECKING:
    from ..agent.agent import Agent
    from ..experimental.bidi.agent.agent import BidiAgent
    from ..multiagent.base import MultiAgentBase

logger = logging.getLogger(__name__)


class RepositorySessionManager(SessionManager):
    """Session manager for persisting agents in a SessionRepository."""

    def __init__(
        self,
        session_id: str,
        session_repository: SessionRepository,
        **kwargs: Any,
    ):
        """Initialize the RepositorySessionManager.

        If no session with the specified session_id exists yet, it will be created
        in the session_repository.

        Args:
            session_id: ID to use for the session. A new session with this id will be created if it does
                not exist in the repository yet
            session_repository: Underlying session repository to use to store the sessions state.
            **kwargs: Additional keyword arguments for future extensibility.

        """
        self.session_repository = session_repository
        self.session_id = session_id
        session = session_repository.read_session(session_id)
        # Create a session if it does not exist yet
        if session is None:
            logger.debug("session_id=<%s> | session not found, creating new session", self.session_id)
            session = Session(session_id=session_id, session_type=SessionType.AGENT)
            session_repository.create_session(session)

        self.session = session

        # Keep track of the latest message of each agent in case we need to redact it.
        self._latest_agent_message: dict[str, Optional[SessionMessage]] = {}

    def append_message(self, message: Message, agent: "Agent", **kwargs: Any) -> None:
        """Append a message to the agent's session.

        Args:
            message: Message to add to the agent in the session
            agent: Agent to append the message to
            **kwargs: Additional keyword arguments for future extensibility.
        """
        # Calculate the next index (0 if this is the first message, otherwise increment the previous index)
        latest_agent_message = self._latest_agent_message[agent.agent_id]
        if latest_agent_message:
            next_index = latest_agent_message.message_id + 1
        else:
            next_index = 0

        session_message = SessionMessage.from_message(message, next_index)
        self._latest_agent_message[agent.agent_id] = session_message
        self.session_repository.create_message(self.session_id, agent.agent_id, session_message)

    def redact_latest_message(self, redact_message: Message, agent: "Agent", **kwargs: Any) -> None:
        """Redact the latest message appended to the session.

        Args:
            redact_message: New message to use that contains the redact content
            agent: Agent to apply the message redaction to
            **kwargs: Additional keyword arguments for future extensibility.
        """
        latest_agent_message = self._latest_agent_message[agent.agent_id]
        if latest_agent_message is None:
            raise SessionException("No message to redact.")
        latest_agent_message.redact_message = redact_message
        return self.session_repository.update_message(self.session_id, agent.agent_id, latest_agent_message)

    def sync_agent(self, agent: "Agent", **kwargs: Any) -> None:
        """Serialize and update the agent into the session repository.

        Args:
            agent: Agent to sync to the session.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        self.session_repository.update_agent(
            self.session_id,
            SessionAgent.from_agent(agent),
        )

    def initialize(self, agent: "Agent", **kwargs: Any) -> None:
        """Initialize an agent with a session.

        Args:
            agent: Agent to initialize from the session
            **kwargs: Additional keyword arguments for future extensibility.
        """
        if agent.agent_id in self._latest_agent_message:
            raise SessionException("The `agent_id` of an agent must be unique in a session.")
        self._latest_agent_message[agent.agent_id] = None

        session_agent = self.session_repository.read_agent(self.session_id, agent.agent_id)

        if session_agent is None:
            logger.debug(
                "agent_id=<%s> | session_id=<%s> | creating agent",
                agent.agent_id,
                self.session_id,
            )

            session_agent = SessionAgent.from_agent(agent)
            self.session_repository.create_agent(self.session_id, session_agent)
            # Initialize messages with sequential indices
            session_message = None
            for i, message in enumerate(agent.messages):
                session_message = SessionMessage.from_message(message, i)
                self.session_repository.create_message(self.session_id, agent.agent_id, session_message)
            self._latest_agent_message[agent.agent_id] = session_message
        else:
            logger.debug(
                "agent_id=<%s> | session_id=<%s> | restoring agent",
                agent.agent_id,
                self.session_id,
            )
            agent.state = AgentState(session_agent.state)

            session_agent.initialize_internal_state(agent)

            # Restore the conversation manager to its previous state, and get the optional prepend messages
            prepend_messages = agent.conversation_manager.restore_from_session(session_agent.conversation_manager_state)

            if prepend_messages is None:
                prepend_messages = []

            # List the messages currently in the session, using an offset of the messages previously removed
            # by the conversation manager.
            session_messages = self.session_repository.list_messages(
                session_id=self.session_id,
                agent_id=agent.agent_id,
                offset=agent.conversation_manager.removed_message_count,
            )
            if len(session_messages) > 0:
                self._latest_agent_message[agent.agent_id] = session_messages[-1]

            # Restore the agents messages array including the optional prepend messages
            agent.messages = prepend_messages + [session_message.to_message() for session_message in session_messages]

            # Fix broken session histories: https://github.com/strands-agents/sdk-python/issues/859
            agent.messages = self._fix_broken_tool_use(agent.messages)

    def _fix_broken_tool_use(self, messages: list[Message]) -> list[Message]:
        """Fix broken tool use/result pairs in message history.

        This method handles three issues:
        1. Orphaned toolUse messages without corresponding toolResult.
           Before 1.15.0, strands had a bug where they persisted sessions with a potentially broken messages array.
           This method retroactively fixes that issue by adding a tool_result outside of session management.
           After 1.15.0, this bug is no longer present.
        2. Orphaned toolResult messages without corresponding toolUse (e.g., when pagination truncates messages)
        3. Duplicate toolResult blocks for the same toolUseId (e.g., when tools are interrupted and retried)

        Args:
            messages: The list of messages to fix
            agent_id: The agent ID for fetching previous messages
            removed_message_count: Number of messages removed by the conversation manager

        Returns:
            Fixed list of messages with proper tool use/result pairs
        """
        # First, check if the oldest message has orphaned toolResult (no preceding toolUse) and remove it.
        if messages:
            first_message = messages[0]
            if first_message["role"] == "user" and any("toolResult" in content for content in first_message["content"]):
                logger.warning(
                    "Session message history starts with orphaned toolResult with no preceding toolUse. "
                    "This typically happens when messages are truncated due to pagination limits. "
                    "Removing orphaned toolResult message to maintain valid conversation structure."
                )
                messages.pop(0)

        # Check for orphaned toolUse messages (toolUse without corresponding toolResult)
        for index, message in enumerate(messages):
            # Check all but the latest message in the messages array
            # The latest message being orphaned is handled in the agent class
            if index + 1 < len(messages):
                if any("toolUse" in content for content in message["content"]):
                    tool_use_ids = [
                        content["toolUse"]["toolUseId"] for content in message["content"] if "toolUse" in content
                    ]

                    # Check if there are more messages after the current toolUse message
                    tool_result_ids = [
                        content["toolResult"]["toolUseId"]
                        for content in messages[index + 1]["content"]
                        if "toolResult" in content
                    ]

                    missing_tool_use_ids = list(set(tool_use_ids) - set(tool_result_ids))
                    # If there are missing tool use ids, that means the messages history is broken
                    if missing_tool_use_ids:
                        logger.warning(
                            "Session message history has an orphaned toolUse with no toolResult. "
                            "Adding toolResult content blocks to create valid conversation."
                        )
                        # Create the missing toolResult content blocks
                        missing_content_blocks = generate_missing_tool_result_content(missing_tool_use_ids)

                        if tool_result_ids:
                            # If there were any toolResult ids, that means only some of the content blocks are missing
                            messages[index + 1]["content"].extend(missing_content_blocks)
                        else:
                            # The message following the toolUse was not a toolResult, so lets insert it
                            messages.insert(index + 1, {"role": "user", "content": missing_content_blocks})

        # Deduplicate toolResult blocks for the same toolUseId.
        # When tools are interrupted and retried, multiple toolResult blocks may exist
        # for the same toolUseId. Keep only the latest one (which is typically the successful result).
        # This must run AFTER the orphaned toolUse fix to avoid re-adding error results.
        messages = self._deduplicate_tool_results(messages)

        return messages

    def _deduplicate_tool_results(self, messages: list[Message]) -> list[Message]:
        """Deduplicate toolResult blocks that have the same toolUseId.

        When tools are interrupted and retried, the session may contain multiple toolResult
        blocks for the same toolUseId (e.g., an error result from interruption followed by
        a success result after retry). This violates Bedrock's constraint that the number of
        toolResult blocks must match the number of toolUse blocks.

        This method keeps only the latest toolResult for each toolUseId, preferring successful
        results over error results when they appear later in the conversation.

        Args:
            messages: The list of messages to deduplicate

        Returns:
            Messages with duplicate toolResult blocks removed
        """
        # Track all toolResult locations: toolUseId -> list of (message_index, content_index)
        tool_result_locations: dict[str, list[tuple[int, int]]] = {}

        for msg_idx, message in enumerate(messages):
            if message.get("role") == "user":
                for content_idx, content in enumerate(message.get("content", [])):
                    if "toolResult" in content:
                        tool_use_id = content["toolResult"].get("toolUseId")
                        if tool_use_id:
                            if tool_use_id not in tool_result_locations:
                                tool_result_locations[tool_use_id] = []
                            tool_result_locations[tool_use_id].append((msg_idx, content_idx))

        # Find duplicates and mark earlier ones for removal
        # We keep the latest toolResult for each toolUseId
        content_to_remove: set[tuple[int, int]] = set()
        for tool_use_id, locations in tool_result_locations.items():
            if len(locations) > 1:
                logger.warning(
                    "tool_use_id=<%s> | Found %d duplicate toolResult blocks for the same toolUseId. "
                    "This typically happens when tools are interrupted and retried. "
                    "Keeping only the latest toolResult to maintain valid conversation structure.",
                    tool_use_id,
                    len(locations),
                )
                # Mark all but the last location for removal
                for location in locations[:-1]:
                    content_to_remove.add(location)

        if not content_to_remove:
            return messages

        # Remove marked content blocks and clean up empty messages
        result_messages: list[Message] = []
        for msg_idx, message in enumerate(messages):
            if message.get("role") == "user":
                new_content = []
                for content_idx, content in enumerate(message.get("content", [])):
                    if (msg_idx, content_idx) not in content_to_remove:
                        new_content.append(content)

                if new_content:
                    result_messages.append({"role": message["role"], "content": new_content})
                else:
                    logger.debug(
                        "message_index=<%d> | Removing empty user message after toolResult deduplication",
                        msg_idx,
                    )
            else:
                result_messages.append(message)

        return result_messages

    def sync_multi_agent(self, source: "MultiAgentBase", **kwargs: Any) -> None:
        """Serialize and update the multi-agent state into the session repository.

        Args:
            source: Multi-agent source object to sync to the session.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        self.session_repository.update_multi_agent(self.session_id, source)

    def initialize_multi_agent(self, source: "MultiAgentBase", **kwargs: Any) -> None:
        """Initialize multi-agent state from the session repository.

        Args:
            source: Multi-agent source object to restore state into
            **kwargs: Additional keyword arguments for future extensibility.
        """
        state = self.session_repository.read_multi_agent(self.session_id, source.id, **kwargs)
        if state is None:
            self.session_repository.create_multi_agent(self.session_id, source, **kwargs)
        else:
            logger.debug("session_id=<%s> | restoring multi-agent state", self.session_id)
            source.deserialize_state(state)

    def initialize_bidi_agent(self, agent: "BidiAgent", **kwargs: Any) -> None:
        """Initialize a bidirectional agent with a session.

        Args:
            agent: BidiAgent to initialize from the session
            **kwargs: Additional keyword arguments for future extensibility.
        """
        if agent.agent_id in self._latest_agent_message:
            raise SessionException("The `agent_id` of an agent must be unique in a session.")
        self._latest_agent_message[agent.agent_id] = None

        session_agent = self.session_repository.read_agent(self.session_id, agent.agent_id)

        if session_agent is None:
            logger.debug(
                "agent_id=<%s> | session_id=<%s> | creating bidi agent",
                agent.agent_id,
                self.session_id,
            )

            session_agent = SessionAgent.from_bidi_agent(agent)
            self.session_repository.create_agent(self.session_id, session_agent)
            # Initialize messages with sequential indices
            session_message = None
            for i, message in enumerate(agent.messages):
                session_message = SessionMessage.from_message(message, i)
                self.session_repository.create_message(self.session_id, agent.agent_id, session_message)
            self._latest_agent_message[agent.agent_id] = session_message
        else:
            logger.debug(
                "agent_id=<%s> | session_id=<%s> | restoring bidi agent",
                agent.agent_id,
                self.session_id,
            )
            agent.state = AgentState(session_agent.state)

            session_agent.initialize_bidi_internal_state(agent)

            # BidiAgent has no conversation_manager, so no prepend_messages or removed_message_count
            session_messages = self.session_repository.list_messages(
                session_id=self.session_id,
                agent_id=agent.agent_id,
                offset=0,
            )
            if len(session_messages) > 0:
                self._latest_agent_message[agent.agent_id] = session_messages[-1]

            # Restore the agents messages array
            agent.messages = [session_message.to_message() for session_message in session_messages]

            # Fix broken session histories: https://github.com/strands-agents/sdk-python/issues/859
            agent.messages = self._fix_broken_tool_use(agent.messages)

    def append_bidi_message(self, message: Message, agent: "BidiAgent", **kwargs: Any) -> None:
        """Append a message to the bidirectional agent's session.

        Args:
            message: Message to add to the agent in the session
            agent: BidiAgent to append the message to
            **kwargs: Additional keyword arguments for future extensibility.
        """
        # Calculate the next index (0 if this is the first message, otherwise increment the previous index)
        latest_agent_message = self._latest_agent_message[agent.agent_id]
        if latest_agent_message:
            next_index = latest_agent_message.message_id + 1
        else:
            next_index = 0

        session_message = SessionMessage.from_message(message, next_index)
        self._latest_agent_message[agent.agent_id] = session_message
        self.session_repository.create_message(self.session_id, agent.agent_id, session_message)

    def sync_bidi_agent(self, agent: "BidiAgent", **kwargs: Any) -> None:
        """Serialize and update the bidirectional agent into the session repository.

        Args:
            agent: BidiAgent to sync to the session.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        self.session_repository.update_agent(
            self.session_id,
            SessionAgent.from_bidi_agent(agent),
        )
