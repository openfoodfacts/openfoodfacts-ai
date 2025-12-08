from pydantic import BaseModel
from pydantic_ai import Agent

EVALUATION_AGENT_ALREADY_CREATED_ERROR = """An EvaluationAgent has already been created. Use EvaluationAgent.get() to
retrieve the existing instance, or EvaluationAgent.reset() to clear the current instance."""

EVALUATION_AGENT_NOT_CREATED_ERROR = """No EvaluationAgent has been created yet. Use EvaluationAgent.set() to create
an instance before attempting to retrieve it."""


class EvaluationAgent:
    """Singleton class to manage a single evaluation agent instance.

    This class ensures that only one instance of an evaluation agent exists
    throughout the application lifecycle. It provides methods to set, get, and
    reset the evaluation agent.

    To configure the evaluation agent, use the `set` method with the desired
    model, instructions, and output type. Once set, the agent can be retrieved
    using the `get` method. The `reset` method allows for clearing the current
    agent instance, enabling the creation of a new one.
    """

    _evaluation_agent = None

    @classmethod
    def set(cls, model: str, instructions: str, output_type: type[BaseModel]) -> None:
        """Set the evaluation agent with the specified model, instructions,
        and output type.

        Args:
            model (str): The model to be used by the evaluation agent.
            instructions (str): The instructions for the evaluation agent.
            output_type (type[BaseModel]): The expected output type of the
                agent.
        Raises:
            ValueError: If an evaluation agent has already been created.
        """
        if cls._evaluation_agent is not None:
            raise ValueError("Agent has already been created.")

        cls._evaluation_agent = cls(
            model=model, instructions=instructions, output_type=output_type
        )

    def __init__(
        self, model: str, instructions: str, output_type: type[BaseModel]
    ) -> None:
        if self._evaluation_agent is not None:
            raise ValueError(EVALUATION_AGENT_ALREADY_CREATED_ERROR)

        self._agent = Agent(model=model, output_type=output_type)
        self._instructions = instructions
        self._model = model
        self._output_type = output_type

    @classmethod
    def reset(cls) -> None:
        """Reset the evaluation agent, allowing for the creation of a new one."""
        cls._evaluation_agent = None

    @classmethod
    def get(cls) -> "EvaluationAgent":
        """Get the current evaluation agent instance.

        Returns:
            EvaluationAgent: The current evaluation agent instance.
        Raises:
            ValueError: If no evaluation agent has been created yet.
        """
        if cls._evaluation_agent is None:
            raise ValueError(EVALUATION_AGENT_NOT_CREATED_ERROR)

        return cls._evaluation_agent

    @property
    def agent(self) -> "Agent":
        """Get the underlying Agent instance."""
        if self._evaluation_agent is None:
            raise ValueError(EVALUATION_AGENT_NOT_CREATED_ERROR)
        return self._agent

    @property
    def model(self) -> str:
        """Get the model used by the evaluation agent."""
        if self._evaluation_agent is None:
            raise ValueError(EVALUATION_AGENT_NOT_CREATED_ERROR)
        return self._model

    @property
    def output_type(self) -> type:
        """Get the expected output type of the evaluation agent."""
        if self._evaluation_agent is None:
            raise ValueError(EVALUATION_AGENT_NOT_CREATED_ERROR)
        return self._output_type

    @property
    def instructions(self) -> str:
        """Get the instructions for the evaluation agent."""
        if self._evaluation_agent is None:
            raise ValueError(EVALUATION_AGENT_NOT_CREATED_ERROR)
        return self._instructions
