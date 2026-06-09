import json

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.models.google import GoogleModelSettings
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from llm_evals.types import OutputMode

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
    def set(
        cls,
        model: str,
        instructions: str,
        output_type: type[BaseModel],
        task_name: str,
        output_mode: OutputMode = "tool",
        thinking_config: str | None = None,
    ) -> None:
        """Set the evaluation agent with the specified model, instructions,
        and output type.

        Args:
            model (str): The model to be used by the evaluation agent.
            instructions (str): The instructions for the evaluation agent.
            output_type (type[BaseModel]): The expected output type of the
                agent.
            task_name (str): Name of the task associated with the evaluation.
            output_mode (OutputMode): The output mode of the agent, which can
                be "tool", "native", or "prompted". Defaults to "tool".
            thinking_config (str | None): Optional configuration for the
                agent's thinking process.
        Raises:
            ValueError: If an evaluation agent has already been created.
        """
        if cls._evaluation_agent is not None:
            raise ValueError("Agent has already been created.")

        cls._evaluation_agent = cls(
            model=model,
            instructions=instructions,
            output_type=output_type,
            task_name=task_name,
            output_mode=output_mode,
            thinking_config=thinking_config,
        )

    def __init__(
        self,
        model: str,
        instructions: str,
        output_type: type[BaseModel],
        task_name: str,
        output_mode: OutputMode = "tool",
        thinking_config: str | None = None,
    ) -> None:
        if self._evaluation_agent is not None:
            raise ValueError(EVALUATION_AGENT_ALREADY_CREATED_ERROR)

        output_mode_cls: ToolOutput | NativeOutput | PromptedOutput | None = None
        if output_mode not in ("tool", "native", "prompted", "native+prompted"):
            raise ValueError(
                f"Invalid output_mode: {output_mode}. Must be one of "
                "'tool', 'native', or 'prompted'."
            )
        elif output_mode == "tool":
            output_mode_cls = ToolOutput
        elif output_mode in ("native", "native+prompt"):
            output_mode_cls = NativeOutput
        else:
            output_mode_cls = PromptedOutput

        self._agent = Agent(
            model=model,
            output_type=output_mode_cls(output_type),
            model_settings=self.get_model_settings(thinking_config, model),
        )
        self._instructions = instructions
        self._model = model
        self._output_type = output_type
        self._output_mode = output_mode
        self._task_name = task_name
        self._thinking_config = thinking_config

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
        return self._agent

    @property
    def model(self) -> str:
        """Get the model used by the evaluation agent."""
        return self._model

    @property
    def output_type(self) -> type[BaseModel]:
        """Get the expected output type of the evaluation agent."""
        return self._output_type

    @property
    def instructions(self) -> str:
        """Get the instructions for the evaluation agent."""

        if self._output_mode == "native+prompted":
            return (
                self._instructions
                + f"\n\nResponse must be formatted as JSON, and follow this JSON schema:\n{json.dumps(self._output_type.model_json_schema())}"
            )

        return self._instructions

    @property
    def output_mode(self) -> OutputMode:
        """Get the output mode of the evaluation agent."""
        return self._output_mode

    @property
    def task_name(self) -> str:
        """Get the name of the task we're evaluating against."""
        return self._task_name

    @property
    def thinking_config(self):
        return self._thinking_config

    @classmethod
    def get_model_settings(
        cls, thinking_config: str | None, model: str
    ) -> GoogleModelSettings | None:
        if thinking_config is None:
            return None

        if (
            model.startswith("gemini")
            or model.startswith("google-vertex:gemini")
            or model.startswith("google-gla:gemini")
        ):
            if thinking_config.isdigit():
                return GoogleModelSettings(
                    google_thinking_config={"thinking_budget": int(thinking_config)}
                )

            if thinking_config not in (
                "MINIMAL",
                "LOW",
                "MEDIUM",
                "HIGH",
                "THINKING_LEVEL_UNSPECIFIED",
            ):
                raise ValueError(
                    f"Invalid thinking_config: {thinking_config}. Must be one of "
                    "'LOW', 'HIGH', or 'THINKING_LEVEL_UNSPECIFIED'."
                )
            return GoogleModelSettings(
                google_thinking_config={"thinking_level": thinking_config}
            )

        if model.startswith("openrouter:"):
            if thinking_config not in ("low", "medium", "high", "0"):
                raise ValueError(
                    f"Invalid thinking_config: {thinking_config}. Must be one of "
                    "'low', 'medium', or 'high'."
                )
            if thinking_config == "0":
                return OpenRouterModelSettings(openrouter_reasoning={"enabled": False})
            return OpenRouterModelSettings(
                openrouter_reasoning={"effort": thinking_config}
            )
        return None
