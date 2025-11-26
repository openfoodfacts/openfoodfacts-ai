from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model


def get_model_name(agent: Agent[None, Any]):
    overriden_model = agent._override_model.get()

    if overriden_model is not None:
        model = overriden_model.value
    else:
        model = agent.model

    if isinstance(model, Model):
        if (
            hasattr(model, "_provider")
            and model._provider
            and hasattr(model, "_model_name")
            and model._model_name
        ):
            model_name = f"{model._provider.name}:{model._model_name}"
        elif hasattr(model, "_model_name") and model._model_name:
            model_name = model._model_name
        else:
            model_name = "unknown"

    else:
        raise ValueError("Unsupported model type")

    return model_name
