import typer
from pydantic_ai import Agent, ImageUrl, InlineDefsJsonSchemaTransformer
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from llm_evals.types import TaskType

DEFAULT_MODEL = "google-vertex:gemini-2.5-flash-lite"

app = typer.Typer()


@app.command()
def run_extraction(
    image_url: str,
    model: str = DEFAULT_MODEL,
    openai_url: str | None = None,
):
    from llm_evals.tasks.food.product_info_extraction.schemas import (
        ProductInfoExtractionResponseModel,
    )

    if openai_url:
        # client = openai.Client(base_url=openai_url)
        # model = client.models.list().data[0].id
        # image_bytes = requests.get(image_url).content
        # response = client.chat.completions.create(
        #     model=model,
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 """Extract all relevant information from this product packaging photo.""",
        #                 base64.b64encode(image_bytes).decode("utf-8"),
        #             ],
        #         }
        #     ],
        #     response_format={
        #         "type": "json_schema",
        #         "json_schema": {
        #             "name": "ProductInfoExtractionResponseModel",
        #             "schema": ProductInfoExtractionResponseModel.model_json_schema(),
        #         },
        #     },
        # )
        # print(response)
        provider = OpenAIProvider(base_url=openai_url)
        profile = OpenAIModelProfile(
            json_schema_transformer=InlineDefsJsonSchemaTransformer,  # Supported by any model class on a plain ModelProfile
            openai_supports_strict_tool_definition=False,  # Supported by OpenAIModel only, requires OpenAIModelProfile
            openai_supports_tool_choice_required=True,
        )
        model = OpenAIChatModel(
            model,
            provider=provider,
            profile=profile,
        )
        agent = Agent(model, output_type=ProductInfoExtractionResponseModel)
    else:
        agent = Agent(
            model,
            output_type=ProductInfoExtractionResponseModel,
        )

        result = agent.run_sync(
            [
                """Extract all relevant information from this product packaging photo.""",
                ImageUrl(
                    image_url,
                ),
            ]
        )
        print(result.output.model_dump_json(indent=2))


@app.command()
def run_multi_image_extraction():
    pass


@app.command()
def evaluate(
    task: TaskType,
    model: str = DEFAULT_MODEL,
    include_output: bool = True,
    include_expected_output: bool = True,
    include_reasons: bool = True,
):
    from llm_evals.evaluate import evaluate_task

    evaluate_task(
        model=model,
        task=task,
        include_output=include_output,
        include_expected_output=include_expected_output,
        include_reasons=include_reasons,
    )


if __name__ == "__main__":
    app()
