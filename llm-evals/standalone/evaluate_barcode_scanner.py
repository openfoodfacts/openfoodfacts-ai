# /// script
# dependencies = [
#     "deepdiff>=8.6.1",
#     "orjson>=3.11.4",
#     "pydantic-ai>=1.104.0",
#     "pydantic-evals>=1.104.0",
#     "tapirscan==1.5.0",
#     "zxing-cpp==3.1.1",
#     "opencv-python-headless==5.0.0.93",
#     "diskcache",
#     "typer",
# ]
# ///


import typing
from dataclasses import dataclass
from typing import Literal
from urllib.request import urlopen

import cv2
import diskcache
import numpy as np
import tapirscan
import typer
import zxingcpp
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

TASK_NAME = "barcode_scan"
cache = diskcache.Cache(directory="./image_cache")


class DetectedBarcode(BaseModel):
    barcode: str
    format: str


class ExpectedOutput(BaseModel):
    barcodes: list[DetectedBarcode]


class Inputs(BaseModel):
    image_url: str


class Output(BaseModel):
    barcodes: list[DetectedBarcode]


class MetaData(BaseModel):
    difficulty: str | None = None


def sort_key_fn(barcode: DetectedBarcode) -> tuple[str, str]:
    return (barcode.barcode, barcode.format)


@dataclass
class CustomEvaluator(Evaluator[Inputs, ExpectedOutput, MetaData]):
    def evaluate(
        self, ctx: EvaluatorContext[Inputs, ExpectedOutput, MetaData]
    ) -> bool | EvaluationReason:
        if not ctx.expected_output:
            raise RuntimeError("expected_output should not be null")

        output = typing.cast(Output, ctx.output)
        return sorted(output.barcodes, key=sort_key_fn) == sorted(
            ctx.expected_output.barcodes, key=sort_key_fn
        )


dataset = Dataset[Inputs, ExpectedOutput, MetaData](
    name=TASK_NAME,
    cases=[
        Case(
            name="i1ymqcK7YN",
            inputs=Inputs(
                image_url="https://prices.openfoodfacts.org/img/0325/i1ymqcK7YN.jpg"
            ),
            expected_output=ExpectedOutput(
                barcodes=[DetectedBarcode(barcode="3560070456352", format="EAN13")]
            ),
        ),
        Case(
            name="fx61TNlZ28",
            inputs=Inputs(
                image_url="https://prices.openfoodfacts.org/img/0325/fx61TNlZ28.jpg"
            ),
            expected_output=ExpectedOutput(
                barcodes=[DetectedBarcode(barcode="7350126086338", format="EAN13")]
            ),
        ),
        Case(
            name="uC8obEO2S0",
            inputs=Inputs(
                image_url="https://prices.openfoodfacts.org/img/0325/uC8obEO2S0.jpg"
            ),
            expected_output=ExpectedOutput(
                barcodes=[
                    DetectedBarcode(barcode="2050401965314", format="EAN13"),
                    DetectedBarcode(barcode="5900027000043", format="EAN13"),
                ]
            ),
        ),
        Case(
            name="kyXzHHYzPm",
            inputs=Inputs(
                image_url="https://prices.openfoodfacts.org/img/0325/kyXzHHYzPm.jpg"
            ),
            expected_output=ExpectedOutput(
                barcodes=[
                    DetectedBarcode(barcode="051500255377", format="UPCA"),
                ]
            ),
        ),
        Case(
            name="000255586",
            inputs=Inputs(
                image_url="https://prices.openfoodfacts.org/img/price-tags/000/255/000255586.webp"
            ),
            expected_output=ExpectedOutput(
                barcodes=[
                    DetectedBarcode(barcode="1210002001724", format="EAN13"),
                ]
            ),
        ),
        Case(
            name="000255580",
            inputs=Inputs(
                image_url="https://prices.openfoodfacts.org/img/price-tags/000/255/000255580.webp"
            ),
            expected_output=ExpectedOutput(
                barcodes=[
                    DetectedBarcode(barcode="1210002001755", format="EAN13"),
                ]
            ),
        ),
        Case(
            name="000255545",
            inputs=Inputs(
                image_url="https://prices.openfoodfacts.org/img/price-tags/000/255/000255545.webp"
            ),
            expected_output=ExpectedOutput(
                barcodes=[
                    DetectedBarcode(barcode="5411788045179", format="EAN13"),
                ]
            ),
            metadata=MetaData(difficulty="easy"),
        ),
        Case(
            name="000255546",
            inputs=Inputs(
                image_url="https://prices.openfoodfacts.org/img/price-tags/000/255/000255546.webp"
            ),
            expected_output=ExpectedOutput(
                barcodes=[
                    DetectedBarcode(barcode="3396411253423", format="EAN13"),
                ]
            ),
            metadata=MetaData(difficulty="easy"),
        ),
        Case(
            name="000255547",
            inputs=Inputs(
                image_url="https://prices.openfoodfacts.org/img/price-tags/000/255/000255547.webp"
            ),
            expected_output=ExpectedOutput(
                barcodes=[
                    DetectedBarcode(barcode="3396411239496", format="EAN13"),
                ]
            ),
        ),
        Case(
            name="000252515",
            inputs=Inputs(
                image_url="https://prices.openfoodfacts.org/img/price-tags/000/252/000252515.webp"
            ),
            expected_output=ExpectedOutput(
                barcodes=[
                    DetectedBarcode(barcode="5061046051307", format="EAN13"),
                ]
            ),
        ),
        Case(
            name="000252513",
            inputs=Inputs(
                image_url="https://prices.openfoodfacts.org/img/price-tags/000/252/000252513.webp"
            ),
            expected_output=ExpectedOutput(
                barcodes=[
                    DetectedBarcode(barcode="3051035111039", format="EAN13"),
                ]
            ),
        ),
        # truncated
        Case(
            name="000194908",
            inputs=Inputs(
                image_url="https://prices.openfoodfacts.org/img/price-tags/000/194/000194908.webp"
            ),
            expected_output=ExpectedOutput(
                barcodes=[
                    DetectedBarcode(barcode="3417960022695", format="EAN13"),
                ]
            ),
        ),
        Case(
            name="000183563",
            inputs=Inputs(
                image_url="https://prices.openfoodfacts.org/img/price-tags/000/183/000183563.webp"
            ),
            expected_output=ExpectedOutput(
                barcodes=[
                    DetectedBarcode(barcode="3330724270081", format="EAN13"),
                ]
            ),
        ),
        Case(
            name="000183564",
            inputs=Inputs(
                image_url="https://prices.openfoodfacts.org/img/price-tags/000/183/000183564.webp"
            ),
            expected_output=ExpectedOutput(
                barcodes=[
                    DetectedBarcode(barcode="3452200002518", format="EAN13"),
                ]
            ),
        ),
        Case(
            name="000183565",
            inputs=Inputs(
                image_url="https://prices.openfoodfacts.org/img/price-tags/000/183/000183565.webp"
            ),
            expected_output=ExpectedOutput(
                barcodes=[
                    DetectedBarcode(barcode="3299431313019", format="EAN13"),
                ]
            ),
        ),
        # Case(
        #     name="",
        #     inputs=Inputs(
        #         image_url=""
        #     ),
        #     expected_output=ExpectedOutput(
        #         barcodes=[
        #             DetectedBarcode(barcode="", format="EAN13"),
        #         ]
        #     ),
        # ),
    ],
    evaluators=[
        CustomEvaluator(),
    ],
)


def normalize_barcode_format(format: zxingcpp.BarcodeFormat) -> str:
    if format == zxingcpp.BarcodeFormat.EAN13:
        return "EAN13"
    elif format == zxingcpp.BarcodeFormat.UPCA:
        return "UPCA"
    raise ValueError()


def evaluate_zxing(image: np.ndarray) -> Output:
    raw_result = zxingcpp.read_barcodes(
        image, formats=[zxingcpp.BarcodeFormat.EAN13, zxingcpp.BarcodeFormat.UPCA]
    )
    results = [
        DetectedBarcode(
            barcode=result.text, format=normalize_barcode_format(result.format)
        )
        for result in raw_result
    ]
    return Output(barcodes=results)


def evaluate_tapirscan(image: np.ndarray, mode: str) -> Output:
    raw_result = tapirscan.scan(image, formats="retail", mode=mode)
    results = [
        DetectedBarcode(barcode=result.text, format=result.format)
        for result in raw_result.barcodes
    ]
    return Output(barcodes=results)


def evaluate(
    model: Literal["tapirscan", "zxing"] = "tapirscan",
    mode: str = "medium",
    include_output: bool = True,
    include_reasons: bool = True,
):
    async def run_task(inputs: Inputs) -> Output:
        image_url = inputs.image_url
        if image_url in cache:
            image = cache[image_url]
        else:
            image_bytes = urlopen(image_url).read()
            image_np = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR_RGB)
            if image is None:
                raise ValueError("Could not load image")
            cache[image_url] = image

        if model == "tapirscan":
            return evaluate_tapirscan(image, mode=mode)
        elif model == "zxing":
            return evaluate_zxing(image)

    # Run the evaluation
    report = dataset.evaluate_sync(run_task)
    # Print the results
    report.print(
        include_output=include_output,
        include_reasons=include_reasons,
        include_expected_output=True,
    )


if __name__ == "__main__":
    typer.run(evaluate)
