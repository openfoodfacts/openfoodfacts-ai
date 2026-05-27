from typing import TypedDict

from pydantic import BaseModel, Field


class PersonalInfo(BaseModel):
    type: str = Field(
        description="the type of personal information detected.",
        enum=["name", "purchase_hour", "fidelity_card_id"],
    )
    value: str = Field(
        description="the value of the personal information detected. Examples: 'John Doe', '14:30', '1234567890'"
    )


class PersonalInfoList(BaseModel):
    items: list[PersonalInfo]


class ExpectedResult(BaseModel):
    items: list[PersonalInfo]


class Input(TypedDict):
    image_urls: list[str]


class MetaData(TypedDict):
    proof_id: int
    tags: list[str] | None
