from typing import Optional, Literal
from peewee import SqliteDatabase, CharField, Model
from playhouse.sqlite_ext import JSONField

db = SqliteDatabase("annotations.db")


class Annotation(Model):
    identifier = CharField(primary_key=True)
    barcode = CharField()
    image_id = CharField()
    action = CharField(
        help_text="eiher `a`, `r` or `u`",
        null=True,
        max_length=1,
        choices=(("a", "accept"), ("r", "reject"), ("u", "update")),
    )
    updated_json = JSONField(null=True)

    class Meta:
        database = db


def create_annotation(
    identifier: str, action: Literal["a", "r", "u"], updated_json: Optional[list] = None
):
    splits = identifier.split("_")
    return Annotation.create(
        identifier=identifier,
        barcode=splits[0],
        image_id=splits[1],
        action=action,
        updated_json=updated_json,
    )


def create_tables():
    db.create_tables([Annotation], safe=True)
