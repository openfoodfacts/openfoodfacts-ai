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
    updated_offsets = JSONField(null=True)

    class Meta:
        database = db


def create_annotation(
    identifier: str,
    action: Literal["a", "r", "u"],
    updated_offsets: Optional[list] = None,
):
    if updated_offsets is not None:
        action = "u"
    splits = identifier.split("_")
    annotation = Annotation.get_or_none(identifier=identifier)
    if annotation is None:
        return Annotation.create(
            identifier=identifier,
            barcode=splits[0],
            image_id=splits[1],
            action=action,
            updated_offsets=updated_offsets,
        )
    else:
        annotation.action = action
        annotation.updated_offsets = updated_offsets
        annotation.save()
        return annotation


def create_tables():
    db.create_tables([Annotation], safe=True)
