import enum
import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt

from db import create_annotation
from utils import (
    fetch_annotations,
    generate_highlighted_text,
    get_root_logger,
    jsonl_iter,
)

logger = get_root_logger()

console = Console()

PATTERNS = {
    "ingredient": re.compile(
        r"ingredienten|ingredientes?|Ingrediënt|רכיבים|ingrediënten|ingredienti|composición|innihald|ingr[ée]dients?|bestanddelen|i̇çi̇ndeki̇ler|inhaltsstoffe|ingredienser|composição|zutaten|sastojci|sastāvs|složení|съставки|sastav|összetevők|состав|склад|ainesosat|συστατικα|sestavine|składniki|Thành phần|(?:bahan-)bahannya|composition|Құрамы|ส่วนประกอบสําคัญ|ainekset|ส่วนประกอบที่สำคัญ|成分|配料|ส่วนประกอบโดยประมาณ|الـمـكـونـات|ส่วนประกอบที่สําคัญ|原料|原材料名|لمكونات",
        re.I,
    ),
    "conservation": re.compile(
        r"conserver avant|consommer de|-18 °c|consommer avant|[àa] conserver|conserver ([àa]|dans)|best before|conditionn[ée]|frais et sec|storage instructions?|store in a (cool|dry)|keep cool|conservare (al|in)|au frais|modo de conservación|modo de conservação|conservar el producto|temperatura [óo]ptima|conservar refrigerado",
        re.I,
    ),
    "allergen": re.compile(
        r"puede contener|peut contenir|może zawierać|trazas eventuales|kan indeholde spor|contient du|contiene ?:|contains ?:|bevat ?:|може містити|peux contenir|kann? sp[uo]ren|kann andere|allerg|traces? [ée]ventuelles?|traces? possibles?|may contain traces?|pr[ée]sence [ée]ventuelle|bevat mogelijk|fabriqué dans une usine|gemaakt in een bedrijf|made in a factory",
        re.I,
    ),
    "other": re.compile(
        r"prodotto essiccazione|product subject to|produit sujet à", re.I
    ),
    "non-food": re.compile(
        r"cigarette|parfum|stearamidopropyl|benzoate sodium|butylphenyl|trideceth-12|alpha-isomethyl|hydroxypropyltrimonium|disodium edta|benzyl salicylate|copolymer|behentrimonium methosulfate|silica",
        re.I,
    ),
    "prepared-with": re.compile(r"bereid met|préparé avec|prepared with", re.I),
    "shake-before": re.compile(r"agiter avant|share before", re.I),
    "cacao-percent": re.compile(
        r"(:?cacao|cacau|cocoa|kakao) \d\d ?% (m[íi]nimo|minimum)|\d\d ?% minimum (:?cacao|cacau|cocoa|kakao)|\d\d ?% de (:?cacao|cacau|cocoa|kakao) m[íi]nimo|mindestens \d\d ?% kakao|(:?cacao|cacau|cocoa|kakao): \d\d% min",
        re.I,
    ),
    "single-word": re.compile(r"\b\w+\b", re.I),
}

UPDATED_PAYLOAD_PATH = Path("/tmp/updated_payload.txt")


def is_corrected_marked_text_valid(original: str, correction: str):
    return original.replace("<b>", "").replace("</b>", "") == correction.replace(
        "<b>", ""
    ).replace("</b>", "")


SPAN_TAG_START = "<b>"
SPAN_TAG_END = "</b>"
mark_tag_re = re.compile(r"<(?:\/)?b>")


class SearchInType(str, enum.Enum):
    ingredient = "ingredient"
    before_ingredient = "before-ingredient"
    after_ingredient = "after-ingredient"


def extract_offsets(marked_text: str) -> list[tuple[int, int]]:
    offsets = []
    start_idx = None
    tag_offset_total = 0
    for match in mark_tag_re.finditer(marked_text):
        text = match.group(0)
        is_ending_tag = text == SPAN_TAG_END
        if is_ending_tag:
            if start_idx is None:
                raise ValueError(f"invalid markup: {marked_text}")
            else:
                end_idx = match.start() - tag_offset_total
                tag_offset_total += len(SPAN_TAG_END)
                offsets.append((start_idx, end_idx))
                start_idx = None
        else:
            if start_idx is not None:
                raise ValueError(f"invalid markup: {marked_text}")
            tag_offset_total += len(SPAN_TAG_START)
            start_idx = match.end() - tag_offset_total
    return offsets


def _annotate(item: dict, action: str, updated_marked_text: Optional[str] = None):
    meta = item["meta"]
    identifier = meta["id"]
    marked_text = item["marked_text"]
    if action in ("a", "r"):
        create_annotation(identifier, action=action)
    elif action == "u":
        if is_corrected_marked_text_valid(marked_text, updated_marked_text):
            updated_offsets = extract_offsets(updated_marked_text)
            create_annotation(identifier, action, updated_offsets)
        else:
            raise ValueError("original text has been modified")


def annotate(item: dict, existing_annotation: Optional[dict] = None):
    meta = item["meta"]
    console.print(f"Image URL: {meta['url'].replace('.json', '.jpg')}")
    identifier = meta["id"]
    console.print(f"ID: {identifier}")
    if existing_annotation is not None:
        console.print(
            f"Annotation already exists: "
            f"action='{existing_annotation['action']}', "
            f"updated_offsets={existing_annotation['updated_offsets']}"
        )
    marked_text = generate_highlighted_text(
        item["text"], [list(x) for x in item["offsets"]]
    )
    marked_text_highlighted = marked_text.replace("<b>", "[red]").replace(
        "</b>", "[/red]"
    )
    console.print(f"Marked text: ```{marked_text_highlighted}```", highlight=False)
    action = Prompt.ask("Action (a/r/u/s)", choices=["a", "r", "u", "s"])
    if action == "s":
        return
    if action in ("a", "r"):
        _annotate(item, action)
        console.print("Created :white_check_mark:")
    else:
        while True:
            UPDATED_PAYLOAD_PATH.unlink(missing_ok=True)
            UPDATED_PAYLOAD_PATH.touch()
            UPDATED_PAYLOAD_PATH.write_text(marked_text)

            Confirm.ask(f"Is the text in {UPDATED_PAYLOAD_PATH} ready to be saved")
            updated_marked_text = UPDATED_PAYLOAD_PATH.read_text()
            try:
                _annotate(item, action, updated_marked_text)
            except ValueError as e:
                console.print(e)
            break

        console.print("Created :white_check_mark:")


def matches_pattern(
    item: dict, pattern: re.Pattern, search_in: SearchInType, context_length: int = 10
) -> Optional[re.Match]:
    text = item["text"]
    offsets = item["offsets"]
    for start_offset, end_offset in offsets:
        if search_in == SearchInType.ingredient:
            span = text[start_offset:end_offset]
        elif search_in == SearchInType.after_ingredient:
            span = text[end_offset : end_offset + context_length]
        else:
            span = text[start_offset - context_length : start_offset]
        if pattern == PATTERNS["single-word"]:
            match = pattern.fullmatch(span)
        else:
            match = pattern.search(span)
        if match is not None:
            return match
    return None


def has_single_word_ingredient(item: dict) -> bool:
    text = item["text"]
    offsets = item["offsets"]
    for start_offset, end_offset in offsets:
        words = text[start_offset:end_offset].split(" ")
        if len(words) == 1:
            return True
    return False


def has_single_ingredient_without_prefix(item: dict) -> bool:
    text = item["text"]
    offsets = item["offsets"]
    for start_offset, end_offset in offsets:
        substring = text[start_offset:end_offset]

        if "," in substring:
            continue
        before = text[max(0, start_offset - 30) : start_offset].rstrip(": \n-;.")
        match = PATTERNS["ingredient"].search(before)
        if not match:
            return True
    return False


def additional_words_after_ingredient_prefix(
    item: dict, context_length: int = 25
) -> bool:
    text = item["text"]
    offsets = item["offsets"]
    for start_offset, _ in offsets:
        match = PATTERNS["ingredient"].search(
            text, start_offset - context_length, start_offset
        )
        if match:
            before = text[match.end() : start_offset]
            if before.strip(" :\n;.*?,-•="):
                print(f"{before=}")
                return True
    return False


def perform_detection(detection_type: str, item: dict) -> bool:
    if detection_type == "single-ingredient":
        return has_single_word_ingredient(item)
    elif detection_type == "single-ingredient-without-prefix":
        return has_single_ingredient_without_prefix(item)
    elif detection_type == "missing-ingredient":
        return additional_words_after_ingredient_prefix(item)
    raise ValueError("unknown detection type")


def run(
    dataset_path: Path = typer.Argument(
        Path("dataset_base.jsonl.gz"),
        help="path of the JSONL dataset",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    pattern: Optional[str] = typer.Option(None),
    pattern_type: Optional[str] = typer.Option(None),
    search_in: SearchInType = typer.Option(SearchInType.ingredient),
    search_context_length: int = typer.Option(10),
    target_identifier: Optional[str] = typer.Option(None),
    count_items: bool = typer.Option(False),
    detection_type: Optional[str] = typer.Option(None),
    skip_if_annotated: bool = True,
):
    existing_annotations = fetch_annotations()
    if pattern_type:
        pattern = PATTERNS[pattern_type]
    else:
        pattern = re.compile(pattern, re.I) if pattern else None

    logger.info("pattern: %s", pattern)
    logger.info("target identifier: %s", target_identifier)
    counts = 0

    for item in jsonl_iter(dataset_path):
        id_ = item["meta"]["id"]
        annotation_exists = id_ in existing_annotations
        existing_annotation = None
        if annotation_exists:
            if skip_if_annotated:
                logger.debug("Skipping item %s (already manually annotated)", id_)
                continue

            existing_annotation = existing_annotations[id_]
            updated_offsets = existing_annotation["updated_offsets"]
            if updated_offsets is not None:
                item["offsets"] = updated_offsets

        if target_identifier and id_ == target_identifier:
            annotate(item, existing_annotation)
            return
        else:
            if pattern and (
                (
                    match := matches_pattern(
                        item, pattern, search_in, search_context_length
                    )
                )
                is not None
            ):
                if count_items:
                    counts += 1
                    continue
                console.print(f"item: {id_}, pattern matched: {match}")
                annotate(item, existing_annotation)
            elif detection_type is not None:
                if perform_detection(detection_type, item):
                    if count_items:
                        counts += 1
                        continue
                    console.print(f"item: {id_} detected ({detection_type})")
                    annotate(item, existing_annotation)

    console.print(f"{counts=}")


if __name__ == "__main__":
    typer.run(run)
