from pathlib import Path
import re
from typing import Optional

import typer
from db import create_annotation
from utils import (
    fetch_annotations,
    generate_highlighted_text,
    get_root_logger,
    jsonl_iter,
)
from rich.prompt import Prompt, Confirm
from rich.console import Console

logger = get_root_logger()

console = Console()

PATTERNS = {
    "ingredient": re.compile(
        r"ingredienten|ingredientes?|ingredienti|ingr[ée]dients?|zutaten|sastojci|съставки|sastav|összetevők|ainesosat|składniki|(?:bahan-)bahannya|配料|لمكونات",
        re.I,
    ),
    "conservation": re.compile(
        r"conserver avant|consommer de|[àa] conserver|conserver ([àa]|dans)|best before|est conditionn[ée]|frais et sec|storage instructions?|store in a (cool|dry)|keep cool|conservare al",
        re.I,
    ),
    "allergen": re.compile(
        r"puede contener|peux contenir|kan sporen|allerg|traces? [ée]ventuelles?", re.I
    ),
    "other": re.compile(
        r"prodotto essiccazione|product subject to|produit sujet à", re.I
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


def annotate(item: dict):
    meta = item["meta"]
    console.print(f"Image URL: {meta['url'].replace('.json', '.jpg')}")
    identifier = meta["id"]
    console.print(f"ID: {identifier}")
    marked_text = (generate_highlighted_text(item["text"], [list(x) for x in item["offsets"]]))
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
                console.print(f"{e.message}")
            break

        console.print("Created :white_check_mark:")


def matches_pattern(item: dict, pattern: re.Pattern) -> Optional[re.Match]:
    text = item["text"]
    offsets = item["offsets"]
    for start_offset, end_offset in offsets:
        span = text[start_offset:end_offset]
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


def additional_words_after_ingredient_prefix(item: dict) -> bool:
    text = item["text"]
    offsets = item["offsets"]
    for start_offset, _ in offsets:
        match = PATTERNS["ingredient"].search(text, start_offset - 15, start_offset)
        if match:
            before = text[match.end() : start_offset]
            if before.strip(" :\n;.*?,-•="):
                print(f"{before=}")
                return True
    return False


def perform_detection(detection_type: str, item: dict) -> bool:
    if detection_type == "single-ingredient":
        return has_single_word_ingredient(item)
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
        if annotation_exists:
            if skip_if_annotated:
                logger.debug("Skipping item %s (already manually annotated)", id_)
                continue
            updated_offsets = existing_annotations[id_]["updated_offsets"]
            if updated_offsets is not None:
                item["offsets"] = updated_offsets

        if target_identifier and id_ == target_identifier:
            annotate(item)
            return
        else:
            if pattern and ((match := matches_pattern(item, pattern)) is not None):
                if count_items:
                    counts += 1
                    continue
                console.print(f"item: {id_}, pattern matched: {match}")
                annotate(item)
            elif detection_type is not None:
                if perform_detection(detection_type, item):
                    if count_items:
                        counts += 1
                        continue
                    console.print(f"item: {id_} detected ({detection_type})")
                    annotate(item)

    console.print(f"{counts=}")


if __name__ == "__main__":
    typer.run(run)
