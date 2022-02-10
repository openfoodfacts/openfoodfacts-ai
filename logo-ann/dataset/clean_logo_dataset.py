import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=Path)
parser.add_argument("--min-count", type=int, default=5)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
input_dir = args.input_dir
min_count = args.min_count
dry_run = args.dry_run

TO_MOVE = {
    "store_lidl": "brand_Lidl",
    "store_netto": "brand_Netto",
    "store_spar": "brand_Spar",
    "store_delhaize": "brand_Delhaize",
    "store_coop": "brand_Coop",
    "store_carrefour": "brand_Carrefour",
    "store_edeka": "brand_Edeka",
    "store_bonpreu": "brand_Bonpreu",
    "label_fr_label-rouge": "label_en_label-rouge",
    "packaging_1-pet": "packaging_01-pet",
    "packaging_5-pet": "packaging_05-pet",
    "packaging_5-pp": "packaging_05-pp",
    "packaging_polyethylene-terephthalate": "packaging_01-pet",
    "label_en_fair-trade": "label_en_fairtrade-international",
    "label_en_2013-gold-medal-of-the-german-agricultural-society": "label_en_gold-medal-of-the-german-agricultural-society",
    "label_en_2014-gold-medal-of-the-german-agricultural-society": "label_en_gold-medal-of-the-german-agricultural-society",
    "label_en_2015-gold-medal-of-the-german-agricultural-society": "label_en_gold-medal-of-the-german-agricultural-society",
    "label_en_2016-gold-medal-of-the-german-agricultural-society": "label_en_gold-medal-of-the-german-agricultural-society",
    "label_en_2017-gold-medal-of-the-german-agricultural-society": "label_en_gold-medal-of-the-german-agricultural-society",
    "label_en_2018-gold-medal-of-the-german-agricultural-society": "label_en_gold-medal-of-the-german-agricultural-society",
    "label_en_2019-gold-medal-of-the-german-agricultural-society": "label_en_gold-medal-of-the-german-agricultural-society",
    "label_fr_medaille-d-or-du-concours-general-agricole-2017": "label_fr_medaille-d-or-du-concours-general-agricole",
    "label_fr_medaille-d-argent-du-concours-general-agricole-2019": "label_fr_medaille-d-argent-du-concours-general-agricole",
}



TO_REMOVE = {
    "packaging_fr_barquette-et-film-plastique-a-jeter",
    "packaging_fr_pensez-au-tri-!",
    "packaging_ja_",
}


def move_dir_content(input_dir: Path, output_dir: Path, dry_run: bool):
    for file_path in input_dir.iterdir():
        if file_path.is_file():
            new_path = output_dir / file_path.name
            print(f"Moving {file_path} to {new_path}")

            if not dry_run:
                if not output_dir.is_dir():
                    output_dir.mkdir(exist_ok=True)
                file_path.rename(new_path)

    if not dry_run:
        input_dir.rmdir()


assert input_dir.is_dir()

print("--- Moving store -> brand")

for source_dir_str, dest_dir_str in TO_MOVE.items():
    source_dir = input_dir / source_dir_str
    dest_dir = input_dir / dest_dir_str
    if source_dir.is_dir():
        move_dir_content(source_dir, dest_dir, dry_run)

    else:
        print(f"Cannot move {source_dir}/* to {dest_dir}/")


for dir_str in TO_REMOVE:
    dir_path = input_dir / dir_str

    if dir_path.is_dir():
        print(f"Removing {dir_str}")

        if not dry_run:
            shutil.rmtree(dir_path)


for child in input_dir.iterdir():
    if not child.is_dir():
        continue

    children = [x for x in child.iterdir() if x.is_file()]
    if len(children) < min_count:
        print(f"Deleting folder {child.name} ({len(children)})")
        if not dry_run:
            for x in children:
                x.unlink()
            child.rmdir()

if dry_run:
    print("--- DRY RUN --")
