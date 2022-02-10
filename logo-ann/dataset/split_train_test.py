import argparse
from collections import defaultdict
import math
from pathlib import Path
import random


parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=Path)
parser.add_argument(
    "--train-split",
    type=float,
    default=0.8,
    help="Fraction of category and samples to use as train",
)
parser.add_argument(
    "--val-split",
    type=float,
    default=0.1,
    help="Fraction of category and samples to use as val",
)
parser.add_argument(
    "--large-category-threshold",
    type=int,
    default=50,
    help="Threshold above which we consider the category to be large enough "
    "to be splitted between train, val and test sets",
)
args = parser.parse_args()
input_dir = args.input_dir
train_split = args.train_split
val_split = args.val_split
large_category_threshold = args.large_category_threshold

assert large_category_threshold >= 10
assert train_split + val_split < 1.0
assert input_dir.is_dir()

grouped_by_category = defaultdict(list)

count = 0
for category_dir in (x for x in input_dir.iterdir() if x.is_dir()):
    assert all(x.name.endswith(".png") for x in category_dir.iterdir()), category_dir
    count += len([x.name.endswith(".png") for x in category_dir.iterdir()])
    grouped_by_category[category_dir.name] = [
        f"{x.parent.name}/{x.name}" for x in category_dir.glob("*.png")
    ]
print(f"{count} logos")

train_paths = []
val_paths = []
test_paths = []
remaining_categories = []
for key, values in sorted(
    grouped_by_category.items(), key=lambda x: len(x[1]), reverse=True
):
    if len(values) >= large_category_threshold:
        print(f"Splitting {key}")
        random.shuffle(values)
        train_count = math.floor(len(values) * train_split)
        val_count = math.floor(len(values) * val_split)
        train_paths.extend(values[:train_count])
        print(
            f"total: {len(values)}, train: {train_count}, "
            f"val: {val_count}, test: {len(values) - train_count - val_count}"
        )
        val_paths.extend(values[train_count : train_count + val_count])
        test_paths.extend(values[train_count + val_count :])
    else:
        remaining_categories.append(key)

random.shuffle(remaining_categories)
offset = 0
for split_name, split_paths, split_fraction in (
    ("train", train_paths, train_split),
    ("val", val_paths, val_split),
    ("test", test_paths, None),
):
    added = 0
    print(f"--- Split {split_name} ---")
    if split_name == "test":
        selected_categories = remaining_categories[offset:]
    else:
        count = math.floor(len(remaining_categories) * split_fraction)
        selected_categories = remaining_categories[offset : offset + count]
        offset += len(selected_categories)

    print(f"Adding {count} categories")

    for category in selected_categories:
        split_paths += grouped_by_category[category]
        added += len(grouped_by_category[category])

    print(f"Added: {added}")

    with open(f"{split_name}.txt", "w") as f:
        f.write("\n".join(split_paths))
