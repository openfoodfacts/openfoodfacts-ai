import click
from pprint import pprint
from utils import load_dataset
from ingredients import format_ingredients


@click.command()
@click.option('--path', help='Path to dataset.')
@click.option('--item-id', help='Item id to review.')
def review(path, item_id):
    items = load_dataset(path)
    filtered_items = [item for item in items if item['_id'] == item_id]
    if not filtered_items:
        raise ValueError(f'Item {item_id} not find.')
    elif len(filtered_items) > 1:
        raise ValueError('More than one item found with id {item_id}')

    item = filtered_items[0]

    print('\n' + f'Id : {item["_id"]}')
    print('\n' + f'Original   : {item["original"]}')
    print('\n' + f'Correct    : {item["correct"]}')
    if 'prediction' in item:
        print('\n' + f'Prediction : {item["prediction"]}')
    if 'tags' in item:
        print('\n' + f'Tags     : {item["tags"]}')

    original_ingredients = format_ingredients(item['original'])
    correct_ingredients = format_ingredients(item['correct'])
    predicted_ingredients = format_ingredients(item['prediction'])

    print('\n' + f'Original ingredients :')
    print(sorted(original_ingredients))
    print('\n' + f'Correct ingredients : ')
    print(sorted(correct_ingredients))
    print('\n' + f'Predicted ingredients : ')
    print(sorted(predicted_ingredients))

    print('\n')

    print('\n' + f'Not original, correct')
    print(sorted(correct_ingredients-original_ingredients))
    print('\n' + f'Not original, predicted')
    print(sorted(predicted_ingredients-original_ingredients))
    print('\n' + f'Not original, correct, predicted')
    print(sorted((correct_ingredients & predicted_ingredients) - original_ingredients))
    print('\n' + f'Original, correct, not predicted')
    print(sorted((original_ingredients & correct_ingredients) - predicted_ingredients))


if __name__ == '__main__':
    review()
