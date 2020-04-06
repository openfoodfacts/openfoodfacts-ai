from pathlib import Path


def format_txt(txt):
    """
    Remove all \n, \t, \r, duplicate whitespaces, ... from a text.
    """
    return ' '.join(txt.split())


def load_dataset(path):
    path = Path(path)

    original_path = path / 'original.txt'
    with original_path.open('r', encoding='utf-8') as original_file:
        original_lines = original_file.readlines()

    correct_path = path / 'correct.txt'
    with correct_path.open('r', encoding='utf-8') as correct_file:
        correct_lines = correct_file.readlines()

    tags_path = path / 'tags.txt'
    with tags_path.open('r', encoding='utf-8') as tags_file:
        tags_lines = tags_file.readlines()

    items = []
    for original_line, correct_line, tags_line in zip(original_lines, correct_lines, tags_lines):
        original_id, original_text = original_line.split('\t')
        correct_id, correct_text = correct_line.split('\t')
        tags_split = tags_line.split()
        tags_id = tags_split[0]
        tags = tags_split[1:]
        assert(original_id == correct_id)
        assert(original_id == tags_id)
        items.append({
            '_id': original_id,
            'original': original_text,
            'correct': correct_text,
            'tags': tags,
        })

    return items


def save_dataset(path, items):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if len(items) == 0:
        return
    first_item = items[0]

    if 'original' in first_item:
        original_path = path / 'original.txt'
        with original_path.open('w', encoding='utf-8') as original_file:
            original_file.write(
                ''.join([
                    item['_id'] + '\t' + format_txt(item['original']) + '\n'
                    for item in items
                ])
            )

    if 'correct' in first_item:
        correct_path = path / 'correct.txt'
        with correct_path.open('w', encoding='utf-8') as correct_file:
            correct_file.write(
                ''.join([
                    item['_id'] + '\t' + format_txt(item['correct']) + '\n'
                    for item in items
                ])
            )

    if 'prediction' in first_item:
        prediction_path = path / 'prediction.txt'
        with prediction_path.open('w', encoding='utf-8') as prediction_file:
            prediction_file.write(
                ''.join([
                    item['_id'] + '\t' + format_txt(item['prediction']) + '\n'
                    for item in items
                ])
            )

    if 'tags' in first_item:
        tags_path = path / 'tags.txt'
        with tags_path.open('w', encoding='utf-8') as tags_file:
            tags_file.write(
                ''.join([
                    ' '.join([item['_id']] + item.get('tags', [])) + '\n'
                    for item in items
                ])
            )
