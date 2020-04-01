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

    items = []
    for original_line, correct_line in zip(original_lines, correct_lines):
        original_id, original_text = original_line.split('\t')
        correct_id, correct_text = correct_line.split('\t')
        assert(original_id == correct_id)
        items.append({
            '_id': original_id,
            'original': original_text,
            'correct': correct_text,
        })

    return items


def save_dataset(path, items):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    original_path = path / 'original.txt'
    with original_path.open('w', encoding='utf-8') as original_file:
        original_file.write(
            ''.join([
                item['_id'] + '\t' + format_txt(item['original']) + '\n'
                for item in items
            ])
        )

    correct_path = path / 'correct.txt'
    with correct_path.open('w', encoding='utf-8') as correct_file:
        correct_file.write(
            ''.join([
                item['_id'] + '\t' + format_txt(item['correct']) + '\n'
                for item in items
            ])
        )
