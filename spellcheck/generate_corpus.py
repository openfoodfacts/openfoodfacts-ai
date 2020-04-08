from mongo import products
from utils import format_txt

from paths import FR_CORPUS_PATH

query = products.aggregate([
    {'$match': {'ingredients_text_fr': {'$exists': True, '$ne': ''}}},
    {'$sample': {'size': 100000}},
    {'$project': {'ingredients_text_fr': 1}},
])

with FR_CORPUS_PATH.open('a') as f:
    for item in query:
        f.write(format_txt(item['ingredients_text_fr']))
        f.write('\n')
