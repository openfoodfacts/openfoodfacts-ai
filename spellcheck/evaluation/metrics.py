
import sys
from statistics import mean
from ingredients import process_ingredients


def evaluation_metrics(items, prediction_txts):
    txt_metrics = per_items_list_txt_metrics(items, prediction_txts)
    ingredients_metrics = [per_item_ingredients_metrics(item, prediction_txt)
                           for item, prediction_txt in zip(items, prediction_txts)]
    return {
        'txt_precision': txt_metrics['precision'],
        'txt_recall': txt_metrics['recall'],
        'ingredients_precision': mean([metric['precision'] for metric in ingredients_metrics]),
        'ingredients_recall': mean([metric['recall'] for metric in ingredients_metrics]),
    }


def per_items_list_txt_metrics(items, prediction_txts):
    """
    Precision :
        number of times we have the correct answer when we changed something
                / number of times we changed something

    Recall :
        number of times we have the correct answer when we changed something
                / number of times we should have changed something
    """
    number_correct_answers_when_changes = 0
    number_changes = 0
    number_should_have_been_changed = 0

    for item, prediction_txt in zip(items, prediction_txts):
        if item['original'] != prediction_txt:
            number_changes += 1
            if item['correct'] == prediction_txt:
                number_correct_answers_when_changes += 1
        if item['original'] != item['correct']:
            number_should_have_been_changed += 1

    return {
        'precision': 100.0 * number_correct_answers_when_changes / (number_changes + sys.float_info.epsilon),
        'recall': 100.0 * number_correct_answers_when_changes / (number_should_have_been_changed + sys.float_info.epsilon),
    }


def per_item_ingredients_metrics(item, prediction_txt):
    """
    Precision :
        number of times we have predicted a correct ingredient
            / number of predicted ingredients

    Recall :
        number of times we have predicted a correct ingredient
            / number of correct ingredients
    """
    correct_ingredients = format_ingredients(item['correct'])
    predicted_ingredients = format_ingredients(prediction_txt)

    true_positives = predicted_ingredients & correct_ingredients

    return {
        'precision': 100.0 * len(true_positives) / (len(predicted_ingredients) + sys.float_info.epsilon),
        'recall': 100.0 * len(true_positives) / (len(correct_ingredients) + sys.float_info.epsilon),
    }


def format_ingredients(ingredients_txt):
    return {
        ' '.join(ingredient.split())
        for ingredient
        in process_ingredients(ingredients_txt).iter_normalized_ingredients()
    }
