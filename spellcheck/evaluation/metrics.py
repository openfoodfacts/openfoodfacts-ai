
import sys
from statistics import mean
from difflib import SequenceMatcher
from ingredients import process_ingredients


# TODO : refacto this scripts as a proper Evaluation class

def evaluation_metrics(items, prediction_txts):
    txt_metrics = per_items_list_txt_metrics(items, prediction_txts)
    # ingredients_metrics = [per_item_ingredients_metrics(item, prediction_txt, remove_originals=False)
    #                        for item, prediction_txt in zip(items, prediction_txts)]
    errors_ingredients_metrics = [per_item_ingredients_metrics(item, prediction_txt, remove_originals=True)
                                  for item, prediction_txt in zip(items, prediction_txts)]
    similarity_metric = [per_item_similarity_based_metric(item, prediction_txt)
                         for item, prediction_txt in zip(items, prediction_txts)]
    return {
        'number_items': txt_metrics['number_items'],
        'number_changed': txt_metrics['number_changed'],
        'number_should_have_been_changed': txt_metrics['number_should_have_been_changed'],
        'txt_precision': txt_metrics['precision'],
        'txt_recall': txt_metrics['recall'],
        # 'ingredients_precision': mean([metric['precision'] for metric in ingredients_metrics]),
        # 'ingredients_recall': mean([metric['recall'] for metric in ingredients_metrics]),
        'ingredients_precision_on_errors': mean([metric['precision'] for metric in errors_ingredients_metrics if metric is not None]),
        'ingredients_recall_on_errors': mean([metric['recall'] for metric in errors_ingredients_metrics if metric is not None]),
        'similarity_metric': mean(similarity_metric),
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
    number_changed = 0
    number_should_have_been_changed = 0

    for item, prediction_txt in zip(items, prediction_txts):
        if item['original'] != prediction_txt:
            number_changed += 1
            if item['correct'] == prediction_txt:
                number_correct_answers_when_changes += 1
        if item['original'] != item['correct']:
            number_should_have_been_changed += 1

    return {
        'number_items': len(items),
        'number_changed': number_changed,
        'number_should_have_been_changed': number_should_have_been_changed,
        'precision': 100.0 * number_correct_answers_when_changes / (number_changed + sys.float_info.epsilon),
        'recall': 100.0 * number_correct_answers_when_changes / (number_should_have_been_changed + sys.float_info.epsilon),
    }


def per_item_ingredients_metrics(item, prediction_txt, remove_originals=False):
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

    if remove_originals:
        original_ingredients = format_ingredients(item['original'])
        predicted_ingredients = predicted_ingredients - (original_ingredients & correct_ingredients)
        correct_ingredients = correct_ingredients - original_ingredients
        if len(correct_ingredients) == 0:
            # Original ingredients == Correct ingredient -> not relevant case
            return

    true_positives = (predicted_ingredients & correct_ingredients)

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


def per_item_similarity_based_metric(item, prediction_txt):
    matcher = SequenceMatcher(is_junk, item['correct'], prediction_txt)
    return 100.0 * matcher.ratio()


def is_junk(c):
    return False
