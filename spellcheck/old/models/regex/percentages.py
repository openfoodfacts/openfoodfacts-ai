import re

PERCENTAGE_REGEX = re.compile(
    r"(\A|.)([0-9]{0,2})([ ]{0,1}?[,|.|;|/]{0,1}[ ]{0,1})([0-9]{0,2})[ ]?(?:%|(?:[\?|/|\\](?:\D|\Z)))"
)
ADDITIVES_REGEX = re.compile(r"(?:E ?\d{3,5}[a-z]*)", re.IGNORECASE)


def format_percentages(txt: str) -> str:
    formatted_txt_list = []
    last_index = 0
    for match in PERCENTAGE_REGEX.finditer(txt):
        first_char, first_digits, sep, last_digits = match.groups()

        start = match.start() + len(first_char)
        end = match.end()  # - len(to_drop)
        nb_first_digits = len(first_digits)
        nb_last_digits = len(last_digits)

        valid_match = False
        pad_before = False
        pad_after = False

        if ADDITIVES_REGEX.match(txt[match.start() : match.end()]):
            # Very conservative rule
            formatted_match = txt[start:end]

        elif nb_first_digits == 0 and nb_last_digits == 0:
            # Not a good match
            formatted_match = txt[start:end]

        elif nb_first_digits == 0:
            formatted_match = f"{sep}{last_digits}%"
            pad_before = False
            pad_after = True

        elif nb_last_digits == 0:
            formatted_match = f"{first_digits}%"
            pad_before = True
            pad_after = True

        elif len(sep) > 0 and (nb_first_digits == 2 or nb_last_digits == 2):
            formatted_match = f"{first_digits},{last_digits}%"
            pad_before = True
            pad_after = True

        elif sep.strip() == "":
            if float(f"{first_digits}{last_digits}") <= 100.0:
                formatted_match = f"{first_digits}{last_digits}%"
                pad_before = True
                pad_after = True
            else:
                formatted_match = f"{first_digits},{last_digits}%"
                pad_before = True
                pad_after = True
        else:
            formatted_match = f"{first_digits},{last_digits}%"
            pad_before = True
            pad_after = True

        if pad_before:
            if start > 0:
                previous_char = txt[start - 1]
                if previous_char.isalnum() or previous_char in ["*", ")", "]", "}"]:
                    formatted_match = " " + formatted_match.lstrip(" ")

        if pad_after:
            if end < len(txt):
                next_char = txt[end]
                if next_char.isalnum() or next_char in ["*", "(", "[", "{"]:
                    formatted_match = formatted_match.rstrip(" ") + " "

        formatted_txt_list.append(txt[last_index:start])
        formatted_txt_list.append(formatted_match)
        last_index = end
    formatted_txt_list.append(txt[last_index:])
    return "".join(formatted_txt_list)


if __name__ == "__main__":
    TEST_PATH = "test_sets/percentages/fr.txt"

    with open(TEST_PATH, "r") as f:
        test_set = [tuple(item.split("\n")[:2]) for item in f.read().split("\n\n")]

    nb_errors = 0
    for original, correct in test_set:
        formatted = format_percentages(original)
        if formatted != correct:
            nb_errors += 1
            print(str(nb_errors) + " -" * 60)
            print(original)
            print(correct)
            print(formatted)
