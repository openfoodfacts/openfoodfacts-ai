import re
from generate_ingredients_test_set import get_context

PERCENTAGE_REGEX = r"([0-9]{0,2})([,|.|;| ]{0,2})([0-9]{0,2})[ ]?([%|\?|/|\\])"


def format_percentages(txt, keep_length=False, replacement_token=None):
    assert keep_length is False or replacement_token is None
    formatted_txt_list = []
    last_index = 0
    for match in re.finditer(PERCENTAGE_REGEX, txt):
        first_digits, sep, last_digits, _ = match.groups()
        if first_digits == "" and last_digits == "":
            # Not a good match
            formatted_match = txt[match.start() : match.end()]
        elif first_digits == "":
            formatted_match = replacement_token or f"{sep}{last_digits}%"
        elif last_digits == "":
            formatted_match = replacement_token or f"{first_digits}%"
        elif sep == "" or sep == " ":
            formatted_match = replacement_token or f"{first_digits}{last_digits}%"
        else:
            formatted_match = replacement_token or f"{first_digits},{last_digits}%"

        if True:
            print("-" * 30)
            print(txt[max(match.start() - 10, 0) : min(match.end() + 10, len(txt))])
            # print(txt[match.start() : match.end()])
            print(
                txt[max(match.start() - 10, 0) : match.start()]
                + formatted_match
                + txt[match.end() : min(match.end() + 10, len(txt))]
            )
            with open("charseqseq.txt", "a") as f:
                f.write(
                    (
                        txt[
                            max(match.start() - 10, 0) : min(match.end() + 10, len(txt))
                        ]
                    ).strip(" \t\r\n")
                )
                f.write("\n")
                f.write(
                    (
                        txt[max(match.start() - 10, 0) : match.start()]
                        + formatted_match
                        + txt[match.end() : min(match.end() + 10, len(txt))]
                    ).strip(" \t\r\n")
                )
                f.write("\n\n")

        formatted_txt_list.append(txt[last_index : match.start()])
        if keep_length:
            formatted_txt_list.append(
                " " * (match.end() - match.start() - len(formatted_match))
            )
        formatted_txt_list.append(formatted_match)
        last_index = match.end()
    formatted_txt_list.append(txt[last_index:])
    return "".join(formatted_txt_list)
