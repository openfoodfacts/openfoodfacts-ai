import difflib


def show_diff(original_text: str, corrected_text: str, missing_element: str = "~"):
    """Unify operations between two compared strings
    seqm is a difflib.SequenceMatcher instance whose a & b are strings
    """
    # Check if the process was not done
    if "<mark>" not in corrected_text:
        seqm = difflib.SequenceMatcher(None, original_text, corrected_text) 
        output= []
        for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
            if opcode == 'equal':
                output.append(seqm.a[a0:a1])
            elif opcode == 'insert':
                output.append("<mark>" + seqm.b[b0:b1] + "</mark>")
            elif opcode == 'delete':
                output.append("<mark>" + missing_element + "</mark>")
            elif opcode == 'replace':
                output.append("<mark>" + seqm.b[b0:b1] + "</mark>")
            else:
                raise RuntimeError("unexpected opcode")
        return ''.join(output)
    else:
        return corrected_text


if __name__ == "__main__":

    # Example usage:
    original_text = "The fast brown fox jum - ped over the lazy dog."
    corrected_text = "The qu - ick brown fox jumped over the lazy dog."

    # highlighted_result = highlight_corrected(original_text, corrected_text)
    # print(highlighted_result)

    print(show_diff(original_text, corrected_text))