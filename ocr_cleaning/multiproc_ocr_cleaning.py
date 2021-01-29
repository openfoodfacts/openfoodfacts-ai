"""
Standalone script. Tested with Python 3.9.

Read a huge .jsonl file containing the OCR output for >2M products.
Pick only French products (EAN13 starting with a 3) and extract the
full text of the picture. Output is very light compared to input
since we lose characters coordinates.

Multiprocessing is used to speed up the operation. The input file is
opened once per process (32 in total). Writes to the output are done
only at the end (output is fully stored in memory).

Input file (ocr.jsonl) must be first downloaded and unzipped from OFF
servers (ask on Slack channel #robotoff for details).
- .gz file weights ~22GB (November 2020)
- .jsonl input file weights ~233GB
- .jsonl output file weights ~420MB

# Example of terminal output:
> python multiproc_ocr_cleaning.py
> Took 309.6s to read 2481313 lines. Average: 8013.5 lines/s
> Found 795609 fr texts out of 2481313 lines (32.1%)
> Took 0.6s to write 795609 lines to output (1261237.4 lines/s).
"""
import itertools
import json
import multiprocessing
import re
import time
import typing
from pathlib import Path

INPUT_PATH = Path("ocr.jsonl")  # weight 233GB !!
OUTPUT_PATH = Path("ocr_fr_text_only.jsonl")  # 420MB only

# Example of a French EAN 13: /325/622/514/8356/6.json (3 for FR)
EAN_13_FR_REGEX = re.compile(r"/3\d{2}/\d{3}/\d{3}/\d{4}/\d+?.json")
# Example of a EAN 8: /29888061/2.json
EAN_8_REGEX = re.compile(r"/\d{8}/\d+.json")

# Heuristics not used in script but interesting to know
AVERAGE_NB_CHARS_PER_LINE = 110000  # Heuristic from the file
TOTAL_NB_CHARS = 230000000000
# => ~2M products

N_PROCESSES = 32
# => ~65k products/process

MAX_NB_LINES_PER_PROCESS = -1
NB_CHARS_PER_PROCESS = TOTAL_NB_CHARS / N_PROCESSES


def pick_fr_text_only(line: str) -> typing.Optional[str]:
    """Parse the JSON input and return only source and text.

    If text is not FR or json schema is not the one expected, return None.
    """
    data = json.loads(line)
    source = data["source"]

    try:
        if len(data["content"]["responses"]) != 1:
            return None

        res = data["content"]["responses"][0]

        if res["textAnnotations"][0]["locale"] != "fr":
            return None

        fr_text = res["textAnnotations"][0]["description"]
        return json.dumps({"source": source, "fr_text": fr_text})

    except KeyError:
        return None


def search_fr_ean_13(offset_str: str):
    """Open the 260GB file and extract relevant lines only.

    Offset bounds are provided as a json-encoded dictionnary
    to ease the use of Python multiprocessing.

    Output tuple:
    - total number of lines read from file
    - list of relevant lines (already post-processed)
    """
    offset = json.loads(offset_str)
    offset_from = offset["from"]
    offset_to = offset["to"]

    with INPUT_PATH.open() as fh:
        # Go to offset
        fh.seek(offset_from)
        fh.readline()

        nb_read_lines = 0
        fr_text_lines = []
        while True:
            line = fh.readline()
            nb_read_lines += 1

            # If relevant barcode is found -> process ti
            if EAN_13_FR_REGEX.search(line):
                fr_text_line = pick_fr_text_only(line)
                if fr_text_line is not None:
                    fr_text_lines.append(fr_text_line)

            # Return earlier when limits are reached
            if nb_read_lines == MAX_NB_LINES_PER_PROCESS:
                break

            if fh.tell() >= offset_to:
                break

        return nb_read_lines, fr_text_lines


# Build offsets
# Dirty hack: offsets are sent as a json string to avoid "unhashable" issues.
offsets = [NB_CHARS_PER_PROCESS * iproc for iproc in range(N_PROCESSES + 1)]
offsets_str = {
    json.dumps({"from": offset_from, "to": offset_to})  # Dirty hack
    for offset_from, offset_to in zip(offsets[:-1], offsets[1:])
}

# Define pool of workers and process
t0 = time.time()
pool = multiprocessing.Pool(processes=N_PROCESSES)
result = pool.map(search_fr_ean_13, offsets_str)
nb_read_lines_list, fr_text_lines_list = list(zip(*result))  # unpack
t1 = time.time()
duration = t1 - t0

# Short summary
total_nb_read_lines = sum(nb_read_lines_list)
total_nb_matches = sum(len(lines) for lines in fr_text_lines_list)
print(
    f"Took {round(duration, 1)}s to read {total_nb_read_lines} lines. Average: {round(total_nb_read_lines/duration, 1)} lines/s"  # noqa: E501
)
print(
    f"Found {total_nb_matches} fr texts out of {total_nb_read_lines} lines ({round(100.0*total_nb_matches/total_nb_read_lines, 1)}%)"  # noqa: E501
)

# Write to output
# NB: output is fully stored in memory (<1GB)
t0 = time.time()
with OUTPUT_PATH.open("w") as f:
    for line in itertools.chain.from_iterable(fr_text_lines_list):
        f.write(line)
        f.write("\n")
t1 = time.time()
duration = t1 - t0
print(
    f"Took {round(duration, 1)}s to write {total_nb_matches} lines to output ({round(total_nb_matches/duration, 1)} lines/s)."  # noqa: E501
)
