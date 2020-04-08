from paths import FR_CORPUS_PATH, FR_LOWER_CORPUS_PATH


with FR_CORPUS_PATH.open() as fr_corpus:
    with FR_LOWER_CORPUS_PATH.open('w') as fr_lower_corpus:
        for line in fr_corpus.readlines():
            fr_lower_corpus.write(line.lower() + '\n')
