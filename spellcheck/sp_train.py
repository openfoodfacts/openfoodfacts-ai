import sentencepiece as spm
from paths import FR_CORPUS_PATH, FR_LOWER_CORPUS_PATH

path = FR_CORPUS_PATH
path = FR_LOWER_CORPUS_PATH

spm.SentencePieceTrainer.Train(
    f'--input={path} --model_prefix=sp_ --model_type=bpe --vocab_size=10000')
