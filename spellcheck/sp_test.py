import code
from pprint import pprint
import sentencepiece as spm

from paths import FR_CORPUS_PATH

sp = spm.SentencePieceProcessor()
sp.Load("sp_.model")

with FR_CORPUS_PATH.open() as f:
    for line in f.readlines():
        print('-'*50)
        print(line)
        pprint(sp.EncodeAsPieces(line))
        code.interact(local=locals())
