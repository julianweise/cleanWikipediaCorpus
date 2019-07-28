import argparse
import logging
from pathlib import Path

from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class WordCorpus:

    def __init__(self, location: Path):
        self._location: Path = location

    def __iter__(self):
        for line in self._location.open("r"):
            for word in line.replace("\n", ""):
                yield word


def main(args):
    logging.info("Start Training!")

    cleaned_corpus: WordCorpus = WordCorpus(args.corpus)
    model: Word2Vec = Word2Vec(cleaned_corpus, size=100, window=5, min_count=1, workers=16, sg=1)
    model.save(Path(args.output, "wiki_word2vec_binary.model"))
    model.wv.save_word2vec_format(Path(args.output, "wiki_word2vec_c_format.txt"), binary=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Path to the cleaned Wikipedia corpus",
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to an locatin where the model should be saved",
        required=True
    )
    main(parser.parse_args())