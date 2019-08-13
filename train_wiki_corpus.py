import argparse
import logging
from pathlib import Path

from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class WordCorpus:

    def __init__(self, location: Path):
        self._location: Path = location

    def __iter__(self):
        with self._location.open("r") as stream:
            for line in stream:
                if not line:
                    continue
                yield from [word for word in line.replace("\n", "").split(" ") if word]


def main(args):
    logging.info("Start Training!")

    cleaned_corpus: WordCorpus = WordCorpus(args.corpus)
    model: Word2Vec = Word2Vec(cleaned_corpus, size=100, window=5, min_count=1, workers=16, sg=1, iter=args.epochs)
    model.save(str(Path(args.output, "wiki_word2vec_binary.model").absolute()))
    model.wv.save_word2vec_format(str(Path(args.output, "wiki_word2vec_c_format.txt").absolute()), binary=False)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : [%(threadName)s] %(levelname)s : %(message)s", level=logging.INFO)

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
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Epochs to train",
        required=True
    )
    main(parser.parse_args())
