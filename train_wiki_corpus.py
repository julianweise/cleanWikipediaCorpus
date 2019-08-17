import argparse
import logging
from pathlib import Path

from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main(args):
    logging.info("Start Training!")

    model: Word2Vec = Word2Vec(corpus_file=str(args.corpus.absolute()), size=100, window=5, min_count=5, workers=16,
                               sg=1, iter=args.epochs, negative=5)
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
