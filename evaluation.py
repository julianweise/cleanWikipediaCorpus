import argparse
import logging
import time
from pathlib import Path
from typing import Tuple

from gensim.models import KeyedVectors


def main(args):
    embedding_file: Path = args.embedding
    analogy_test_set_file: Path = args.analogy_test_set
    similarity_test_set_file: Path = args.similarity_test_set

    logging.info("Start Evaluation Framework")

    model: KeyedVectors = KeyedVectors.load_word2vec_format(str(embedding_file.absolute()))
    run_analogy_test(model, analogy_test_set_file)
    # run_similarity_test(model, similarity_test_set_file)


def run_analogy_test(model: KeyedVectors, test_set: Path):
    logging.info(f"Start Analogy Testing")
    start = time.time()
    accuracy, sections = model.evaluate_word_analogies(str(test_set.absolute()), case_insensitive=True, restrict_vocab=None)
    end = time.time()
    logging.info(f"Analogy Testing finished")
    logging.info(f"Analogy Test Result: {accuracy * 100}%. Test execution took {end-start} sec.")


def run_similarity_test(model: KeyedVectors, test_set: Path):
    logging.info(f"Start Similarity Testing")
    start = time.time()
    result: Tuple[Tuple[float, float]] = model.evaluate_word_pairs(str(test_set.absolute()))
    pearson: Tuple[float, float] = result[0]
    spearman: Tuple[float, float] = result[1]
    end = time.time()
    logging.info(f"Similarity Testing finished")
    logging.info(f"Similarity Test Result: pearson: {pearson}, spearman: {spearman}. Test execution took {end-start} sec.")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : [%(threadName)s] %(levelname)s : %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding",
        type=Path,
        help="Path to the embedding file",
        required=True
    )
    parser.add_argument(
        "--analogy-test-set",
        type=Path,
        help="Path to the analogy test set",
        required=True
    )
    parser.add_argument(
        "--similarity-test-set",
        type=Path,
        help="Path to the similarity test set",
        required=True
    )
    main(parser.parse_args())