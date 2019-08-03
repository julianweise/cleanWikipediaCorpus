import argparse
import logging
from pathlib import Path

from gensim.corpora import WikiCorpus


def main(args):
    dump_path: Path = args.wiki_dump
    output_path: Path = args.output

    wiki = WikiCorpus(str(dump_path.absolute()), lemmatize=False)

    logging.info("Start processing Wikipedia Dump")

    # Ensure pip install pattern has been installed
    with Path(output_path, "cleaned_wikipedia_corpus.txt").open("w+") as output_stream:
        articles_counter = 0
        for text in wiki.get_texts():
            articles_counter += 1
            output_stream.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')

            if articles_counter % 10000 == 0:
                logging.info(f"Processed {articles_counter} articles")

    logging.info("Successfully processed all articles of dump")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : [%(threadName)s] %(levelname)s : %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wiki-dump",
        type=Path,
        help="Path to the wikipedia dump.",
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to an locatin where the result will be saved",
        required=True
    )
    main(parser.parse_args())

