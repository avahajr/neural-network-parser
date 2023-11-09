from extract_training_data import *


def main():
    WORD_VOCAB_FILE = "data/words.vocab"
    POS_VOCAB_FILE = "data/pos.vocab"

    try:
        word_vocab_f = open(WORD_VOCAB_FILE, "r")
        pos_vocab_f = open(POS_VOCAB_FILE, "r")
    except FileNotFoundError:
        print(
            "Could not find vocabulary files {} and {}".format(
                WORD_VOCAB_FILE, POS_VOCAB_FILE
            )
        )
        sys.exit(1)
    buffer = []
    with open("data/test.conll", "r") as in_file:
        extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
        print("Starting feature extraction... (each . represents 100 sentences)")
        inputs, outputs = get_training_matrices(extractor, in_file)


if __name__ == "__main__":
    main()
