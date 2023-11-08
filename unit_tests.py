from extract_training_data import *


def main():
    buffer = []
    with open("test.conll", "r") as in_file:
        extractor = FeatureExtractor("data/words.vocab", "data/pos.vocab")
        print("Starting feature extraction... (each . represents 100 sentences)")
        inputs, outputs = get_training_matrices(extractor, in_file)


if __name__ == "__main__":
    main()
