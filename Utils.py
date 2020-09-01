import os
import matplotlib.pyplot as plt
from collections import Counter

""""""""""""""""""""""""""
#     Dorin Keshales
""""""""""""""""""""""""""


def read_data(src_file, trf_file, start_token='<s>', end_token='</s>'):

    # Read all the source sentences
    with open(src_file, 'r', encoding='utf-8') as f:
        f_data = f.readlines()

    # Read all the target sentences
    with open(trf_file, 'r', encoding='utf-8') as e:
        e_data = e.readlines()

    # Create list of tuples where each tuple is composed from the source and target sentences
    d = [tuple([start_token] + sentence.strip().split() + [end_token] for sentence in pair) for pair in
         zip(f_data, e_data)]

    src = [elem[0] for elem in d]
    trg = [elem[1] for elem in d]

    return d, src, trg


# Considerate rare words like if the were unknown words in order to train the corresponding embedding vector
def convert_rare_words_to_unknown_token(data, num_occurrences=3, unknown_token='<unk>'):

    count = Counter()
    convert_to_unk = set()

    # Count the number of occurrences of each word in the training set
    for sentence in data:
        count.update(sentence)

    # Collect the words in the training set that appear only once
    for word, amount in count.items():
        if amount <= num_occurrences:
            convert_to_unk.add(word)

    # Go over each sentence in the training set
    for sentence in data:

        # For each word in the sentence
        for i in range(len(sentence)):

            # If the current word appears only once then considerate it as unknown word
            if sentence[i] in convert_to_unk:
                sentence[i] = unknown_token

    # Return the updated training set data.
    return data


# Making a words vocabulary and tags vocabulary where each word, tag in the training set has a unique index
def create_words_vocabulary(data, start_token='<s>', end_token='</s>', unknown_token='<unk>'):

    vocab_words = set()

    # Go over each sentence in the data set
    for sentence in data:

        # For each word in the sentence
        for word in sentence:
            vocab_words.add(word)

    # Remove the unknown_token, start_token and end_token from the set in order to give them meaningful indexes
    vocab_words.remove(unknown_token)
    vocab_words.remove(start_token)
    vocab_words.remove(end_token)

    # Sort the set
    vocab_words = sorted(vocab_words)

    # Add the unknown_token, start_token and end_token with indexes 0, 1, 2 respectively
    vocab_words = [unknown_token, start_token, end_token] + vocab_words

    # Map each word to a unique index
    word_to_ix = {word: i for i, word in enumerate(vocab_words)}
    ix_to_word = {i: word for i, word in enumerate(vocab_words)}

    return word_to_ix, ix_to_word


# Replace each source word and target word with their unique indexes.
def src_and_trg_to_indexes(data, src_to_idx, trg_to_idx, unknown_token='<unk>'):

    d = [([src_to_idx[word] if word in src_to_idx else src_to_idx[unknown_token] for word in src],
          [trg_to_idx[word] if word in trg_to_idx else trg_to_idx[unknown_token] for word in trg]) for src, trg in data]

    src = [elem[0] for elem in d]
    trg = [elem[1] for elem in d]

    return src, trg


# Update the textual data withe the words which were replaced with the unknown token
def update_with_unk(trg_data, trg_to_idx, unknown_token='<unk>'):
    return [[word if word in trg_to_idx else unknown_token for word in seq] for seq in trg_data]


# Replace each word in the data set with its corresponding index
def convert_data_to_indexes(data, vocab, unknown_token='<unk>'):

    sentences, words_indexes = [], []

    # Go over each sentence in the training set
    for sentence in data:

        # For each word in the sentence
        for word in sentence:
            # Find its corresponding index - if not exist then assign the index of the unknown_token
            ix = vocab.get(word) if word in vocab else vocab.get(unknown_token)
            words_indexes.append(ix)

        # Keep the words in the data set in sentences order
        sentences.append(words_indexes)
        words_indexes = []

    # Return the updated data
    return sentences


def plot_graph(title, to_plot, color):

    plt.title(title + " vs Epochs")

    # Plot
    ticks = [i for i in range(1, len(to_plot) + 1)]
    plt.plot(ticks, to_plot, color=color)

    # x y labels
    plt.ylabel(title)
    plt.xlabel("Epochs")

    plt.xticks(ticks)
    plt.show()


def test_predictions(predictions, file_name, start_token='<s>', end_token='</s>'):

    # Clear the content of the file if it already exists; Otherwise, create the file
    if os.path.exists(file_name):
        os.remove(file_name)
    f = open(file_name, "a+")

    # Write each prediction to the predictions file
    for prediction in predictions:
        f.write("%s\n" % prediction[len(start_token) + 1:])

    # Close the file
    f.close()
