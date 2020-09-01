import sys
import time
import torch
import numpy as np
import sacrebleu
import torch.nn as nn
from torch import optim
from Utils import *
import torch.nn.functional as F

""""""""""""""""""""""""""
#     Dorin Keshales
""""""""""""""""""""""""""

SRC_TRAIN = str(sys.argv[1])
TRG_TRAIN = str(sys.argv[2])
SRC_DEV = str(sys.argv[3])
TRG_DEV = str(sys.argv[4])

# Create a directory in the current working directory in which the output files will be saved
output_dir = "./Outputs_Part_2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ENCODER_FILE = output_dir + "/encoderFile"
DECODER_FILE = output_dir + "/decoderFile"
DICT_FILE = output_dir + "/dictFile"


class EncoderRNN(nn.Module):

    def __init__(self, embedding_dim, lstm_out_dim, vocab_size):
        super(EncoderRNN, self).__init__()

        # Keep parameters for reference
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_out_dim = lstm_out_dim

        # Embedding matrix
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Unidirectional lstm with one layer that will encode the input sequence
        self.lstm = nn.LSTM(embedding_dim, lstm_out_dim)

    def forward(self, word_input, hidden):

        # Get the embedding of the current input word
        embedded = self.embeddings(word_input).view(1, 1, -1)

        # Run through LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)

        return lstm_out, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.lstm_out_dim), torch.zeros(1, 1, self.lstm_out_dim)


class DecoderRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, lstm_out_dim, drop=0.3):
        super(DecoderRNN, self).__init__()

        # Keep parameters for reference
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_out_dim = lstm_out_dim

        # Embedding matrix
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Unidirectional lstm with one layer as the decoder
        self.lstm = nn.LSTM(self.embedding_dim + self.lstm_out_dim, self.lstm_out_dim)

        # Dropout layer
        self.dropout = nn.Dropout(drop)

        # Fully connected layer as part of calculating the attention weights
        self.attn = nn.Linear(self.embedding_dim + self.lstm_out_dim, self.lstm_out_dim)

        # Activation function
        self.tanh = nn.Tanh()

        # Tensor that is considered as module parameter and will be learned during the training
        self.v = nn.Parameter(torch.rand(self.lstm_out_dim) * 0.1)

        # Fully connected layer as part of computing ht~
        self.attn_hidden = nn.Linear(self.embedding_dim + self.lstm_out_dim, self.lstm_out_dim)

        # Fully connected layer
        self.out = nn.Linear(self.lstm_out_dim, self.vocab_size)

    def forward(self, encoder_out, word_input, last_context, hidden):

        # Get the embedding of the current input word
        embedded = self.embedding(word_input).view(1, 1, -1)

        # Combine last context and embedded input word, run through RNN
        lstm_input = torch.cat((last_context.unsqueeze(0), embedded), 2)
        ht, hidden = self.lstm(lstm_input, hidden)

        # Apply dropout to prevent over-fitting
        ht = self.dropout(ht)

        # Initialize the attention weights vector
        attn_weights = torch.zeros(len(encoder_out))

        # Calculate attention from current RNN state and all encoder outputs
        for i in range(len(encoder_out)):
            attn_weights[i] = self.v.dot(
                self.tanh(self.attn(torch.cat((ht[0], encoder_out[i].view(1, -1)), dim=1))).view(-1))

        # Normalize attention weights to weights in range 0 to 1
        normalized_weights = F.softmax(attn_weights, dim=0).unsqueeze(0).unsqueeze(0)

        # Make the context vector by applying the attention weights to encoder outputs
        context = normalized_weights.bmm(encoder_out.unsqueeze(0))

        # Calculate the attentional vector ht~
        ht_tilda = self.tanh(self.attn_hidden(torch.cat((context, ht), dim=2)))

        # Feeding ht~ through the softmax layer and producing predictive distribution
        output = F.log_softmax(self.out(ht_tilda.view(1, -1)), dim=1)

        return output, hidden, ht_tilda[0], normalized_weights[0]

    def init_hidden(self):
        return torch.zeros(1, 1, self.lstm_out_dim), torch.zeros(1, 1, self.lstm_out_dim)


def train(train_src, train_trg, dev_src, dev_trg, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          trg_text, i2t, epochs=10):

    start_time = time.time()

    dev_losses, bleu_scores = [], []
    best_bleu_score = 0.0

    for epoch in range(1, epochs + 1):

        # Declaring training mode
        encoder.train()
        decoder.train()

        # Shuffle the data before starting a new epoch
        c = list(zip(train_src, train_trg))
        np.random.shuffle(c)
        src, trg = zip(*c)

        sum_loss = 0.0
        for idx, (src_seq, trg_seq) in enumerate(zip(src, trg)):

            # Zero gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_length = src_seq.size(0)
            target_length = trg_seq.size(0)

            # The current sentence loss
            loss = 0.0

            # Initialize the encoder hidden vector and outputs matrix
            encoder_hidden = encoder.init_hidden()
            encoder_outputs = torch.zeros(input_length, encoder.lstm_out_dim)

            for i in range(input_length):

                # Forward pass via the Encoder
                encoder_output, encoder_hidden = encoder(src_seq[i], encoder_hidden)

                # Save each encoder output
                encoder_outputs[i] = encoder_output.view(-1)

            # Initialize the decoder hidden and context vectors
            decoder_hidden = decoder.init_hidden()
            decoder_context = torch.zeros(1, decoder.lstm_out_dim)

            for i in range(target_length - 1):

                # Forward pass via the Decoder
                decoder_output, decoder_hidden, decoder_context, decoder_attention = decoder(
                    encoder_outputs, trg_seq[i], decoder_context, decoder_hidden)

                # Summing the NLL loss on every token
                loss += criterion(decoder_output, trg_seq[i + 1].unsqueeze(0))

            # Averaging the total loss of the sentence
            sum_loss += (loss.item() / (target_length - 1))

            # Back propagation
            loss.backward()

            # Updating weights
            encoder_optimizer.step()
            decoder_optimizer.step()

        # Compute the loss on the training set in the current epoch
        train_loss = sum_loss / len(train_src)

        # Compute the loss and the Bleu score on the dev set in the current epoch
        dev_loss, bleu = evaluate(dev_src, dev_trg, encoder, decoder, criterion, i2t, trg_text)

        # Save the dev's loss and BLEU score results.
        dev_losses.append(dev_loss)
        bleu_scores.append(bleu.score)

        # Save the model with the best BLEU score
        if bleu.score > best_bleu_score:
            best_bleu_score = bleu.score
            torch.save(encoder.state_dict(), ENCODER_FILE)
            torch.save(decoder.state_dict(), DECODER_FILE)

        print("Epoch: {}/{}...".format(epoch, epochs),
              "Train Loss: {:.3f}...".format(train_loss),
              "Dev Loss: {:.3f}...".format(dev_loss),
              "BLEU Score: {:.3f}".format(bleu.score))

    passed_time = time.time() - start_time
    print('The training ended after %.3f minutes' % (passed_time / 60))

    return dev_losses, bleu_scores


def evaluate(src_tensors, trg_tensors, encoder, decoder, criterion, i2t, trg_text, start_token='<s>', end_token='</s>'):

    # Declaring evaluation mode
    encoder.eval()
    decoder.eval()

    sum_loss = 0.0
    preds = []

    with torch.no_grad():

        # Shuffle the data before starting a new epoch
        c = list(zip(src_tensors, trg_tensors, trg_text))
        np.random.shuffle(c)
        src_tensors, trg_tensors, trg_text = zip(*c)

        # Calculate the maximum sentence length from the target sentences
        max_length = max([len(sentence) for sentence in trg_tensors])
        for idx, (src_seq, trg_seq) in enumerate(zip(src_tensors, trg_tensors)):

            input_length = src_seq.size(0)
            target_length = trg_seq.size(0)

            # The current sentence loss
            loss = 0.0

            # Initialize the encoder hidden vector and outputs matrix
            encoder_hidden = encoder.init_hidden()
            encoder_outputs = torch.zeros(input_length, encoder.lstm_out_dim)

            for i in range(input_length):

                # Forward pass via the Encoder
                encoder_output, encoder_hidden = encoder(src_seq[i], encoder_hidden)

                # Save each encoder output
                encoder_outputs[i] = encoder_output.view(-1)

            """""Calculating the loss"""""

            # Initialize the decoder hidden and context vectors
            decoder_hidden = decoder.init_hidden()
            decoder_context = torch.zeros(1, decoder.lstm_out_dim)

            for i in range(target_length - 1):

                # Forward pass via the Decoder
                decoder_output, decoder_hidden, decoder_context, decoder_attention = decoder(
                    encoder_outputs, trg_seq[i], decoder_context, decoder_hidden)

                # Summing the NLL loss on every token
                loss += criterion(decoder_output, trg_seq[i + 1].unsqueeze(0))

            # Averaging the total loss of the sentence
            sum_loss += (loss.item() / (target_length - 1))

            """""Calculating the BLEU Score"""""

            # Initialize the decoder hidden and context vectors
            decoder_hidden = decoder.init_hidden()
            decoder_context = torch.zeros(1, decoder.lstm_out_dim)

            decoded_words = []
            decoder_input = trg_seq[0]

            for i in range(max_length):

                # Forward pass via the Decoder
                decoder_output, decoder_hidden, decoder_context, decoder_attention = decoder(
                    encoder_outputs, decoder_input, decoder_context, decoder_hidden)

                # Make last decoder's prediction as input to the next decoder step
                decoder_input = np.argmax(decoder_output.data, axis=1)

                # Save each decoded words
                decoded_words.append(i2t[decoder_input.item()])

                # Stop decoding when prediction the EOS token
                if i2t[decoder_input.item()] == end_token:
                    break

            # Save the decoded sentence
            preds.append(' '.join([start_token] + decoded_words))

        # References for the BLEU score computation
        refs = [' '.join(ref) for ref in trg_text]

        # Calculate the BLEU score
        bleu = sacrebleu.corpus_bleu(preds, [refs])

    # Return the loss and the BLEU score
    return sum_loss / len(src_tensors), bleu


def main():

    """" Preparing the training set """""

    # Loading the training set
    train_data, train_src, train_trg = read_data(SRC_TRAIN, TRG_TRAIN)

    # Consider rare words as unknown words
    train_src = convert_rare_words_to_unknown_token(train_src)
    train_trg = convert_rare_words_to_unknown_token(train_trg)

    # Assigning a unique index to each source word and target word in the vocabulary
    s2i, i2s = create_words_vocabulary(train_src)
    t2i, i2t = create_words_vocabulary(train_trg)

    # Update to indexes representation
    train_src_idx, train_trg_idx = src_and_trg_to_indexes(train_data, s2i, t2i)

    # Make each sequence a tensor
    train_src_tensors = [torch.LongTensor(x) for x in train_src_idx]
    train_trg_tensors = [torch.LongTensor(x) for x in train_trg_idx]

    """" Preparing the dev set """""

    # Loading the dev set
    dev_data, _, dev_trg = read_data(SRC_DEV, TRG_DEV)

    dev_trg_text = update_with_unk(dev_trg, t2i)

    # Update to indexes representation
    dev_src_idx, dev_trg_idx = src_and_trg_to_indexes(dev_data, s2i, t2i)

    # Make each sequence a tensor
    dev_src_tensors = [torch.LongTensor(x) for x in dev_src_idx]
    dev_trg_tensors = [torch.LongTensor(x) for x in dev_trg_idx]

    """" Model Settings """""

    # Create instances of the Encoder and the Decoder
    encoder = EncoderRNN(embedding_dim=128, lstm_out_dim=128, vocab_size=len(s2i))
    decoder = DecoderRNN(vocab_size=len(t2i), embedding_dim=128, lstm_out_dim=256)

    # Loss function
    criterion = nn.NLLLoss()

    # Using Adam as optimizer
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.00015)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.00015)

    # Training
    losses, bleu_scores = train(train_src_tensors, train_trg_tensors, dev_src_tensors, dev_trg_tensors, encoder,
                                decoder, encoder_optimizer, decoder_optimizer, criterion, dev_trg_text, i2t)

    # Plot loss and BLEU Score graphs
    plot_graph("Dev Loss", losses, color='teal')
    plot_graph("BLEU Score", bleu_scores, color='indigo')

    # Saving the dictionaries (vocabularies)
    torch.save({
        'src_to_index': s2i,
        'index_to_src': i2s,
        'trg_to_index': t2i,
        'index_to_trg': i2t
    }, DICT_FILE)


if __name__ == '__main__':
    main()
