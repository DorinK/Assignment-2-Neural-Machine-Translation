import sys
import torch
import numpy as np
import sacrebleu
from Utils import *
from EncoderDecoderTrain import EncoderRNN, DecoderRNN

""""""""""""""""""""""""""
#     Dorin Keshales
""""""""""""""""""""""""""

ENCODER_FILE = str(sys.argv[1])
DECODER_FILE = str(sys.argv[2])
DICT_FILE = str(sys.argv[3])
SRC_TEST = str(sys.argv[4])
TRG_TEST = str(sys.argv[5])


def evaluate(src_tensors, trg_tensors, encoder, decoder, i2t, trg_text, start_token='<s>', end_token='</s>',
             preds_file_name="./Outputs_Part_1/EncoderDecoderTest.pred"):

    preds = []

    with torch.no_grad():

        # Calculate the maximum sentence length from the target sentences
        max_length = max([len(sentence) for sentence in trg_tensors])
        for idx, (src_seq, trg_seq) in enumerate(zip(src_tensors, trg_tensors)):

            input_length = src_seq.size(0)
            target_length = trg_seq.size(0)

            # Initialize the encoder hidden vector and outputs matrix
            encoder_hidden = encoder.init_hidden()
            encoder_output = torch.zeros(1, 1, encoder.lstm_out_dim)

            # Forward pass via the Encoder
            for i in range(input_length):
                encoder_output, encoder_hidden = encoder(src_seq[i], encoder_hidden)

            # Initialize the decoder hidden vector
            decoder_hidden = decoder.init_hidden()

            decoded_words = []
            decoder_input = trg_seq[0]

            for i in range(max_length):

                # Forward pass via the Decoder
                decoder_output, decoder_hidden = decoder(encoder_output, decoder_input, decoder_hidden)

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

    # Produce predictions file
    test_predictions(preds, preds_file_name)

    # Return the BLEU score
    return bleu


def main():

    # Loading the dictionaries (vocabularies)
    dict = torch.load(DICT_FILE)

    s2i = dict['src_to_index']
    i2s = dict['index_to_src']
    t2i = dict['trg_to_index']
    i2t = dict['index_to_trg']

    # Redefining the Encoder and the Decoder
    encoder = EncoderRNN(embedding_dim=128, lstm_out_dim=128, vocab_size=len(s2i))
    decoder = DecoderRNN(vocab_size=len(t2i), embedding_dim=128, lstm_out_dim=256)

    # Loading the state dictionary of the Encoder
    state_dict_encoder = torch.load(ENCODER_FILE)
    encoder.load_state_dict(state_dict_encoder)

    # Loading the state dictionary of the Decoder
    state_dict_decoder = torch.load(DECODER_FILE)
    decoder.load_state_dict(state_dict_decoder)

    # Declaring evaluation mode
    encoder.eval()
    decoder.eval()

    """ Preparing the test set """

    # Loading the test set
    test_data, _, test_trg = read_data(SRC_TEST, TRG_TEST)

    dev_trg_text = update_with_unk(test_trg, t2i)

    # Update to indexes representation
    test_src_idx, test_trg_idx = src_and_trg_to_indexes(test_data, s2i, t2i)

    # Make each sequence a tensor
    src_tensors = [torch.LongTensor(x) for x in test_src_idx]
    trg_tensors = [torch.LongTensor(x) for x in test_trg_idx]

    """ Evaluating """

    # Compute the loss and the Bleu score on the test set
    bleu = evaluate(src_tensors, trg_tensors, encoder, decoder, i2t, dev_trg_text)

    print("BLEU Score on the Test Set: {:.3f}".format(bleu.score))


if __name__ == '__main__':
    main()
