import os
import random
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import EncoderRNN, DecoderRNN, AttnDecoderRNN

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
SOS_token = 2
EOS_token = 3

def load_files(data_dir):
    with open(os.path.join(data_dir, 'train_origin_ids.pickle'), 'rb') as f:
        train_ids = pickle.load(f)
    with open(os.path.join(data_dir, 'train_target_ids.pickle'), 'rb') as f:
        target_ids = pickle.load(f)
    with open(os.path.join(data_dir, 'test_origin_ids.pickle'), 'rb') as f:
        test_ids = pickle.load(f)

    return train_ids, target_ids, test_ids


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
        decoder_optimizer, criterion, max_length=20, teacher_forcing_ratio = 0.5):
    encoder_hidden = encoder.initHidden(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # if use_teacher_forcing:
    #     # Teacher forcing: Feed the target as the next input
    #     for di in range(target_length):
    #         decoder_output, decoder_hidden, decoder_attention = decoder(
    #             decoder_input, decoder_hidden, encoder_outputs)
    #         loss += criterion(decoder_output, target_tensor[di])
    #         decoder_input = target_tensor[di]  # Teacher forcing
    # else:
    #     # Without teacher forcing: use its own predictions as the next input
    #     for di in range(target_length):
    #         decoder_output, decoder_hidden, decoder_attention = decoder(
    #             decoder_input, decoder_hidden, encoder_outputs)
    #         topv, topi = decoder_output.topk(1)
    #         decoder_input = topi.squeeze().detach()  # detach from history as input

    #         loss += criterion(decoder_output, target_tensor[di])
    #         if decoder_input.item() == EOS_token:
    #             break
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def test():
    pass

def run(data_dir, hidden_size=256, learning_rate=0.01, epoches=100, max_length=20, teacher_forcing_ratio = 0.5):
    # load data
    train_x, train_y, test_x = load_files(data_dir)
    train_words_num = max(max(train_x))
    target_words_num = max(max(train_y))
    train_x, train_y, test_x = train_x[:10], train_y[:10], test_x[:10]
    # get model
    encoder = EncoderRNN(train_words_num, hidden_size).to(device)
    # decoder = AttnDecoderRNN(hidden_size, target_words_num, max_length=max_length, dropout_p=0.1).to(device)
    decoder = DecoderRNN(hidden_size, target_words_num)
    # optim
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # loss
    criterion = nn.NLLLoss()

    # train
    encoder.train()
    decoder.train()
    for epoch in range(epoches):
        loss_total = 0.0
        for id in tqdm(range(len(train_x))):
            input_tensor = torch.tensor(train_x[id]).unsqueeze(1)
            target_tensor = torch.tensor(train_y[id]).unsqueeze(1)
            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, criterion, max_length=max_length, teacher_forcing_ratio = 0.5)
            loss_total += loss
        print('Epoch: {0}, avg-loss: {1}'.format(epoch+1, loss_total / len(train_x)))

    # test
    encoder.eval()
    decoder.eval()
    

if __name__ == "__main__":
    cur_path = os.path.dirname(__file__)
    data_dir = os.path.join(cur_path, '../data/processing')
    run(data_dir, hidden_size=256, epoches=100, teacher_forcing_ratio = 0.5)