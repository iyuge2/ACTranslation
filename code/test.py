# import os
# import torch
# from torch.autograd import Variable
# import torch.nn as nn

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

# class Model(nn.Module):
#     def __init__(self, mode, embedding_size):
#         super(Model, self).__init__()
#         self.mode = mode
#         self.embedding_size = embedding_size
#         self.nb_layers = 1
#         self.dropout = 0
#         self.batch_size = 1

#         if self.mode == 'GRU':
#             self.document_rnn = nn.GRU(embedding_size, embedding_size, num_layers=self.nb_layers, bias=True, dropout=self.dropout, bidirectional=False, batch_first=True)
#         elif self.mode == 'LSTM':
#             self.document_rnn = nn.LSTM(embedding_size, embedding_size, num_layers=self.nb_layers, bias=True, dropout=self.dropout, bidirectional=False, batch_first=True)
#         self.document_rnn_hidden = self.init_hidden()

#         self.init_hidden()

#     def init_hidden(self):
#         document_rnn_init_h = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(self.nb_layers, self.batch_size, self.embedding_size).type(torch.FloatTensor).cuda()), requires_grad=True)
#         if self.mode == 'GRU':
#             return document_rnn_init_h
#         elif self.mode == 'LSTM':
#             document_rnn_init_c = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(self.nb_layers, self.batch_size, self.embedding_size).type(torch.FloatTensor).cuda()), requires_grad=True)
#             return (document_rnn_init_h, document_rnn_init_c)

#     def forward(self, sentence_hidden_embeddings, nb_sentences_per_doc):
#         all_sentence_embeddings_per_doc = torch.split(sentence_hidden_embeddings.unsqueeze(0), nb_sentences_per_doc, dim=1)[:-1]

#         document_embeddings = []
#         for sentence_embeddings_per_doc in all_sentence_embeddings_per_doc:
#             self.document_rnn_hidden = self.init_hidden()
#             output, hidden = self.document_rnn(sentence_embeddings_per_doc, self.document_rnn_hidden)

#             # output[-1][-1] == hidden[-1][-1] (GRU) and output[-1][-1] == hidden[0][-1][-1] (LSTM)
#             doc_emb = hidden[-1] if self.mode == 'GRU' else (hidden[0][-1] if self.mode == 'LSTM' else None)
#             document_embeddings.append(doc_emb)

#             # TODO Remove. Doing only this perfectly works on GPU
#             #doc_emb = torch.mean(sentence_embeddings_per_doc, dim=1)
#             #document_embeddings.append(doc_emb)
#         cluster_embedding = torch.mean(torch.cat(document_embeddings), dim=0)

#         return document_embeddings, cluster_embedding

# sentence_hidden_embeddings = Variable(torch.randn(657, 700).cuda())
# nb_sentences_per_doc = [26, 13, 12, 20, 25, 26, 535]

# model = Model('LSTM', 700)
# model = model.cuda()
# model(sentence_hidden_embeddings, nb_sentences_per_doc)

import torch
rnn = torch.nn.LSTM(10,10)  # same error with e.g. torch.nn.GRU(10,10,1)
rnn.cuda()