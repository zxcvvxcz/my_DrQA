# %%

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchtext
import os
import tensorboard
from multiprocessing import freeze_support


# %%
def run():
    # torch.multiprocessing.freeze_support()
    freeze_support()
    from torchtext.experimental.datasets import SQuAD1
    from torchtext.data.utils import get_tokenizer
    # data_dir = '.data'
    # data_names = ['dev-v1.1.json', 'train-v1.1.json']
    # for data_name in data_names:
    #     if not os.path.isfile(os.path.join(data_dir, data_name)):
    #         print('download')
    #         train, dev = SQuAD1()
    #         break
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    # dataset shape: (paragraph, question, answer, span)
    trainset, devset = SQuAD1(tokenizer=tokenizer)

    # %%

    vocab = trainset.get_vocab()

    # %%

    # import re
    # errors = 0
    # print('length of vocab before filtering:', len(vocab.stoi))
    # for key, value in list(vocab.stoi.items()):
    #     if re.search('\n', key) or re.search(' ', key):
    #         errors += 1
    #         print(key)
    #         vocab.stoi.pop(key)
    #         vocab.itos.pop(value)
    #         vocab.freqs.pop(key)
    #         # vocab.freqs[key] -= 1
    #         # if vocab.freqs[key] < 1:
    #         #     vocab.freqs.pop(key)
    #
    # print(errors)
    # print('length of vocab after filtering:', len(vocab.stoi))

    # %%

    # trainset, devset = SQuAD1(vocab=vocab)

    # %%
    def remove_large_text(data):
        return data[0] <= 400

    # %%

    train_data = [(len(paragraph), len(question), idx, paragraph, question, answer, span)
                  for idx, (paragraph, question, answer, span) in enumerate(trainset)]
    dev_data = [(len(paragraph), len(question), idx, paragraph, question, answer, span)
                for idx, (paragraph, question, answer, span) in enumerate(trainset)]

    train_data = list(filter(remove_large_text, train_data))
    dev_data = list(filter(remove_large_text, dev_data))

    train_data.sort()  # sort by length and pad sequences with similar lengths
    dev_data.sort()
    # paragraph, question: tensor of indices of words, use itos to get word

    # Generate the pad id
    pad_id = vocab['<pad>']

    # %%

    # print(train_data[0][3])
    # for idx in train_data[0][3]:
    #     print(train.get_vocab().itos[idx], sep=' ')

    # %%

    def pad_data(data):
        # Find max length of the mini-batch
        # train.get_vocab()['pad'], dev.get_vocab()['pad'] is equal to 22949
        max_p_len = max(list(zip(*data))[0])
        max_q_len = max(list(zip(*data))[1])
        paragraph_list = list(zip(*data))[3]
        question_list = list(zip(*data))[4]
        answer_list = list(zip(*data))[5]
        span_list = list(zip(*data))[6]
        padded_paragraphs = torch.stack([torch.cat((paragraph,
                                                    torch.LongTensor([pad_id] * (max_p_len - len(paragraph))))) \
                                         for paragraph in paragraph_list])
        padded_questions = torch.stack([torch.cat((question,
                                                   torch.tensor([pad_id] * (max_q_len - len(question))).long())) \
                                        for question in question_list])
        paragraph_pad_mask = torch.zeros_like(padded_paragraphs).masked_fill(padded_paragraphs == pad_id, 1)
        question_pad_mask = torch.zeros_like(padded_questions).masked_fill(padded_questions == pad_id, 1)

        return padded_paragraphs, padded_questions, span_list, answer_list, \
               paragraph_pad_mask, question_pad_mask

    # %%

    BATCH_SIZE = 32
    from torch.utils.data import DataLoader
    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_data, num_workers=4)
    testloader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_data, num_workers=4)

    # %%

    # for i, (p, q, a, s) in enumerate(train):
    #     print(p,q,a,s)
    #     nps = s[0].numpy()
    #     tokens = tokenizer(train.data[i][0])
    #     print(tokens[int(nps[0])])
    #     print(tokens[nps[1]])
    #     if i > 5:
    #         break

    # %%

    # for idx, (padded_paragraphs, padded_questions, span_list, answer_list,
    #            paragraph_pad_mask, question_pad_mask) in enumerate(trainloader):
    #     print(idx, padded_paragraphs, padded_questions, span_list, answer_list,
    #            paragraph_pad_mask, question_pad_mask)
    #     print(span_list)
    #     if idx > 0:
    #         break

    # %%

    # print(trainset.get_vocab()['pad'], dev.get_vocab()['pad'])

    # %%

    glove_vec = torchtext.vocab.GloVe(name='840B', dim=300)

    # %%

    def build_word_embedding(vocab, pre_trained_emb_vec):
        # print(pre_trained_emb_vec.dim)
        weights_matrix = np.zeros((len(vocab), pre_trained_emb_vec.dim))
        words_found = 0
        no_word = 0
        for i, (word, _) in enumerate(vocab.freqs.most_common()):
            try:
                word_index = pre_trained_emb_vec.stoi[word]
                weights_matrix[i] = pre_trained_emb_vec[word_index]
                words_found += 1
            except:
                no_word += 1  # no such word in pre_trained_embedding: zero vector
        print('words not found:', no_word)
        print('words found:', words_found)
        return torch.FloatTensor(weights_matrix)

    # %%

    # for key, value in vocab.freqs.items():
    #     if re.search(' ', key):
    #         print(key, value)
    # for i, word in enumerate(vocab.freqs.most_common()):
    #     print(word)
    #     if i > 5:
    #         break

    # %%

    word_emb_table = build_word_embedding(vocab, glove_vec)

    # %%

    # glove_vec.vectors[:5]

    # %%

    import spacy
    nlp = spacy.load('en_core_web_sm')

    def exact_match(paragraphs_indices, questions_indices, vocab):
        # process one paragraph batch, one question batch
        # print(paragraphs_indices.size())
        # print(questions_indices.size())
        #
        # j = 0
        # for (paragraph_indices, question_indices) in \
        #         zip(paragraphs_indices, questions_indices):
        #     j += 1
        # print('j:',j)
        exact_match_table = np.zeros((len(paragraphs_indices), len(paragraphs_indices[0]), 3))
        # print(exact_match_table.shape)

        for i, (paragraph_indices, question_indices) in \
                enumerate(zip(paragraphs_indices, questions_indices)):
            # print(paragraphs_indices)
            # print(paragraphs_indices.size())
            # paragraph_processed = nlp(paragraph_sentence)
            # question_lemmas = [lem.lemma_ for lem in question_processed]
            for j, paragraph_index in enumerate(paragraph_indices):
                paragraph_word = vocab.itos[paragraph_index]
                if paragraph_word == '<pad>':
                    # print('got pad')
                    continue
                em_tensor = torch.LongTensor([0, 0, 0])
                # original
                if paragraph_index in question_indices:
                    em_tensor[0] = 1
                # lemma
                if vocab.stoi[nlp(paragraph_word)[0].lemma_] in question_indices:
                    em_tensor[1] = 1
                # uncased
                if vocab.stoi[paragraph_word.lower()] and \
                        vocab.stoi[paragraph_word.lower()] in question_indices:
                    em_tensor[2] = 1
                exact_match_table[i][j] = em_tensor

        return torch.LongTensor(exact_match_table)

    # %%

    class AlignedQuestionEmbedding(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.relu = nn.ReLU()
            self.linear = nn.Linear(input_dim, input_dim)

        def forward(self, paragraph, question, question_pad_mask):
            p = self.linear(paragraph)
            p = self.relu(p)

            q = self.relu(self.linear(question))
            q = question.permute(0, 2, 1)

            dot_product = torch.bmm(p, q)
            # print(dot_product.size())
            # print(question_pad_mask.size())
            question_mask_expand = question_pad_mask.unsqueeze(1).expand(dot_product.size())
            dot_product = dot_product.masked_fill(question_mask_expand == 1, -np.inf)

            dot_product_flatten = dot_product.view(-1, question.size(1))

            attn_score = F.softmax(dot_product_flatten, dim=1)
            attn_score = attn_score.view(-1, paragraph.shape[1], question.shape[1])

            aligned_embedding = torch.bmm(attn_score, question)
            return aligned_embedding

    # %%

    P_ENCODING_NUM = 2

    class MultiLayerBiLSTM(nn.Module):

        def __init__(self, input_size, hidden_size, nlayers, dropout):
            super().__init__()
            self.nlayers = nlayers

            self.lstms = nn.ModuleList()
            self.dropout = nn.Dropout(p=dropout)
            self.lstms.append(nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True))
            for i in range(1, nlayers):
                self.lstms.append(nn.LSTM(hidden_size * 2, hidden_size,
                                          batch_first=True, bidirectional=True))

        def forward(self, x):
            # x = self.dropout(x)
            lstm_output, (hidden_state, cell_state) = self.lstms[0](x)
            hidden_states = [lstm_output]
            # print(lstm_output.size(), hidden_state.size(), cell_state.size())
            for i in range(1, self.nlayers):
                lstm_output = self.dropout(lstm_output)
                lstm_output, (hidden_state, cell_state) = self.lstms[i](lstm_output)
                # print(lstm_output.size(), hidden_state.size(), cell_state.size())
                hidden_states.append(lstm_output)

            output = torch.cat(hidden_states, dim=2)

            output = self.dropout(output)
            return output

    # %%

    class QuestionEncoding(nn.Module):
        def __init__(self, input_size, hidden_size, nlayers, dropout):
            super().__init__()
            self.input_size = input_size
            self.linear = nn.Linear(input_size, 1)
            self.lstm = MultiLayerBiLSTM(input_size, hidden_size, nlayers, dropout)

        def forward(self, x, question_mask):
            x_lstm = self.lstm(x)
            x = x.view(-1, self.input_size)
            x = self.linear(x)  # attention score
            x = x.view(question_mask.shape[0], -1)
            # print(x.size(), question_mask.size())
            x = x.masked_fill(question_mask == 1, -np.inf)  # masking
            x = F.softmax(x, dim=1)

            x = x.unsqueeze(1)
            # print(x.size(), x_lstm.size())
            encoding = torch.bmm(x, x_lstm)
            encoding = encoding.squeeze(1)
            return encoding

    # %%

    class PredictionLayer(nn.Module):
        def __init__(self, p_size, q_size):
            super().__init__()
            self.linear = nn.Linear(q_size, p_size)

        def forward(self, paragraph, question, paragraph_mask):
            Wq = self.linear(question)
            Wq = Wq.unsqueeze(2)
            pWq = paragraph.bmm(Wq)
            pWq = pWq.squeeze(2)
            pWq = pWq.masked_fill(paragraph_mask == 1, -np.inf)
            return pWq

    # %%

    def fix_embedding(grad):
        grad[1000:] = 0
        return grad

    class DocumentReader(nn.Module):
        def __init__(self, hidden_size, embedding_size, nlayers, dropout, device):
            super().__init__()
            self.device = device

            self.word_embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(word_emb_table).to(device),
                                                                     freeze=False)
            self.word_embedding_layer.weight.register_hook(fix_embedding)
            # print(embedding_size)
            self.aligned_embedding_layer = AlignedQuestionEmbedding(embedding_size)
            self.paragraph_lstm = MultiLayerBiLSTM(embedding_size * 2 + 3, hidden_size, nlayers, dropout)
            # self.paragraph_lstm = MultiLayerBiLSTM(embedding_size * 2, hidden_size, nlayers, dropout)

            self.question_encoder = QuestionEncoding(embedding_size, hidden_size, nlayers, dropout)

            self.prediction_layer_start = PredictionLayer(hidden_size * nlayers * 2,
                                                          hidden_size * nlayers * 2)
            self.prediction_layer_end = PredictionLayer(hidden_size * nlayers * 2,
                                                        hidden_size * nlayers * 2)

            self.dropout = nn.Dropout(dropout)

        def forward(self, paragraph, question, paragraph_mask, question_mask):
            em_embedding = exact_match(paragraph, question, vocab)
            # print(em_embedding.size())
            p_word_embedding = self.word_embedding_layer(paragraph)
            q_word_embedding = self.word_embedding_layer(question)
            p_word_embedding = self.dropout(p_word_embedding)
            q_word_embedding = self.dropout(q_word_embedding)
            aligned_embedding = self.aligned_embedding_layer(p_word_embedding, q_word_embedding, question_mask)
            # print(p_word_embedding.size())
            # print(aligned_embedding.size())

            paragraph_embeddings = torch.cat(
                [em_embedding.to(device), p_word_embedding.to(device), aligned_embedding.to(device)], dim=2)
            paragraph_encoding = self.paragraph_lstm(paragraph_embeddings)
            # print(question.size(), question_mask.size())
            question_encoding = self.question_encoder(q_word_embedding, question_mask)

            prediction_start = self.prediction_layer_start(paragraph_encoding, question_encoding, paragraph_mask)
            prediction_end = self.prediction_layer_end(paragraph_encoding, question_encoding, paragraph_mask)

            return prediction_start, prediction_end

    # %%

    HIDDEN_SIZE = 128
    EMB_SIZE = 300
    NLAYERS = 3
    DROPOUT = 0.3
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DocumentReader(HIDDEN_SIZE,
                           EMB_SIZE,
                           NLAYERS,
                           DROPOUT,
                           device).to(device)

    # %%

    optimizer = torch.optim.Adamax(model.parameters())

    # %%

    def train(model, train_dataset):
        '''
        Trains the model.
        '''

        print("Start training ........")

        train_loss = 0.

        # put the model in training mode
        model.train()

        # iterate through training data
        for i, (paragraphs, questions, span_list, answer_list,
                paragraph_mask, question_mask) in enumerate(train_dataset):

            if i % 500 == 0:
                print(f"Starting batch: {i}")

            # place the tensors on GPU
            paragraphs = paragraphs.to(device)
            paragraph_mask = paragraph_mask.to(device)
            questions = questions.to(device)
            question_mask = question_mask.to(device)
            # span_list = span_list.to(device)

            # forward pass, get the predictions
            preds = model(paragraphs, questions, paragraph_mask, question_mask)

            start_pred, end_pred = preds
            # print('preds:', start_pred, end_pred)
            # separate labels for start and end position
            span_start = []
            span_end = []
            for span in span_list:
                span_start.append(span[0][0])
                span_end.append(span[0][1])
            # print('span:', span_start, span_end)
            span_start = torch.LongTensor(span_start).to(device)
            span_end = torch.LongTensor(span_end).to(device)
            # calculate loss
            loss = F.cross_entropy(start_pred, span_start) + F.cross_entropy(end_pred, span_end)

            # backward pass, calculates the gradients
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            # update the gradients
            optimizer.step()

            # zero the gradients to prevent them from accumulating
            optimizer.zero_grad()

            train_loss += loss.item()

        return train_loss / len(train_dataset)

    # %%

    train_loss = train(model, trainloader)


if __name__ == '__main__':
    run()
