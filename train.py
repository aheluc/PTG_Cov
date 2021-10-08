import torch
from trainer import Trainer
from model import Encoder, Decoder
import json
import torch.utils.data as data 
import numpy as np
import os
from gensim.models import word2vec
import sys
from torch.utils.data.dataloader import default_collate

with open('./config.json', encoding='utf-8') as file:
    config = json.load(file)

def str_collate(batch):
    new_batch = []
    ids = []
    for _batch in batch:
        new_batch.append(_batch[:-1])
        ids.append(_batch[-1])
    return default_collate(new_batch), ids

class Dataset(data.Dataset):
    
    def __init__(self, folder, max_content, max_output, word2index):
        # Declare the hyperparameter
        self.folder = folder
        self.files = os.listdir(folder)
        self.max_content = max_content
        self.max_target = max_output
        self.word2index = word2index
        self.pad_index = word2index["<pad>"]
        self.oov_index = word2index["<oov>"]
        self.show = ()

    def __getitem__(self, index):

        # 1.Picked one file
        file = self.files[index]

        # 2.Read the content of file
        input_txt, target_txt = read_content(self.folder + file)
        input_txt = input_txt.replace('\n', '').split(" ") + ["<eos>"]
        target_txt = target_txt.replace('\n', '').split(" ") + ["<eos>"]
        
        content_id, ex_content_id, target_id, ex_target_id, oov = self.get_id(input_txt, target_txt)

        # 4.Padding the data
        input, input_padding_mask = self.padding_mask(content_id, content=True)
        target, target_padding_mask = self.padding_mask(target_id, content=False)
        ex_content_id = self.padding(ex_content_id, content=True)
        ex_target_id = self.padding(ex_target_id, content=False)

        return (input, target, input_padding_mask, target_padding_mask, ex_content_id, ex_target_id, len(oov), oov)
    
    def get_id(self, input_txt, target_txt):
        content_id = []
        target_id = []
        ex_content_id = []
        ex_target_id = []
        oov = []
        for word in input_txt:
            if word in self.word2index:
                content_id.append(self.word2index[word])
                ex_content_id.append(self.word2index[word])
            else:
                content_id.append(self.oov_index)
                if word not in oov:
                    oov.append(word)
                word_id = oov.index(word)
                ex_content_id.append(len(self.word2index) + word_id)
        for word in target_txt:
            if word in self.word2index:
                target_id.append(self.word2index[word])
                ex_target_id.append(self.word2index[word])
            else:
                target_id.append(self.oov_index)
                if word in oov:
                    word_id = oov.index(word)
                    ex_target_id.append(len(self.word2index) + word_id)
                else:
                    ex_target_id.append(self.oov_index)
        return content_id, ex_content_id, target_id, ex_target_id, oov

    def padding_mask(self, data, content):
        # padding to max_content
        if content:
            max_length = self.max_content
        else:
            max_length = self.max_target
        len_data = min([max_length, len(data)])
        mask = np.zeros(max_length)
        mask[:len(data)] = 1
        data_pad = np.pad(data, (0, max(0, max_length - len_data)), 'constant', constant_values=(self.pad_index))[:max_length]
        return data_pad, mask
    
    def padding(self, data, content):
        # padding to max_target
        if content:
            max_length = self.max_content
        else:
            max_length = self.max_target
        len_data = min([max_length, len(data)])
        data_pad = np.pad(data, (0, max(0, max_length - len_data)), 'constant', constant_values=(self.pad_index))[:max_length]
        return data_pad

    def __len__(self):
        return len(self.files)

def read_content(dir_file):
    with open(dir_file, "r", encoding='utf-8') as rb:
        content = rb.readlines()
    return content

word2vec_model = word2vec.Word2Vec.load(config['word2vec_path'] + config['word2vec_name'])
word2idx = word2vec_model.wv.key_to_index.copy()
idx2word = word2vec_model.wv.index_to_key.copy()
print('词表长度为 {}'.format(len(word2idx)))
pre_vocab_size = len(word2idx)
word2idx['<go>'] = pre_vocab_size
word2idx['<oov>'] = pre_vocab_size + 1
word2idx['<eos>'] = pre_vocab_size + 2
word2idx['<pad>'] = pre_vocab_size + 3
max_word_id = pre_vocab_size + 3
vocab_size = len(word2idx) + 4
emb_dim = word2vec_model.wv.vectors.shape[1]

train_dataset = Dataset(config['data_path'] + 'train_data/', config['max_content'], config['max_output'], word2idx)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=config['batch_size'],
                                         shuffle=True,
                                         collate_fn=str_collate)
valid_dataset = Dataset(config['data_path'] + 'valid_data/', config['max_content'], config['max_output'], word2idx)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                         batch_size=config['batch_size'],
                                         shuffle=True,
                                         collate_fn=str_collate)

gpu_id = str(config['gpus'][0])
device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
encoder = Encoder(vocab_size, emb_dim, config['vector_dim'])
decoder = Decoder(vocab_size, emb_dim, config['vector_dim'], config['max_content'])
pretrained = word2vec_model.wv.vectors.copy()
encoder.embedding.from_pretrained(torch.from_numpy(pretrained).to(device), freeze=False)
decoder.embedding.from_pretrained(torch.from_numpy(pretrained).to(device), freeze=False)
encoder = torch.nn.DataParallel(encoder, device_ids=config['gpus']).to(device)
decoder = torch.nn.DataParallel(decoder, device_ids=config['gpus']).to(device)
# encoder.to(device)
# decoder.to(device)
optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=config['learning_rate'])
optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=config['learning_rate'])

print('model building finished.')
trainer = Trainer(config, encoder, decoder, optimizer_encoder,
                optimizer_decoder, train_loader, word2idx, idx2word,
                valid_loader, sys.argv[1])
trainer.train()
