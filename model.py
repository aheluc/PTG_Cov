import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

rand_unif_init_mag = 0.02

import json
with open('./config.json', encoding='utf-8') as file:
    config = json.load(file)

gpu_id = str(config['gpus'][0])
device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")

def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

class Encoder(torch.nn.Module):

    def __init__(self, vocab_size, emb_size, vector_dim):
        super(Encoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, 
                                            embedding_dim=emb_size)
        self.embedding.weight.data.normal_(std=1e-4)
        self.lstm = torch.nn.LSTM(input_size=emb_size,
                                hidden_size=vector_dim,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)
        init_lstm_wt(self.lstm)

    def forward(self, input, article_length):
        # Embedding
        embedding = self.embedding(input)
        embedding = pack_padded_sequence(embedding, article_length.cpu(), batch_first=True)
        out, hidden = self.lstm(embedding)
        out, _ = pad_packed_sequence(out, batch_first=True)  # [batch, max(seq_lens), 2*hid_dim]
        out = out.contiguous()
        # [句子数, 句长, vector_dim * 2(因为是双向LSTM)]
        return out, hidden

class AdditiveAttention(nn.Module):
    """ 
    使用加性注意力
    """

    def __init__(self, vector_dim, input_length):
        super(AdditiveAttention, self).__init__()
        # 向量大小
        self.vector_dim = vector_dim
        # 句子的最大长度
        self.input_length = input_length
        
        self.W_h = nn.Linear(vector_dim * 2, vector_dim * 2, bias=False)
        
        self.W_s = nn.Linear(vector_dim * 2, vector_dim * 2)
        
        self.W_x = nn.Linear(1, vector_dim * 2, bias=False)
        
        self.v = nn.Linear(vector_dim * 2, 1, bias=False)

    def forward(self, state, context, coverage, padding_mask):
        # 明确输入大小
        # 句子数
        batch_size, max_input_len, double_vector_dim = context.size()
        
        hidden_state, cell_state = state
        state = torch.cat((hidden_state.view(-1, self.vector_dim),
                             cell_state.view(-1, self.vector_dim)), 1)
        
        state_feature = self.W_s(state)
        state_feature_expanded = state_feature.unsqueeze(1).expand(batch_size, max_input_len, double_vector_dim).contiguous()
        state_feature_expanded = state_feature_expanded.view(-1, double_vector_dim)
        
        encoder_feature = context.view(-1, double_vector_dim) # encoder_outputs
        encoder_feature = self.W_h(encoder_feature)
        
        coverage_input = coverage.view(-1, 1)
        coverage_feature = self.W_x(coverage_input)
        
        # e = vTtanh(WhH + WsS + WxX + b)
        attention_feature = encoder_feature + state_feature_expanded + coverage_feature
        e = torch.tanh(attention_feature)
        score = self.v(e)
        score = score.view(-1, max_input_len)
        
        attention = F.softmax(score, dim=1)
        
        # 将padding上去的部分注意力置为0
        masked_attention = attention * padding_mask
        #重新归一化
        normalization_factor = masked_attention.sum(dim=1, keepdim=True)
        normed_masked_attention = masked_attention / normalization_factor
        
        # 每一步的hidden state乘以自己的注意力权重
        attention = normed_masked_attention.reshape(batch_size, -1, max_input_len)
        #context = context.reshape(batch_size, max_input_len, self.vector_dim * 2)
        # 注意力加权上下文向量
        # weighted_context shape: [句子数, 1, 隐层维度 * 2]
        weighted_context = torch.bmm(attention, context)
        
        # weighted_context为 [句子数, 1, 隐层维度 * 2]
        # attention为 [句子数, 1, 最大句长]
        return weighted_context, normed_masked_attention, coverage, score

class ScaledDotProductAttention(nn.Module):
    """ modified attention based on self-attention
    注意力的计算方式根据scaled dot-product attention的计算方式进行了修改
    """

    def __init__(self, vector_dim, input_length):
        super(ScaledDotProductAttention, self).__init__()
        # 向量大小
        self.vector_dim = vector_dim
        # 句子的最大长度
        self.input_length = input_length
        # 对输出产生一个向量
        self.w_q = nn.Linear(vector_dim * 2, vector_dim)
        # 对每个词的coverrage输出产生一个向量
        self.coverage_proj = nn.Linear(1, vector_dim * 2)
        # 对encoder每一步的输出产生一个向量
        self.w_k = nn.Linear(vector_dim * 2, vector_dim)
        # 缩小乘出来的注意力数值，防止点积过大，梯度消失
        self.scale = math.sqrt(float(vector_dim))

    def forward(self, state, context, coverage, padding_mask):
        # 明确输入大小
        # 句子数
        batch_size = context.size(0)
        hidden_state, cell_state = state
        query = torch.cat((hidden_state.view(-1, self.vector_dim),
                             cell_state.view(-1, self.vector_dim)), 1)
        
        # 得到关于当前输出的Q矩阵
        # Q shape: [句子数, 向量大小]
        Q = torch.tanh(self.w_q(query))
        
        # 根据encoder每一步的输出和coverage得到K矩阵
        word_feature = context + self.coverage_proj(coverage.unsqueeze(2))
        # K shape: [句子数, 句子的最大长度, 向量大小]
        K = torch.tanh(self.w_k(word_feature))
        
        # 改变形状以进行乘法broadcast
        # Q shape: [句子数, 1, 向量大小]
        Q = Q.reshape(batch_size, 1, self.vector_dim) / self.scale
        # QK点乘获得注意力权重
        score = (Q * K).sum(dim=2)
        # 将 padding 上去的部分 score 置为充分小的 -1e18
        #print(score.shape, padding_mask.shape)
        max_input_len = score.shape[1]
        score = score.masked_fill((1 - padding_mask).type(torch.BoolTensor).to(device), -1e18)
        # 用softmax归一化
        attention = F.softmax(score, dim=1)
        
        # 将padding上去的部分注意力置为0
        #masked_attention = attention * padding_mask
        #重新归一化
        #masked_sum = masked_attention.sum(dim=1).view(-1, 1)
        #normed_masked_attention = masked_attention / masked_sum
        # 更新coverage
        #coverage = coverage + attn
        
        # 每一步的hidden state乘以自己的注意力权重
        attention_left = attention.reshape(batch_size, -1, max_input_len)
        context_right = context.reshape(batch_size, max_input_len, self.vector_dim * 2)
        # 注意力加权上下文向量
        # weighted_context shape: [句子数, 1, 隐层维度 * 2]
        weighted_context = torch.bmm(attention_left, context_right)
        
        # weighted_context为 [句子数, 1, 隐层维度 * 2]
        # attention为 [句子数, 1, 最大句长]
        return weighted_context, attention, coverage, score

class Decoder(torch.nn.Module):
    
    def __init__(self, vocab_size, emb_size, vector_dim, input_length, use_additive_attention=False):
        super(Decoder, self).__init__()
        # 句子的最大长度
        self.input_length = input_length
        # 隐层维度
        self.vector_dim = vector_dim
        # 词汇量
        self.vocab_size = vocab_size

        # Embedding
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, 
                                            embedding_dim=emb_size)
        self.embedding.weight.data.normal_(std=1e-4)
        # 用来产生LSTM的输入
        self.linear = torch.nn.Linear(emb_size + vector_dim * 2, vector_dim)
        self.lstm = torch.nn.LSTM(input_size=vector_dim,
                                hidden_size=vector_dim,
                                num_layers=1,
                                bidirectional=False,
                                batch_first=True)
        init_lstm_wt(self.lstm)
        if use_additive_attention:
            self.attention = AdditiveAttention(vector_dim, input_length)
        else:
            self.attention = ScaledDotProductAttention(vector_dim, input_length)
        
        self.reduce_hidden_state = nn.Linear(vector_dim * 2, vector_dim)
        self.reduce_cell_state = nn.Linear(vector_dim * 2, vector_dim)
        
        self.pregenerate = nn.Linear(vector_dim * 3, vector_dim)
        # 生成模式下是产生某词的概率
        self.generate = torch.nn.Linear(vector_dim, vocab_size)
        # 生成的概率
        self.get_p_gen = torch.nn.Linear(vector_dim * 5, 1)
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, input, hidden, encoder_output, previous_context, extended_input, coverage, padding_mask, ex_size, output_step):
        batch_size = encoder_output.size(0)
        max_input_len = coverage.shape[1]
        # Embedding
        embedding = self.embedding(input)
        #拼接输入和上一轮的context_vector
        combine = torch.cat([embedding, previous_context], 2)
        x = self.linear(combine)
        
        # 给encoder的双向LSTM传来的state降维
        if output_step == 0:
            hidden_state, cell_state = hidden
            hidden_state = hidden_state.transpose(0, 1).contiguous().view(-1, self.vector_dim * 2)
            cell_state = cell_state.transpose(0, 1).contiguous().view(-1, self.vector_dim * 2)
            reduced_hidden_state = F.relu(self.reduce_hidden_state(hidden_state))
            reduced_cell_state = F.relu(self.reduce_cell_state(cell_state))
            hidden = (reduced_hidden_state.unsqueeze(0), reduced_cell_state.unsqueeze(0))
        
        # Call the LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 计算注意力
        weighted_context, attention, coverage, score = self.attention(state=hidden, context=encoder_output, coverage=coverage, padding_mask=padding_mask)
        
        # 计算生成的概率
        hidden_state, cell_state = hidden
        p_gen = self.sigmoid(self.get_p_gen(torch.cat([x.squeeze(), weighted_context.squeeze(), hidden_state.squeeze(), cell_state.squeeze()], 1)))
        
        # 生成模式下，输出生成一个词的概率
        output_with_context = torch.cat((weighted_context, lstm_out), dim=2)
        # 这个output是用来生成vocab_dist的一个输入
        pregenerate = self.pregenerate(output_with_context.view(-1, 3 * self.vector_dim)).view(batch_size, -1)
        # [句数, 词汇总量]
        vocab_dist = F.softmax(self.generate(pregenerate), dim=1)
        
        # 计算生成的概率和拷贝的概率
        # 生成概率 * 生成的词的概率
        vocab_dist = p_gen * vocab_dist
        # 拷贝概率 * 原文中每个词的注意力
        attention_dist = (1 - p_gen) * attention
        
        # 为生成词汇的概率扩展词汇
        ex_vocab = torch.zeros([batch_size, ex_size], requires_grad=True).to(device)
        vocab_dist = torch.cat([vocab_dist, ex_vocab], dim=1).to(device)
        
        final_dist = vocab_dist.scatter_add(1, index=extended_input, src=attention_dist)
        
        return final_dist, hidden, weighted_context, attention, coverage, p_gen
