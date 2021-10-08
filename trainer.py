from time import time, strftime, localtime
import random
from torch.nn.utils import clip_grad_norm_
import torch

learning_rate = 0.001

coverage_loss_ratio = 1
eps = 1e-12

class Trainer(object):
    def __init__(self, config, encoder, decoder, optimizer_encoder, 
                optimizer_decoder, train_loader, word2idx, idx2word,
                valid_loader, train_name):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_decoder = optimizer_decoder
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_loss = []
        self.valid_loss = []
        self.accuracy = []
        self.batch_size = config['batch_size']
        self.max_epoch = config['max_epoch']
        self.valid_step = config['valid_step']
        self.report_step = config['report_step']
        self.checkpoint_path = config['checkpoint_path']
        self.max_content = config['max_content']
        self.max_output = config['max_output']
        self.vector_dim = config['vector_dim']
        self.teacher_forcing_ratio = config['teacher_forcing_max_ratio']
        self.teacher_forcing_decay = config['teacher_forcing_decay']
        self.teacher_forcing_decay_step = config['teacher_forcing_decay_step']
        self.teacher_forcing_min_ratio = config['teacher_forcing_min_ratio']
        gpu_id = str(config['gpus'][0])
        self.device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
        self.train_name = train_name
        self.step = 0
        self.encoder.train()
        self.decoder.train()
    
    def train(self):
        for epoch in range(self.max_epoch):
            start_time = time()
            for batch_id, batch in enumerate(self.train_loader):
                self.step += 1
                self._train(batch)
                if self.step % self.report_step == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, elapsed {:.2f} s' 
                        .format(epoch + 1, self.max_epoch, self.step % len(self.train_loader), len(self.train_loader), self.train_loss[-1][1], time() - start_time))
                    start_time = time()
                if self.step % self.valid_step == 0:
                    with torch.no_grad():
                        loss, accuracy = self._validate()
                        if self.valid_loss:
                            best_loss = min(list(zip(*self.valid_loss))[1])
                            best_accuracy = max(list(zip(*self.accuracy))[1])
                            # 如果优于历史，则保存模型
                            if loss < best_loss or accuracy > best_accuracy:
                                print('save checkpoint')
                                self._save(epoch, loss, accuracy)
                        self.valid_loss.append((self.step, loss))
                        self.accuracy.append((self.step, accuracy))
                    start_time = time()
    
    def _train(self, batch):
        self.optimizer_encoder.zero_grad()
        self.optimizer_decoder.zero_grad()
        
        # 准备训练与计算数据
        input, target, mask, target_mask, extended_input, extended_target, ex_size, article_length, oov = self._to_data(batch)
        max_input_len = article_length.max().item()
        if input.size(0) != self.batch_size:
            return
        encoder_out, encoder_hidden = self.encoder(input, article_length)
        decoder_input = torch.tensor([self.word2idx["<go>"]] * self.batch_size, 
                                         dtype=torch.long).view(self.batch_size, -1).to(self.device)
        decoder_hidden = encoder_hidden
        previous_context = torch.zeros((self.batch_size, 1, self.vector_dim * 2), requires_grad=True).to(self.device)
        coverage = torch.zeros((self.batch_size, max_input_len), requires_grad=True).to(self.device)
        step_losses = []
        if self.step % self.report_step == 0:
            check_id = 5
            generated_abstract = []
            mark = ['<go>', '<oov>', '<eos>', '<pad>']
            ex_idx2word = self.idx2word + mark + oov[check_id]
        
        if self.step % self.teacher_forcing_decay_step == 0:
            self.teacher_forcing_ratio = max(self.teacher_forcing_ratio - self.teacher_forcing_decay, self.teacher_forcing_min_ratio)
        
        for output_step in range(self.max_output):
            decoder_output, decoder_hidden, previous_context, attn, coverage, p_gen = self.decoder(decoder_input.to(self.device),
                                                                                       decoder_hidden,
                                                                                       encoder_out.to(self.device),
                                                                                       previous_context.to(self.device), extended_input.to(self.device),
                                                                                       coverage.to(self.device), mask.to(self.device), ex_size,
                                                                                       output_step)
            if self.step % self.report_step == 0:
                _, decoder_input = torch.max(decoder_output, 1)
                generated_word = decoder_input[check_id].item()
                generated_abstract.append(generated_word)
                
            # teacher forcing
            if random.random() <= self.teacher_forcing_ratio:
                # 强行用目标数据 feed
                decoder_input = target[:, output_step].view(self.batch_size, -1)
            else:
                _, decoder_input = torch.max(decoder_output, 1)
                max_word_id = len(self.word2idx) - 1
                decoder_input = torch.where(decoder_input <= max_word_id, decoder_input, self.word2idx['<oov>'])
                decoder_input = decoder_input.view(self.batch_size, -1)
            
            decoder_hidden = decoder_hidden
            step_coverage_loss = torch.sum(torch.min(attn, coverage), 1)
            y = extended_target[:, output_step].view(self.batch_size, 1)
            gold_probs = torch.gather(decoder_output, 1, index=y).squeeze()
            # 最小化gold_probs
            step_loss = (-torch.log(gold_probs + eps)) + coverage_loss_ratio * step_coverage_loss
            # padding部分不计算loss
            step_loss = step_loss * target_mask[:, output_step]
            step_losses.append(step_loss)
            
            coverage = coverage + attn
        if self.step % self.report_step == 0:
            print('reference abstract:', ' '.join([ex_idx2word[id] for id in extended_target[check_id] if self.is_a_word(id)]))
            generated_abstract = ' '.join([ex_idx2word[id] for id in generated_abstract if self.is_a_word(id)])
            print('generated abstract:', generated_abstract)
            
        sum_losses = torch.stack(step_losses, 1).sum(1)
        batch_avg_loss = sum_losses / target_mask.sum(1)
        loss = torch.mean(batch_avg_loss)
        self.train_loss.append((self.step, loss.item()))
        loss.backward()
        
        clip_grad_norm_(self.encoder.parameters(), 2)
        clip_grad_norm_(self.decoder.parameters(), 2)
        
        self.optimizer_encoder.step()
        self.optimizer_decoder.step()

    def _validate(self):
        correct = 0
        total = 0
        losses = []
        check_batch_id = 5
        check_sentence_ids = [2, 11, 17]
        for batch_id, batch in enumerate(self.valid_loader):
            input, target, mask, target_mask, extended_input, extended_target, ex_size, article_length, oov = self._to_data(batch)
            max_input_len = article_length.max().item()
            if input.size(0) != self.batch_size:
                continue
            # Encoder   
            encoder_out, encoder_hidden = self.encoder(input, article_length)
            
            # Decoder 
            # declare the first input <go>
            decoder_input = torch.tensor([self.word2idx["<go>"]] * self.batch_size, 
                                        dtype=torch.long).view(self.batch_size, -1).to(self.device)
            decoder_hidden = encoder_hidden
            previous_context = torch.zeros([self.batch_size, 1, self.vector_dim * 2]).to(self.device)
            coverage = torch.zeros([self.batch_size, max_input_len]).to(self.device)
            step_losses = []
            if batch_id == check_batch_id:
                mark = ['<go>', '<oov>', '<eos>', '<pad>']
                ex_idx2word = [self.idx2word + mark + oov[check_sent_id] for check_sent_id in check_sentence_ids]
                generated_abstract = [[], [], []]
            for output_step in range(self.max_output):
                decoder_output, decoder_hidden, previous_context, attn, coverage, p_gen = self.decoder(decoder_input.to(self.device),
                                                                                  decoder_hidden,
                                                                                  encoder_out.to(self.device),
                                                                                  previous_context.to(self.device),
                                                                                  extended_input.to(self.device), coverage.to(self.device),
                                                                                  mask.to(self.device), ex_size,
                                                                                  output_step)
                
                _, decoder_input = torch.max(decoder_output, 1)
                if batch_id == check_batch_id:
                    for id, check_sent_id in enumerate(check_sentence_ids):
                        generated_word = decoder_input[check_sent_id].item()
                        generated_abstract[id].append(generated_word)
                
                max_word_id = len(self.word2idx) - 1
                decoder_input = torch.where(decoder_input <= max_word_id, decoder_input, self.word2idx['<oov>'])
                decoder_input = decoder_input.view(self.batch_size, -1)
                decoder_hidden = decoder_hidden

                total += target_mask.sum()
                correct += ((torch.max(decoder_output, 1)[1] == extended_input[:, output_step]) * target_mask[:, output_step]).sum().item()
                
                step_coverage_loss = torch.sum(torch.min(attn, coverage), 1)
                
                # 选取每一个正确词汇被选中的概率
                y = extended_target[:, output_step].view(self.batch_size, 1)
                gold_probs = torch.gather(decoder_output, 1, index=y).squeeze()
                # 优化使得每一个正确词汇被选中的概率趋向于1
                step_loss = (-torch.log(gold_probs + eps)) + coverage_loss_ratio * step_coverage_loss
                # padding部分不计算loss
                step_loss = step_loss * target_mask[:, output_step]
                step_losses.append(step_loss)
                coverage = coverage + attn
            
            if batch_id == check_batch_id:
                for id, check_sent_id in enumerate(check_sentence_ids):
                    print('reference abstract:', ' '.join([ex_idx2word[id][wid] for wid in extended_target[check_sent_id] if self.is_a_word(wid)]))
                    print('generated abstract:', ' '.join([ex_idx2word[id][wid] for wid in generated_abstract[id] if self.is_a_word(wid)]))
            sum_losses = torch.stack(step_losses, 1).sum(1)
            batch_avg_loss = sum_losses / target_mask.sum(1)
            loss = torch.mean(batch_avg_loss)
            losses.append(loss.item())
        loss = sum(losses) / len(losses)
        accuracy = 100 * (correct / total)
        print('the loss of the model on the validation data: {:.4f}'.format(loss))
        print('the accuracy of the model on the validation data: {:.3f} %'.format(accuracy.item()))
        return loss, accuracy
    
    def _to_data(self, batch):
        data, oov = batch
        input = data[0].type(torch.LongTensor)
        target = data[1].type(torch.LongTensor).to(self.device)
        mask = data[2].type(torch.LongTensor)
        target_mask = data[3].type(torch.LongTensor).to(self.device)
        extended_input = data[4].type(torch.LongTensor)
        extended_target = data[5].type(torch.LongTensor).to(self.device)
        ex_size = data[6].type(torch.LongTensor).max().item()
        
        article_length = mask.sum(1)

        # 将输入截取至最大长度
        max_input_len = article_length.max().item()
        input = input[:, :max_input_len].to(self.device)
        mask = mask[:, :max_input_len].to(self.device)
        extended_input = extended_input[:, :max_input_len].to(self.device)
        
        mask = mask.to(self.device)
        article_length, index = article_length.sort(descending=True)
        index = index.to(self.device)
        # oov 需要接受 index 的重新排序！！！！
        sorted_oov = [oov[i] for i in index]
        input = input.index_select(0, index)
        target = target.index_select(0, index)
        mask = mask.index_select(0, index)
        extended_input = extended_input.index_select(0, index)
        extended_target = extended_target.index_select(0, index)
        
        return input, target, mask, target_mask, extended_input, extended_target, ex_size, article_length, sorted_oov
    
    def _save(self, epoch, loss, accuracy):
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
        }
        timestamp = strftime('%Y%m%d', localtime())
        checkpoint_name = '{}_{}_epoch_{}_step_{}_loss_{:2f}_accuracy_{:1f}.pth.tar'.format(self.train_name, timestamp, epoch, self.step, loss, accuracy)
        path = self.checkpoint_path + '/' + checkpoint_name
        torch.save(checkpoint, path)
    
    def is_a_word(self, id):
            return id not in (self.word2idx['<eos>'], self.word2idx['<pad>'])
