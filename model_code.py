import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
train_seq_x1 = torch.from_numpy(train_seq_x).to(torch.int64).to(device)
train_y1 = torch.from_numpy(train_y).to(torch.int64).to(device)

valid_seq_x1 = torch.from_numpy(valid_seq_x).to(torch.int64).to(device)
valid_y1 = torch.from_numpy(valid_y).to(torch.int64).to(device)


batch_size = 50
train_loader = DataLoader(TensorDataset(train_seq_x1, train_y1), batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(TensorDataset(valid_seq_x1, valid_y1), batch_size = batch_size, shuffle = True)



class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_num, output_num, layer_num):
      super().__init__()
      self.vocab_size = vocab_size
      self.layer_num = layer_num
      self.hidden_num = hidden_num

      self.embedding = nn.Embedding(vocab_size, embedding_size)
      self.lstm = nn.LSTM(embedding_size, hidden_num, layer_num)
      self.fc = nn.Linear(hidden_num, output_num)
      self.relu = nn.ReLU()

        
    def forward(self, word_seq):
      word_emb = self.embedding(word_seq)
      lstm_out,h = self.lstm(word_emb)
      lstm_out = lstm_out.contiguous().view(-1, self.hidden_num)
      fc_out = self.fc(lstm_out)
      relu_out = self.relu(fc_out)
      relu_out = relu_out[:,-1]
      return relu_out, h



def train(data_loader, classifier, loss_function, optimizer):
    classifier.train()
    loss = 0
    losses = []
    
    accuracy = 0
    accuracies = []
    for i, (texts, labels) in enumerate(data_loader):
        if(texts.shape[0] != batch_size):
            break
        
        texts = texts.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        predictions,h = classifier(texts)

        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())        
    return losses.mean()


n_vocab = len(embedding_matrix)
n_embed = 300
n_hidden = 512
n_output = 1
n_layers = 2

rnn_model = LSTM_Model(n_vocab, n_embed, n_hidden, n_output, n_layers)

loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(rnn_model.parameters(), lr=0.0001, momentum=0.9)
epochs = 5
for epoch in range(0, epochs):
    print("epoch:", epoch + 1)
    train(train_loader, rnn_model, loss_function, optimizer)
    #print("training_loss:", training_loss)

