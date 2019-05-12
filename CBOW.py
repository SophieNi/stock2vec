import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import json

def create_cbow_dataset():
    for index, row in df.iterrows():
        if (len(row['Top 20 tickers']) > 2):
            data.append((eval(row['Top 20 tickers']), row['Ticker']))
    return data

class CBOW(nn.Module):
    def __init__(self, vocab_size, embd_size, context_size, hidden_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)
        self.linear1 = nn.Linear(2*context_size*embd_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, inputs):
        embedded = self.embeddings(inputs).view((1, -1))
        hid = F.relu(self.linear1(embedded))
        out = self.linear2(hid)
        log_probs = F.log_softmax(out)
        return log_probs

def train_cbow():
    hidden_size = 64
    losses = []
    loss_fn = nn.NLLLoss()
    model = CBOW(vocab_size, embd_size, CONTEXT_SIZE, hidden_size)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(n_epoch):
        total_loss = .0
        print(epoch)
        for context, target in cbow_train:
            ctx_idxs = [w2i[w] for w in context]

            ctx_var = Variable(torch.LongTensor(ctx_idxs))

            model.zero_grad()
            log_probs = model(ctx_var)

            loss = loss_fn(log_probs, Variable(torch.LongTensor([w2i[target]])))

            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
        losses.append(total_loss)
    return model, losses

def train_model(corpseFilePath):
	torch.manual_seed(1)
	df = pd.read_csv(corpseFilePath)

	df['Top 20 tickers'] = df['Top 20 tickers'].map(lambda x: str(eval(x)[:10]))
	df['Top 20 cor vals'] = df['Top 20 cor vals'].map(lambda x: str(eval(x)[:10]))
	df = df[df['Top 20 tickers'] != "[]"]

	data = []
	names = df.Ticker.unique()
	test = set(names)
	w2i = {w: i for i, w in enumerate(test)}
	i2w = {i: w for i, w in enumerate(test)}
	i2w_names = list(i2w.values())

	cbow_train = create_cbow_dataset()
	embd_size = 100
	learning_rate = 0.001
	n_epoch = 30
	vocab_size = len(names)
	CONTEXT_SIZE = 10

	cbow_model, cbow_losses = train_cbow()

train_model(corpseFilePath)
