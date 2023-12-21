#!/usr/bin/env python
# coding: utf-8

# ### PyTorch model for text generation
# 
# 
# 

# **Import libraries**

# In[1]:


# Import necessary libraries

import torch
import torch.nn as nn
import numpy as np 
import pandas as pd
import warnings
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)


# **Data preparation**

# In[2]:


# Data reading, preparation

data = pd.read_csv('Shakespeare_data.csv')
data = data.rename(columns={'PlayerLine': 'text'})

data = data['text']
length = len(data)
#print(f"There are {length} sentences in dataset.", '\n')
data.head()

text = list(data) 


# **GPU checking**

# In[3]:


# Device Selection

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("GPU is found and connected!")
elif torch.cuda.is_available():
    device = torch.cuda.device(0)
else:
    device = torch.device("cpu")
    print("No GPU, will do training with CPU")


# **Data preprocessing**

# In[4]:


# Model Hyperparameters

batch_size = 16
seq_size = 32
embedding_size = 64
lstm_size = 64
gradients_norm = 5


# Define function to join all sentences into one long sentence
def joinStrings(text):
    return ' '.join(string for string in text)
text = joinStrings(text)


# Get list of words from document
def doc2words(doc):
    lines = doc.split('\n')
    lines = [line.strip(r'\"') for line in lines]
    words = ' '.join(lines).split()
    return words

# Remove punctuations
def removepunct(words):
    punct = set(string.punctuation)
    words = [''.join([char for char in list(word) if char not in punct]) for word in words]
    return words

# Create a vocabulary where words are ordered by their frequency of occurrence
def getvocab(words):
    wordfreq = Counter(words)
    sorted_wordfreq = sorted(wordfreq, key=wordfreq.get)
    return sorted_wordfreq

# Get dictionary of int to words and word to int
def vocab_map(vocab):
    int_to_vocab = {k:w for k,w in enumerate(vocab)}
    vocab_to_int = {w:k for k,w in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int

# Text Preprocessing
words = removepunct(doc2words(text))
vocab = getvocab(words)
int_to_vocab, vocab_to_int = vocab_map(vocab)


# **Define batches**

# In[5]:


# Function for generating batches 

def get_batches(words, vocab_to_int, batch_size, seq_size):
    # Generate a Xs and Ys of shape (batchsize * num_batches) * seq_size
    word_ints = [vocab_to_int[word] for word in words]
    # Determine Number of Batches
    num_batches = int(len(word_ints) / (batch_size * seq_size))
    # Prepare Input and Target Sequences
    Xs = word_ints[:num_batches*batch_size*seq_size]
    Ys = np.zeros_like(Xs)
    Ys[:-1] = Xs[1:]
    Ys[-1] = Xs[0]
    Xs = np.reshape(Xs, (num_batches*batch_size, seq_size))
    Ys= np.reshape(Ys, (num_batches*batch_size, seq_size))
    
    # Batch Generation
    for i in range(0, num_batches*batch_size, batch_size):
        yield Xs[i:i+batch_size, :], Ys[i:i+batch_size, :]


# **Define RNN model and Loss function**

# In[6]:


# PyTorch module named RNNModule        
        
class RNNModule(nn.Module):
    # initialize RNN module
    def __init__(self, n_vocab, seq_size=32, embedding_size=64, lstm_size=64):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)
        
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state
    
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),torch.zeros(1, batch_size, self.lstm_size))
    
    
# Loss function and the optimization algorithm for training a neural network model     
    
def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer


# **Train Model**
# 
# *Note:* As I already trained the model, now we do not need to train the model again, we just nead to load the saved model and test it. However, if you want to train the model with larger epochs for some reason to get better results please uncomment below cell.

# In[7]:


# # Train an RNN-based text generation model using the specified data and hyperparameters. 
# # The training loop performs backpropagation and updates the model's parameters to minimize the loss. 
    
# def train_rnn(words, vocab_to_int, int_to_vocab, n_vocab):
    
#     # RNN instance
#     net = RNNModule(n_vocab, seq_size, embedding_size, lstm_size)
#     net = net.to(device)
#     criterion, optimizer = get_loss_and_train_op(net, 0.01)

#     iteration = 0
    
#     for e in range(10):
#         batches = get_batches(words, vocab_to_int, batch_size, seq_size)
#         state_h, state_c = net.zero_state(batch_size)

#         # Transfer data to GPU
#         state_h = state_h.to(device)
#         state_c = state_c.to(device)
#         for x, y in batches:
#             iteration += 1

#             # Tell it we are in training mode
#             net.train()

#             # Reset all gradients
#             optimizer.zero_grad()

#             # Transfer data to GPU
#             x = torch.tensor(x).to(device)
#             y = torch.tensor(y).to(device)

#             logits, (state_h, state_c) = net(x, (state_h, state_c))
#             loss = criterion(logits.transpose(1, 2), y)

#             state_h = state_h.detach()
#             state_c = state_c.detach()

#             loss_value = loss.item()

#             # Perform back-propagation
#             loss.backward(retain_graph=True)

#             _ = torch.nn.utils.clip_grad_norm_(net.parameters(), gradients_norm)
            
#             # Update the network's parameters
#             optimizer.step()

#             if iteration % 100 == 0:
#                 print('Epoch: {}/{}'.format(e, 10),'Iteration: {}'.format(iteration),'Loss: {}'.format(loss_value))

#             # if iteration % 1000 == 0:
#                 # predict(device, net, flags.initial_words, n_vocab,vocab_to_int, int_to_vocab, top_k=5)
#                 # torch.save(net.state_dict(),'checkpoint_pt/model-{}.pth'.format(iteration))
                
#     return net

# rnn_net = train_rnn(words, vocab_to_int, int_to_vocab, len(vocab))


# **Generate new Shakespeare-style text!**

# In[8]:


# Generate text of a specified length with a random starting word using a trained neural network model
    
def generate_text(device, net, vocab_to_int, int_to_vocab, length, top_k=5):
    net.eval()

    # Initialize the hidden state
    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)

    # Randomly choose a starting word from the vocabulary
    starting_word = np.random.choice(list(vocab_to_int.keys()))

    # Initialize the words list with the starting word
    words = [starting_word]

    for _ in range(length):
        # Convert the current word to its integer representation
        ix = torch.tensor([[vocab_to_int[words[-1]]]]).to(device)
        
        # Get the output and update the hidden state
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        # Get the top-k choices from the output
        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        
        # Randomly select one of the top-k words
        choice = np.random.choice(choices[0])
        
        # Append the chosen word to the words list
        words.append(int_to_vocab[choice])

    generated_text = ' '.join(words)
    print(generated_text)


# **Load saved Models**

# In[9]:


# To save whole model use below code
# torch.save(rnn_net, 'model.pth')

# To save only the optimized weights of the model use below code
# torch.save(rnn_net.state_dict(), 'model_dict.pth')


# *Note:* This first model was trained by 10 epochs. I saved the whole model in this case. Let us load and test it. To test the model just run the code cells and provide desired text length for the "Enter the desired text length: ". I commented this models output in order to run the seconds model which is trained with longer epochs, but you can uncomment anytime to test also 10 epochs case.

# In[10]:


model = torch.load('model_10epoch.pth')
rnn_net = model
rnn_net


# In[11]:


# User input for text length
# desired_length = int(input("Enter the desired text length: "))

# generate_text(device, rnn_net, vocab_to_int, int_to_vocab, desired_length)


# *Note:* This second model was trained by 50 epochs. I saved the not whole model in this case, but model dict. Let us load and test it. To test the model just run the code cells and provide desired text length for the "Enter the desired text length: ".

# In[12]:


# Define a new instance of RNNModule
loaded_rnn_net = RNNModule(len(vocab), seq_size, embedding_size, lstm_size)
loaded_rnn_net = loaded_rnn_net.to(device)

# Load the saved state dictionary
loaded_rnn_net.load_state_dict(torch.load('model_dict_50epoch.pth'))
loaded_rnn_net.eval()

# User input for text length
desired_length = int(input("Enter the desired text length: "))

# Use a random starting word for prediction
generate_text(device, loaded_rnn_net, vocab_to_int, int_to_vocab, desired_length)


# In[ ]:




