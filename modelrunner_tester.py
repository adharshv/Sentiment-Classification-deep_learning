#!/usr/bin/env python
# coding: utf-8

# In[1]:


from modelrunner import*


# In[2]:


res1 = read_file("movie_reviews.tar")


# In[3]:


text = res1[0]


# In[4]:


vocab = preprocess(text)


# In[5]:


encoded_reviews = encode_review(vocab, text)


# In[6]:


labels = res1[1]


# In[7]:


encoded_labels = encode_labels(labels)


# In[8]:


encoded_reviews = pad_zeros(encoded_reviews, seq_length = 200)


# In[9]:


embedding_file = "D:\\Third sem\\Natural Language Processing by Dan Moldovan\\Projects\\Project 1\\project\\project\\others\\wiki-news-300d-1M.vec\\wiki-news-300d-1M.vec"


# In[10]:


embed_dict = load_embedding_file(embedding_file, vocab)


# In[11]:


# default: batch-size = 32, learning rate = 0.01, n_epochs = 30
# embed_dictarray = np.fromiter(embed_dict.items(), dtype = None, count=len(embed_dict))


# In[12]:


# dataset = create_data_loader(encoded_reviews, encoded_labels, batch_size = 16)
dataset = create_data_loader(encoded_reviews, encoded_labels, batch_size = 32)
# dataset = create_data_loader(encoded_reviews, encoded_labels, batch_size = 50)

# print(len(dataset[0]) # gives 50 (50 batches of 32 each)
# print(len(dataset[0].dataset)) # gives 1600

# print(len(dataset[1])) # gives 7
# print(len(dataset[1].dataset)# gives 200


# In[13]:


import torch
# from torch import optim


# In[14]:


basemodel = BaseSentiment() # instantiating a BaseSentiment model
basemodel = basemodel.cuda() # .cuda() => use gpu
# print(basemodel)


# In[15]:


# try changing optimizer, learning rate, n_epochs, batch-size etc..
trainval(model = basemodel, trainloader = dataset[0], validationloader = dataset[1], n_epochs = 30,  optimizer = optim.SGD(basemodel.parameters(), lr=0.01), loss_fn = nn.BCELoss())
# trainval(model = basemodel, trainloader = dataset[0], validationloader = dataset[1], n_epochs = 30,  optimizer = optim.SGD(basemodel.parameters(), lr=0.01), loss_fn = nn.BCELoss())
# learning rate = 0.01
# batch_size = 16:
# Training accuracy: 0.624, Validation accuracy: 0.510, n_epochs = 30
# batch_size = 32:
# Training accuracy: 0.609, Validation accuracy: 0.600, n_epochs = 30
# batch_size = 50:
# Training accuracy: 0.601, Validation accuracy: 0.450, n_epochs = 30
# learning rate = 0.05: # batch_size = 32:
# Training accuracy: 0.614, Validation accuracy: 0.480, n_epochs = 30
# learning rate = 0.03:
# Training accuracy: 0.624, Validation accuracy: 0.555, n_epochs = 30
# n_epochs = 60:
# Training accuracy: 0.635, Validation accuracy: 0.535
# n_epochs = 90:
# Training accuracy: 0.657, Validation accuracy: 0.500


# In[16]:


test(model = basemodel, testloader = dataset[2]) # no need for loss function since we are using accuracy 
# learning rate = 0.01:
# batch_size = 16:
# Testing accuracy: 0.550, n_epochs = 30
# batch_size = 32:
# Testing accuracy: 0.505, n_epochs in training = 30
# batch_size = 50:
# Testing accuracy: 0.485, n_epochs in training = 30
# learning rate = 0.05: # batch_size = 32:
# Testing accuracy: 0.545, n_epochs in training = 30
# learning rate = 0.03:
# Testing accuracy: 0.565
# n_epochs = 60:
# Testing accuracy: 0.445
# n_epochs = 90:
# Testing accuracy: 0.465


# In[17]:


# type = vanilla_rnn, gru or lstm
# try hidden_dim = 512
rnnmodel = RNNSentiment(rnn_type = "vanilla_rnn", hidden_dim = 512, layer_dim = 1, bidirec = False)
# rnnmodel = RNNSentiment(rnn_type = "vanilla_rnn", hidden_dim = 512, layer_dim = 2, bidirec = False)
# rnnmodel = RNNSentiment(rnn_type = "vanilla_rnn", hidden_dim = 512, layer_dim = 1, bidirec = True) # instantiating a RNNSentiment model
rnnmodel = rnnmodel.cuda()
#print(rnnmodel)


# In[18]:


trainval(model = rnnmodel, trainloader = dataset[0], validationloader = dataset[1], n_epochs = 30, optimizer = optim.SGD(rnnmodel.parameters(), lr=0.01), loss_fn = nn.BCELoss())
# trainval(model = rnnmodel, trainloader = dataset[0], validationloader = dataset[1], n_epochs = 30, optimizer = optim.SGD(rnnmodel.parameters(), lr=0.03), loss_fn = nn.BCELoss())

# learning rate = 0.01:
# batch_size = 16:
# Training accuracy: 0.816, Validation accuracy: 0.550, n_epochs = 30
# batch_size = 32:
# Training accuracy: 0.734, Validation accuracy: 0.575, n_epochs = 30
# try with dropout in the definition of rnn layer in modelrunner.py to reduce overfitting(layer_dim needs to be > 1)
# bidirectional: Training accuracy: 0.616, Validation accuracy: 0.545, n_epochs = 30
# batch_size = 50:
# Training accuracy: 0.666, Validation accuracy: 0.505, n_epochs = 30
# learning rate = 0.05:# batch_size = 32:
# Training accuracy: 0.865, Validation accuracy: 0.560, n_epochs = 30
# learning rate = 0.03:
# Training accuracy: 0.862, Validation accuracy: 0.510
# n_epochs = 60:
# Training accuracy: 0.876, Validation accuracy: 0.530
# n_epochs = 90:
# Training accuracy: 0.901, Validation accuracy: 0.530
# no of layers = 2: Training accuracy: 0.821, Validation accuracy: 0.475


# In[19]:


test(model = rnnmodel, testloader = dataset[2]) 
# learning rate = 0.01:
# batch_size = 16:
# Testing accuracy: 0.565, n_epochs = 30
# batch_size = 32:
# Testing accuracy: 0.515, n_epochs in training = 30
# overfitting without dropout, improvement with dropout (layer_dim needs to be > 1)
# bidirectional: Testing accuracy: 0.490, n_epochs = 30
# batch_size = 50:
# Testing accuracy: 0.485, n_epochs in training = 30
# learning rate = 0.05:# batch_size = 32:
# Testing accuracy: 0.510, n_epochs in training = 30
# learning rate = 0.03:
# Testing accuracy: 0.555
# n_epochs = 60:
# Testing accuracy: 0.475
# n_epochs = 90:
# Testing accuracy: 0.445
# no of layers = 2: Testing accuracy: 0.525


# In[20]:


# type = vanilla_rnn, gru or lstm
grumodel = RNNSentiment(rnn_type = "gru", hidden_dim = 512, layer_dim = 1, bidirec = False)
# grumodel = RNNSentiment(rnn_type = "gru", hidden_dim = 512, layer_dim = 2, bidirec = False)
# grumodel = RNNSentiment(rnn_type = "gru", hidden_dim = 512, layer_dim = 1, bidirec = True) # instantiating a RNNSentiment model
grumodel = grumodel.cuda()


# In[21]:


trainval(model = grumodel, trainloader = dataset[0], validationloader = dataset[1], n_epochs = 30, optimizer = optim.SGD(grumodel.parameters(), lr=0.01), loss_fn = nn.BCELoss())
# trainval(model = grumodel, trainloader = dataset[0], validationloader = dataset[1], n_epochs = 30, optimizer = optim.SGD(grumodel.parameters(), lr=0.03), loss_fn = nn.BCELoss())
# learning rate = 0.01:
# batch_size = 16:
# Training accuracy: 0.642, Validation accuracy: 0.555, n_epochs = 30
# batch_size = 32:
# Training accuracy: 0.614, Validation accuracy: 0.455, n_epochs = 30
# bidirectional: Training accuracy: 0.584, Validation accuracy: 0.545, n_epochs = 30
# batch_size = 50:
# Training accuracy: 0.591, Validation accuracy: 0.510, n_epochs = 30
# learning rate = 0.05:# batch_size = 32:
# Training accuracy: 0.752, Validation accuracy: 0.525, n_epochs = 30
# learning rate = 0.03:
# Training accuracy: 0.691, Validation accuracy: 0.525
# n_epochs = 60:
# Training accuracy: 0.683, Validation accuracy: 0.495
# n_epochs = 90:
# Training accuracy: 0.702, Validation accuracy: 0.520 
# no of layers = 2: Training accuracy: 0.551, Validation accuracy: 0.520


# In[22]:


test(model = grumodel, testloader = dataset[2])
# learning rate = 0.01:
# batch_size = 16:
# Testing accuracy: 0.565, n_epochs = 30
# batch_size = 32:
# Testing accuracy: 0.475, n_epochs in training = 30
# bidirectional: Testing accuracy: 0.510, n_epochs = 30
# batch_size = 50:
# Testing accuracy: 0.500, n_epochs in training = 30
# learning rate = 0.05:# batch_size = 32:
# Testing accuracy: 0.530, n_epochs in training = 30
# learning rate = 0.03:
# Testing accuracy: 0.530
# n_epochs = 60:
# Testing accuracy: 0.505
# n_epochs = 90:
# Testing accuracy: 0.530
# no of layers = 2: Testing accuracy: 0.475


# In[23]:


# type = vanilla_rnn, gru or lstm
lstmmodel = RNNSentiment(rnn_type = "lstm", hidden_dim = 512, layer_dim = 1, bidirec = False)
# lstmmodel = RNNSentiment(rnn_type = "lstm", hidden_dim = 512, layer_dim = 2, bidirec = False)
# lstmmodel = RNNSentiment(rnn_type = "lstm", hidden_dim = 512, layer_dim = 1, bidirec = True) # instantiating a RNNSentiment model
# lstmmodel = RNNSentiment(rnn_type = "lstm", hidden_dim = 512, layer_dim = 2, bidirec = True)
lstmmodel = lstmmodel.cuda()    
# print(lstmmodel)


# In[24]:


trainval(model = lstmmodel, trainloader = dataset[0], validationloader = dataset[1], n_epochs = 30, optimizer = optim.SGD(lstmmodel.parameters(), lr=0.01), loss_fn = nn.BCELoss())
# trainval(model = lstmmodel, trainloader = dataset[0], validationloader = dataset[1], n_epochs = 30, optimizer = optim.SGD(lstmmodel.parameters(), lr=0.03), loss_fn = nn.BCELoss())
# learning rate = 0.01:
# batch_size = 16:
# Training accuracy: 0.608, Validation accuracy: 0.545, n_epochs = 30
# batch_size = 32:
# Training accuracy: 0.549, Validation accuracy: 0.475, n_epochs = 30
# bidirectional: Training accuracy: 0.534, Validation accuracy: 0.525, n_epochs = 30
# batch_size = 50:
# Training accuracy: 0.557, Validation accuracy: 0.500, n_epochs = 30
# learning rate = 0.05:# batch_size = 32:
# Training accuracy: 0.710, Validation accuracy: 0.455, n_epochs = 30
# learning rate = 0.03:
# Training accuracy: 0.616, Validation accuracy: 0.490
# n_epochs = 60:
# Training accuracy: 0.581, Validation accuracy: 0.540
# n_epochs = 90:
# Training accuracy: 0.643, Validation accuracy: 0.550
# no of layers = 2: Training accuracy: 0.528, Validation accuracy: 0.510 


# In[25]:


test(model = lstmmodel, testloader = dataset[2])
# learning rate = 0.01:
# batch_size = 16:
# Testing accuracy: 0.510, n_epochs = 30
# batch_size = 32:
# Testing accuracy: 0.470, n_epochs in training = 30
# bidirectional: Testing accuracy: 0.540, n_epochs = 30
# 2 layer bidirectional - testing accuracy - 0.520, n_epochs = 30
# batch_size = 50:
# Testing accuracy: 0.465, n_epochs in training = 30
# learning rate = 0.05:# batch_size = 32:
# Testing accuracy: 0.495, n_epochs in training = 30
# learning rate = 0.03:
# Testing accuracy: 0.520
# n_epochs = 60:
# Testing accuracy: 0.475
# n_epochs = 90:
# Testing accuracy: 0.480
# no of layers = 2: Testing accuracy: 0.465


# In[26]:


attnmodel = AttentionSentiment(hidden_dim = 2)
# attnmodel = AttentionSentiment()
attnmodel = attnmodel.cuda()
# print(attnmodel)


# In[27]:


trainval(model = attnmodel, trainloader = dataset[0], validationloader = dataset[1], n_epochs = 30, optimizer = optim.SGD(attnmodel.parameters(), lr=0.01), loss_fn = nn.BCELoss())
# trainval(model = attnmodel, trainloader = dataset[0], validationloader = dataset[1], n_epochs = 30, optimizer = optim.SGD(attnmodel.parameters(), lr=0.03), loss_fn = nn.BCELoss())

# learning rate = 0.01:
# batch_size = 16:
# Training accuracy: 0.666, Validation accuracy: 0.570, n_epochs = 30
# batch_size = 32:
# Training accuracy: 0.648, Validation accuracy: 0.640, n_epochs = 30
# batch_size = 50:
# Training accuracy: 0.638, Validation accuracy: 0.620 , n_epochs = 30
# learning rate = 0.05: # batch_size = 32:
# Training accuracy: 0.509, Validation accuracy: 0.455, n_epochs = 30
# learning rate = 0.03:
# Training accuracy: 0.694, Validation accuracy: 0.595
# n_epochs = 60:
# Training accuracy: 0.686, Validation accuracy: 0.600
# n_epochs = 90:
# Training accuracy: 0.708, Validation accuracy: 0.560


# In[28]:


test(model = attnmodel, testloader = dataset[2])
# learning rate = 0.01:
# batch_size = 16:
# Testing accuracy: 0.525, n_epochs = 30
# batch_size = 32:
# Testing accuracy: 0.535, n_epochs = 30
# batch_size = 50:
# Testing accuracy: 0.615, n_epochs = 30
# learning rate = 0.05: # batch_size = 32:
# Testing accuracy: 0.470, n_epochs = 30
# learning rate = 0.03:
# Testing accuracy: 0.635
# n_epochs = 60:
# Testing accuracy: 0.615
# n_epochs = 90:
# Testing accuracy: 0.570


# In[29]:


cnnmodel = CNNSentiment()
cnnmodel = cnnmodel.cuda()
# print(cnnmodel)


# In[30]:


trainval(model = cnnmodel, trainloader = dataset[0], validationloader = dataset[1], n_epochs = 30, optimizer = optim.SGD(cnnmodel.parameters(), lr=0.01), loss_fn = nn.BCELoss())
# lr = 0.01, n_epochs = 30, batch size = 32
# Training accuracy = 0.531, Validation accuracy: 0.540


# In[31]:


test(model = cnnmodel, testloader = dataset[2])
# lr = 0.01, n_epochs = 30, batch size = 32
# Testing accuracy: 0.475


# In[ ]:




