import tarfile
import sys
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import re
# import pandas as pd
import numpy as np
from collections import Counter

"""
How to check if pytorch is using the GPU?

In [1]: import torch

In [2]: torch.cuda.current_device()
Out[2]: 0

In [3]: torch.cuda.device(0)
Out[3]: <torch.cuda.device at 0x241b438f8c8>

In [4]: torch.cuda.device_count()
Out[4]: 1

In [5]: torch.cuda.get_device_name(0)
Out[5]: 'GeForce GTX 1050 Ti'

In [6]: torch.cuda.is_available()
Out[6]: True

This tells me the GPU GeForce GTX 1050 Ti is being used by PyTorch.
"""


"""
How to run network in GPU.:

    use .cuda() on any input batches/tensors (see x.cuda() in forward method of any model class in this source code)
    Not only data batch but also true labels batch(for calculating loss)
    (for training, validation and testing)
    
    
    use .cuda() on your network module, which will hold your network, like:

    class MyModel(nn.Module):
    def init(self):
    self.layer1 = nn. …
    self.layer2 = nn. …
    … etc …

then just do:

model = MyModel()
model.cuda()
"""





# Task 1: Load the data
# For this task you will load the data, create a vocabulary and encode the reviews with integers

def read_file(path_to_dataset):
    """
    :param path_to_dataset: a path to the tar file (dataset)
    :return: two lists, one containing the movie reviews and another containing the corresponding labels
    """
    # print("hello there")
    if not tarfile.is_tarfile(path_to_dataset):
        sys.exit("Input path is not a tar file")
        
    dirent = tarfile.open(path_to_dataset) # a TarFile object is created using tarfile.open()
    # review_list = dirent.getmembers() # Return the members of the archive as a list of TarInfo objects
    # A TarInfo object represents one member in a TarFile.
    # print(len(review_list))
    # print(review_list)
    
    count = 1
    data = [] # output list of reviews
    labels = [] # output list of review labels
    
    dirent.extractall(path=".", members=None) # Extract all members from the archive to the current working directory 
    dirent.close() # we dont need to keep this file open anymore 
    
    pospath = sys.path[0]+"\\movie_reviews\\pos"
    #print(pospath)
    # posdir = open(pospath)
    for posfilename in os.listdir(pospath): # listdir() returns a list containing the names of the entries in the directory
        
        # skipping filenames starting with "._"
        pattern = re.compile("^\._")
        if(pattern.match(posfilename)):
            continue
        
        # print(pospath+"\\"+posfilename)
        # print(posfilename)
        posfile = open(pospath+"\\"+posfilename)
        contents = posfile.read()
#         contents = contents.lower() # no need since we have vectors for both The & the
        data.append(contents) # adding reviews to data list
        labels.append("positive") # adding review sentiments to labels list
        # print(contents)
        # break
        
    posfile.close()   
    
    #print(data[len(data)-1])
    #print(labels[len(data)-1])
    
    negpath = sys.path[0]+"\\movie_reviews\\neg"
    
    for negfilename in os.listdir(negpath):
        pattern = re.compile("^\._")
        if(pattern.match(negfilename)):
            continue
        
        negfile = open(negpath+"\\"+negfilename)
        contents = negfile.read()
        data.append(contents) # adding reviews to data list
        labels.append("negative") # adding review sentiments to labels list
        
    negfile.close()    
        
    #print(data[len(data)-1])
    #print(labels[len(data)-1])
    
    # dirent.extract("movie_reviews") # same as the above method
    
    # Generate a random order of elements with np.random.permutation and simply index into the arrays data and labels with those to shuffle the dataset, else it will be [0,0,...0,0,0,0,0,1,1,1,....1,1,1]
    
    x = []
    y = []
    idx = np.random.permutation(len(data)) # generates a list of random indices within range
#     print(idx)
    for i in idx:
        x.append(data[i])
        y.append(labels[i])
        
    # Now we have randomly arranged sampled dataset with reviews and its labels in same order
#     x,y = data[idx], labels[idx]
#     print(x)
    
    print("Loaded reviews and labels from the TAR file successfully!") 

    return x, y # by default, a tuple of (data, labels) is returned
    
    """
    for member in dirent.getmembers(): #.getmembers() returns the members of the archive as a list of TarInfo objects
        if not member.isfile():
            continue
        
        #print(member.name)
        list1 = os.listdir(sys.path[0]+"\\"+member.name)
        print(list1[0])
        break
        
        f = dirent.extractfile(member) # f is directory called movie_reviews. we need whats inside
        path1 = f
        posdirec = open(os.path.join("pos", "5_1.txt"), "r") # opening another directory called pos
        content = posdirec.read()
        # content=f.read()
        
      """
        
        # print(content.decode('utf-8', "ignore"))
        
       
    
    
# Task 1.1 done!

def preprocess(text): # receives list of reviews, have to assign unique integer to each word
    """
    :param text: list of sentences or movie reviews
    :return: a dict of all tokens you encounter in the dataset. i.e. the vocabulary of the dataset
    Associate each token with a unique integer
    """

    if type(text) is not list:
        sys.exit("Please provide a list to the method")
        
#     alltext = ""
        
    all_text = ' '.join(text) # concatenating all reviews together
    
    words = all_text.split() # create a list of all words in all reviews
    
    count_words = Counter(words) # Count all the words using Counter Method

    total_words = len(words)
    sorted_words = count_words.most_common(total_words)
    
#     print (count_words) # dictionary of words to their frequency
    
#     vocab_to_int = {w:i for i, (w,c) in enumerate(sorted_words)} # creates index from 0, we need from 1 only cos we do 0 padding later
      
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)} # creates a indexing dictionary such that frequent words in review set gets small indices
        
#     print (vocab_to_int)
    print("Assigned a unique integer to each word in the vocabulary!")
    
    return vocab_to_int

"""        
#     uniqwords = []
    vocab = {}
    id = 1 # since we use zero padding later, we cant use 0 as a word code
    for review in text:
        wordlist = review.split()
#         print(wordlist[0])
        for word in wordlist:
            if word not in vocab:
                vocab[word] = id
                id += 1
#         break

#     print(vocab["films"])
    print("Assigned a unique integer to each word in the vocabulary!")
    return vocab
"""

# Task 1.2 done!

def encode_review(vocab, text):
    """
    :param vocab: the vocabulary dictionary you obtained from the previous method
    :param text: list of movie reviews obtained from the previous method
    :return: encoded reviews
    """

    if type(vocab) is not dict or type(text) is not list:
        sys.exit("Please provide a dict and list to the method")
        
    
    data = [] # encoded reviews
    
    for review in text:
        encoded = []
        for word in review.split():
#             print(word)
            encoded.append(vocab[word])
        
        data.append(encoded)
    print("Encoded all reviews by assigning a unique integer for each word in vocabulary!")    
    return data
    
    """
    COMPLETE THE REST OF THE METHOD
    """
# Task 1.3 done!

def encode_labels(labels): # Note this method is optional (if you have not integer-encoded the labels)
    """
    :param labels: list of labels associated with the reviews
    :return: encoded labels
    """

    if type(labels) is not list:
        sys.exit("Please provide a list to the method")

        
    encoded_labels = [1 if label =='positive' else 0 for label in labels]
    encoded_labels = np.array(encoded_labels)
    
#     print(encoded_labels)
    print("Encoded labels list - 1 for positive & 0 for negative!")
    return encoded_labels
    
"""
    for i in range(0, len(labels)):
        if labels[i] == "positive":
            labels[i] = 1
        else:
            labels[i] = 0
    
#     print("hello there!")
    print("Encoded labels list - 1 for positive & 0 for negative!")
    return labels  
"""        

# Task 1.4 done!

def pad_zeros(encoded_reviews, seq_length = 200):
    """
    :param encoded_reviews: integer-encoded reviews obtained from the previous method
    :param seq_length: maximum allowed sequence length for the review
    :return: encoded reviews after padding zeros
    """

    if type(encoded_reviews) is not list:
        sys.exit("Please provide a list to the method")

        
    ''' Return features of encoded_reviews, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(encoded_reviews), seq_length), dtype = int) # creating numpy array of zeroes
    
    for i, review in enumerate(encoded_reviews): # iterating through the encoded_reviews list
        review_len = len(review) # getting length of each review
        
        if review_len <= seq_length: # needs zero padding
            zeroes = list(np.zeros(seq_length-review_len)) # getting list of additional zeroes required
            new = zeroes+review   # padding zeroes in the front and not in the back     
            
        elif review_len > seq_length: # encoded review needs to be truncated
            new = review[0:seq_length] # getting truncated list of review 
        
        features[i,:] = np.array(new) # ith element of the numpy array - 'features' is the numpy array of each encoded review of equal length
    
#     print(features)
    print("Made all encoded reviews to be of equal length (= seq_length) by padding/truncating!")
    return features
    
    
"""        
    data = [] # encoded reviews list with zero padding
        
    for encoded_review in encoded_reviews:
        res = [0]*seq_length # making a list of zeroes
        i = 0
        for digit in encoded_review:
            # print(digit)
            res[i] = digit
            i += 1
            if i == 200:
                break # the encoded review has length > 200, but we need only first 200 encoded words
        data.append(res)
        break
#     print("hello!")
    print("Padded zeroes to all encoded reviews!")
    return data
"""

# Task 1.5 done!


# Task 2: Load the pre-trained embedding vectors
# For this task you will load the pre-trained embedding vectors from Word2Vec

# CONSTRUCTING THE EMBEDDING LAYER !!!!!!
def load_embedding_file(embedding_file, token_dict): # token_dict = {"the":1, ",":2,....}
    """
    :param embedding_file: path to the embedding file
    :param token_dict: token-integer mapping dict obtained from previous step
    :return: embedding dict: embedding vector-integer mapping
    """

    if not os.path.isfile(embedding_file):
        sys.exit("Input embedding path is not a file")
    if type(token_dict) is not dict:
        sys.exit("Input a dictionary!")

#     print("Hey there!")
    print("Loading the embedding vec file and creating dictionary - {word : vector}...")    
    vecfile = open(embedding_file, 'r', encoding="utf-8")
#     print(vecfile.read(5000))
  
    token_vectors = {}# dict of numpy array vector for each word
    for line in vecfile.readlines():
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]]) # masterpiece line 
        token_vectors[word] = embedding 
    
#     return token_vectors

#     print("vector for The = " + token_vectors["the"]) 
    
    # This will take some time, but at least the server wont crash now
    print("Vectors for all words were loaded from vec file successfully!")    
    
    vecfile.close()
  
#     dummy = [0.0] * 300
    
    embed_dict = {}
    
    
    print("Creating embed_dictionary - {word code : vector}...")
#     dummy = np.zeros(300, dtype='float32') # [0.0] * 300
    
    for each_word in token_dict.keys(): # for each unique word in vocabulary
        if each_word in token_vectors:
            embed_dict[token_dict[each_word]] = token_vectors[each_word]
        else:
            embed_dict[token_dict[each_word]] = np.zeros(300, dtype='float32') # for words in vocabulary with no vector representation in the vec file
    
#     print(embed_dict[52])
    print("Embed_dictionary successfully created!")
    
    
    
    return embed_dict
    
    # dimension of vec file is 300. IT'S GIVEN IN THE FILENAME ITSELF!!!!



# Task 2 done!


"""    
#     Each line in .vec file is like this: "the" 0.456456 0.345345 0.345345 ...300 elements
#     1 Million words, each with a 300 dimension vector

    token_vectors = {} # {"the" : [0.456456 0.345345 0.345345 ...300 elements]}
    firstline = True
    
#     df0 = pd.read_csv(embedding_file, sep='\n')
#     print(df0[["the"]])

 
    for line in vecfile.readlines():
#         if firstline:
#             firstline = False
#             continue
        words = line.split()
        vector = []
        for i in range(1, len(words)):# skipping the token and going for vectors
            vector.append(words[i])
        
        token_vectors[words[0]] = vector # creating 
#         break    
    
#     print(token_vectors)
     
    df = pd.DataFrame.from_dict(token_vectors)
    print(df[["the"]])
"""
    
        
# Task 3: Create a TensorDataset and DataLoader

def create_data_loader(encoded_reviews, encoded_labels, batch_size = 32): # Create a TensorDataset and DataLoader
    """
    :param encoded_reviews: zero-paddded integer-encoded reviews
    :param labels: integer-encoded labels
    :param batch_size: batch size for training
    :return: DataLoader object
    """
    
#     encoded_reviews =  encoded_reviews.tolist()
#     encoded_labels =  encoded_labels.tolist()

#     if type(encoded_reviews) is not list or type(encoded_labels) is not list:
#         sys.exit("Please provide a list to the method")
        
        
    # Now that we have got our data in nice shape, we will split it into training, validation and test sets 
    # train= 80% | valid = 10% | test = 10%
    
#     features = encoded_reviews
    
    
    len_encoded_reviews = len(encoded_reviews)
    
    split_frac = 0.8 # 80% for training
    train_x = encoded_reviews[0:int(split_frac*len_encoded_reviews)] # getting a fraction of the reviews (encoded) from list
    train_y = encoded_labels[0:int(split_frac*len_encoded_reviews)] # getting same fraction of labels (encoded) from list
    
    remaining_x = encoded_reviews[int(split_frac*len_encoded_reviews):]
    remaining_y = encoded_labels[int(split_frac*len_encoded_reviews):]
    
    valid_x = remaining_x[0:int(len(remaining_x)*0.5)]
    valid_y = remaining_y[0:int(len(remaining_y)*0.5)]
    
    test_x = remaining_x[int(len(remaining_x)*0.5):]
    test_y = remaining_y[int(len(remaining_y)*0.5):]
    
    
    """
    After creating our training, test and validation data. Next step is to create dataloaders for this data. We will use a    TensorDataset for batching our data into batches. This is one of a very useful utility in PyTorch for using our data with DataLoaders with exact same ease as of torchvision datasets.
    """
    
    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    
    # dataloaders
#     batch_size = 50  # already given as 32
    
    # make sure to SHUFFLE your data (within the batch)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return train_loader, valid_loader, test_loader
    
"""
    # In order to obtain one batch of training data for visualization purpose we will create a data iterator


    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()
    
    print('Sample input size: ', sample_x.size()) # will show batch_size (no. of input samples), seq_length(dimension of each)
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size()) # batch_size (50 corresponding labels, all labels are single digit: 1 or 0)
    print('Sample label: \n', sample_y)
        
"""

# Task 3 done!

# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)
    
    
    
# Task 4: Define the Baseline model here

# This is the baseline model that contains an embedding layer and an fcn for classification
class BaseSentiment(nn.Module):
    """
    The layers are as follows:
    0. Tokenize : This is not a layer for the network but a mandatory step of converting our words into tokens (integers)
    1. Embedding Layer: that converts our word tokens (integers) into embedding of specific size   
    2. Fully Connected Layer: that maps output of embedding layer to a desired output size
    3. Sigmoid Activation Layer: that turns all output values in a value between 0 and 1 (binary classifier: 1 or 0)
    4. Output: Sigmoid output from the last timestep is considered as the final output of this network (output 1 dimension)
    """
    
    
    
    def __init__(self, vocab_size = 50921, embedding_dim = 300):
        super(BaseSentiment, self).__init__()
#         self.embedding = nn.Embedding(vocab_size (no. of unique words in dataset + 1), vector_size (dimension in vec file))
#         vocab_size = len(vocab)+1 # +1 for the 0 padding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.embedding = self.embedding.cuda()
        
        # [32, 200, 300] - output dimension of embedding layer, 32 (batch size), 200(dim of each sample), 300 (word vector dim)
        
        # self.flatten = Flatten()  ## describing the flattening layer
        
        # putting the embedding layer output into fc layer it has to be flattened out - so dim = 32*200*300 
        self.fc = nn.Linear(300, 1) # output size = 1 (binary classifier)
#         self.fc = self.fc.cuda()
        
    def forward (self, x): # x = input_words at the start
        
        x = x.type(torch.LongTensor) # converting to scalar long from int tensor as required
        
        x = x.cuda()
        
        x = self.embedding(x) # first layer
        
        
        # Flatten x for the FC layer
        # x = x.view(x.shape[0], -1)
#         x = x.reshape(-1) 
        
        x = x.view(-1, 300) # flattening input to dimension of 300 for fc layer
        
        # x = self.flatten(x)   # second layer - using flatten layer 
        
        # Fully connected layer with ReLU activation
        x = F.relu(self.fc(x)) # third layer
        
        # x = nn.Sigmoid(x) # output layer # nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
#         x = F.sigmoid(x)
        x = torch.sigmoid(x)
        return x
        pass


    
    
        
            
            
                 
            
            
# Task 5: Define the RNN model here

# This model contains an embedding layer, an rnn and an fcn for classification
class RNNSentiment(nn.Module): #If there is only one layer, dropout is not applied
    def __init__(self, rnn_type, hidden_dim, layer_dim, bidirec, vocab_size = 50921, embedding_dim = 300):
        
        # input_dim of rnn = embedding dim  #  WKT output_dim of model is 2 (binary classifier)
        
        super(RNNSentiment, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
        # [32, 200, 300] - output dimension of embedding layer, 32 (batch size), 200(dim of each sample), 300 (word vector dim)
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # self.input_dim = embedding_dim
        
#         self.bidirec = bidirec
        
        if rnn_type == "vanilla_rnn" :
            # RNN

#             self.rnn = nn.RNN(embedding_dim, hidden_dim, layer_dim, dropout=0.7, batch_first=True)
#             dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got num_layers=1. # dropout to prevent overfitting which is happening
            
            # The arguments have to be in this order (input_dim = embedding_dim)
            self.rnn = nn.RNN(embedding_dim, hidden_dim, layer_dim, bidirectional=bidirec, batch_first=True)
        elif rnn_type == "gru" :
            # GRU

            self.rnn = nn.GRU(embedding_dim, hidden_dim, layer_dim, bidirectional=bidirec, batch_first=True)
        elif rnn_type == "lstm" : 
            #LSTM

            self.rnn = nn.LSTM(embedding_dim, hidden_dim, layer_dim, bidirectional=bidirec, batch_first=True)
            
        # putting the rnn layer output into fc layer 
        self.fc = nn.Linear(hidden_dim, 1) # output size = 1 (binary classifier)
        
    def forward (self, x): # x = input_words at the start
        
        x = x.type(torch.LongTensor) # converting to scalar long from int tensor as required
        
        x = x.cuda() # converting to cuda datatype for cuda
        
        x = self.embedding(x) # first layer
        
        # initializing the hidden state to 0
        hidden = None # instead of dropouts..
        
        # One time step
        x, h = self.rnn(x, hidden) # second layer
    
        
        # Fully connected layer with ReLU activation
        x = F.relu(self.fc(x.contiguous().view(-1, self.hidden_dim))) # third layer # flattening o/p of rnn to give to fc layer
        
        x = torch.sigmoid(x)
        return x
        pass
        
        
        
        


# This model contains an embedding layer, self-attention and an fcn for classification
class AttentionSentiment(nn.Module):
    def __init__(self, hidden_dim, vocab_size = 50921, embedding_dim = 300, output_size = 1): # setting hidden_dim = 2
        super(AttentionSentiment, self).__init__()
             
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.Attn = nn.MultiheadAttention(embedding_dim, num_heads=1)
        self.fc = nn.Linear(hidden_dim, output_size) # output size = 1 (binary classifier)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.type(torch.LongTensor)
        x = x.cuda()
        embeds = self.embedding(x)
        out = self.Attn(embeds, embeds, embeds)[0]
        sig_out = self.sig(out)
#         sig_out = sig_out.view(batch_size, -1)
#         sig_out = sig_out[:, -1]
        return sig_out
 

# CNN
"""
The idea of the Convolutional Neural Netwok for Text Classification is very simple. We perform convolution operation on the embedding matrix whose shape for each batch is (sample_dim, embedding_length) with kernel of varying height but constant width which is same as the embedding_length. We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected to the output layers consisting two units which basically gives us the logits for both positive and negative classes.
"""

class CNNSentiment(nn.Module):
    # reference : https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/selfAttention.py
    # reference :  https://www.aiworkbox.com/lessons/how-to-define-a-convolutional-layer-in-pytorch
    def __init__(self, vocab_size = 50921, embedding_dim = 300):
        super(CNNSentiment, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        # The kernel size can also be given as a tuple of two numbers indicating the height and width of the filter respectively if a square filter is not desired
        # out_channels : Number of output channels after convolution operation performed on the input matrix (ur choice)
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, (kernel_height[0]=3, embedding_dim), stride=1, padding=1)
        self.conv1 = nn.Conv2d(1, 16, (3, embedding_dim), stride=1, padding=0)
        # kernel_size is the size of the filter that is run over the images
        # The stride argument indicates how far the filter is moved after each computation. 
        self.conv2 = nn.Conv2d(1, 16, (3, embedding_dim), stride=1, padding=1)
        self.conv3 = nn.Conv2d(1, 16, (3, embedding_dim), stride=1, padding=1)
#         self.dropout = nn.Dropout(keep_probab)
        self.dropout = nn.Dropout(0.7)
        
# kernel_height : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.        
#  With a stride of 1 in the first convolutional layer, a computation will be done for every pixel in the image.
# With a stride of 2, every second pixel will have computation done on it, and the output data will have a height and width that is half the size of the input data. I would not recommend changing the stride from 1 without a thorough understanding of how this impacts the data moving through the network. 
# The padding argument indicates how much 0 padding is added to the edges of the data during computation.    
# Without good reason to change this, the padding should be equal to the kernel size minus 1 divided by 2. 
# This prevents the image shrinking as it moves through the layers. 
# keep_probab : Probability of retaining an activation node during dropout operation

#         self.fc = nn.Linear(len(kernel_heights)*out_channels, output_size)
        self.fc = nn.Linear(48, 1) # len(kernel_size) = 3, 3*out_channels = 3*16 = 48


    def conv_block(self, input, conv_layer):
        
        # problem is in conv layer, last dim of o/p is 3, not 1 as needed******************
        
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1) != ([32, 16, 198, 3])
#         activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)

#         print("before: " + str(conv_out.size()))
#         conv_out = torch.squeeze(conv_out, conv_out.size()[3]) # removing the dimension = 3
        conv_out = torch.mean(conv_out, -1) # removing last dimension by applying mean over the last dimension
#         conv_out = conv_out.squeeze(1)

#         print("after: " + str(conv_out.size()))
    
        activation = F.relu(conv_out)
#         activation = F.relu(conv_out.squeeze())# activation.size() = (batch_size, out_channels, dim)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)
        
        return max_out    
    

    
    
    def forward(self, x): # x = input_words at the start
        
        x = x.type(torch.LongTensor) 
        
        x = x.cuda() 
        
        x = self.embedding(x) # x.size() = (batch_size=32, sample_dim=200, embedding_length=300)      
        
        x = x.unsqueeze(1) # x.size() = (batch_size, 1, sample_dim, embedding_length)
          
#         print("before conv_block: "+str(x.size()))   # above claim is correct 
        
        max_out1 = self.conv_block(x, self.conv1)
        max_out2 = self.conv_block(x, self.conv2)
        max_out3 = self.conv_block(x, self.conv3)
        
        # concatenating
        all_out = torch.cat((max_out1, max_out2, max_out3), 1) # all_out.size() = (batch_size, num_kernels*out_channels)
        
        fc_in = self.dropout(all_out) # fc_in.size()) = (batch_size, num_kernels*out_channels)
        
        y = self.fc(fc_in) 
    
        y = torch.sigmoid(y)
    
        return y
                      
        pass 
























"""
ALL METHODS AND CLASSES HAVE BEEN DEFINED! TIME TO START EXECUTION!!
"""

# Task 7: Start model training and testing

# Instantiate all hyper-parameters and objects here

# Define loss and optimizer here

# Training starts!!

# Testing starts!!
def trainval (model, trainloader, validationloader, n_epochs, optimizer, loss_fn): # hyperparameters: n_epochs, learning rate in optimizer
    # cudnn.benchmark = True # seems useless & causes error
    
    
    for epoch in range(n_epochs):
            
            # Resetting before each epoch because we are measuring after each epoch
            # Keep track of training and validation loss
        train_loss = 0.0
        train_acc = 0
            
            
        
#       running_loss = 0
            
        model.train()
        for batch_idx,train in enumerate(trainloader):

            x,y = train[0], train[1].cuda() # converting to cuda datatype for cuda
            # x will be converted to cuda when passed through network, but y will remain cpu datatype, so will create error in loss calculation section 
       
            optimizer.zero_grad()    

# When you do multiple backwards passes with the same parameters, the gradients are accumulated. This means that 
# you need to zero the gradients on each training pass or you'll retain gradients from previous training batches.
  
            y_pred = model.forward(x)
            batch_size = x.shape[0]
            y_pred = y_pred.view(batch_size, -1) #  to reshape the output such that rows = batch size

        #       print ('Final sentiment prediction, ', y_pred[:,-1])
            y_pred = y_pred[:,-1] # we only want the output after the last sequence (after the last timestep)
                
                # calculates loss for this batch
            loss = loss_fn(y_pred,y.float()) # needs the labels as float, not int
                
                
        #       print(loss)
        #       break   

            y_pred = (y_pred > 0.5).float() # returns float 1 or 0
            correct = (y_pred == y).float().sum() # number of correct predictions in the batch

# Now that we know how to calculate a loss, how do we use it to perform backpropagation? Torch provides a module,
# autograd, for automatically calculating the gradients of tensors. We can use it to calculate the gradients of all 
# our parameters with respect to the loss. Autograd works by keeping track of operations performed on tensors, then 
# going backwards through those operations, calculating gradients along the way. To make sure PyTorch keeps track of 
# operations on a tensor and calculates the gradients, you need to set requires_grad = True on a tensor. 
# You can do this at creation with the requires_grad keyword, or at any time with x.requires_grad_(True).

                # print('Before backward pass: \n', model.fc.weight.grad)
            loss.backward()
                # print('After backward pass: \n', model.fc.weight.grad)
        
                # Take an update step and few the new weights
            optimizer.step() # optimizer.step performs a parameter update based on the current gradient and the update rule
                #print('Updated weights - ', model[0].weight)
        
#             train_loss += loss.item() # adds loss over all batches 
            
#                 print("correct = " + str(correct))
            train_acc += correct.item() # adding up the correct predictions over all samples
                # correct.item() - to convert from 1D tensor to int
                
            
            
            
            # after 1 epoch of full training set pass
            

            
            # calculate average losses for each epoch 
#         train_loss = train_loss/len(trainloader) # len(trainloader) gives number of batches
            # train_loss has added up loss for all batches, not over all samples like train_acc
            
            ########################### (1 epoch = all samples) !!!!!!!
            
            # since there are just 50 batches (found that out) in training set, let it print out 50 times for each epoch
#             print("train_loss for epoch "+ str(epoch) + "/n_epochs = " + str(train_loss))             
            
    
    
#             The y_pred In this case is the last batch output, where we will validate on for each epoch. So we should be dividing the batch size of the last iteration of the epoch. loss is also from the last batch only.    
    
            #Accuracy # Printing after every epoch
#             y_pred = (y_pred > 0.5).float() # returns float 1 or 0
#             correct = (y_pred == y).float().sum() # number of correct predictions in the batch
            
#             print("train_acc = "+str(train_acc/len(trainloader.dataset)))
            
########            print("Epoch {}/{}, Training loss: {:.3f}, Training accuracy: {:.3f}".format(epoch+1,n_epochs, train_loss, train_acc/len(trainloader.dataset))) # epoch+1 since 0 indexing
        
        # train_acc/len(trainloader.dataset) : out of all samples, how many were correctly predicted
        # len(trainloader.dataset) gives total training samples (1600)
        
        
                
        
        valid_loss = 0.0
        valid_acc = 0        
        
        
        model.eval() # no dropout, this is just evaluation, not training

        for batch_idx,val in enumerate(validationloader):
            x,y = val[0], val[1].cuda()
            optimizer.zero_grad()
            y_pred = model.forward(x)
            batch_size = x.shape[0]
            y_pred = y_pred.view(batch_size, -1)
            y_pred = y_pred[:,-1] 
            loss = loss_fn(y_pred,y.float())
                
                # no loss.backward() & optimizer.step() for validation 
                
            y_pred = (y_pred > 0.5).float() # returns float 1 or 0
            correct = (y_pred == y).float().sum() # number of correct predictions in the batch
#             valid_loss += loss.item()
            valid_acc += correct.item()

                

#         valid_loss = valid_loss/len(validationloader) # len(validationloader) = no. of batches = 7
        
   

            
        if((epoch+1)% 10 == 0): # print every 10 epochs
#             print("Epoch {}/{}, Training loss: {:.3f}, Training accuracy: {:.3f}".format(epoch+1,n_epochs, train_loss, train_acc/len(trainloader.dataset))) # epoch+1 since 0 indexing
#             print("Epoch {}/{}, Validation loss: {:.3f}, Validation accuracy: {:.3f}".format(epoch+1,n_epochs, valid_loss, valid_acc/len(validationloader.dataset)))
            # printing only accuracy
            print("Epoch {}/{}, Training accuracy: {:.3f}".format(epoch+1,n_epochs, train_acc/len(trainloader.dataset))) # epoch+1 since 0 indexing
            print("Epoch {}/{}, Validation accuracy: {:.3f}".format(epoch+1,n_epochs, valid_acc/len(validationloader.dataset)))

    print('Finished Training!!!!!!')            
        

        
def test (model, testloader):
       
    test_acc = 0       
    model.eval() 

    for batch_idx, test in enumerate(testloader):
        x,y = test[0], test[1].cuda()
        y_pred = model.forward(x)
        batch_size = x.shape[0]
        y_pred = y_pred.view(batch_size, -1)
        y_pred = y_pred[:,-1] 
                
        y_pred = (y_pred > 0.5).float() # returns tensor of batch-size float 1 or 0 so as to compare with true labels
        correct = (y_pred == y).float().sum() # number of correct predictions in the batch
        test_acc += correct.item()            
            
            
    print("Testing accuracy: {:.3f}".format(test_acc/len(testloader.dataset)))

    print('Finished Testing!!!!!!')
