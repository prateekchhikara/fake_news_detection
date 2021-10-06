# Author : Prateek Chhikara
# Email  : prateekchhikara24@gmail.com

import conf
import torch
import transformers
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from keras.preprocessing.sequence import pad_sequences

class PreProcessing:
    def __init__(self, input_data):
        """
            The input data is preprocessed and input and target variables are extracted.\
            The pre-trained BERT tokenizer is initialized

            Parameters:
            -----------
            input_data : this is the dataset .tsv file having 13 input variables and one target variable.
                
        """
        self.statements, self.labels = self.feature_selection(input_data)
        print("Loading the pre-trained BERT tokenizer...")
        self.tokenizer = transformers.BertTokenizer.from_pretrained(conf.BERT_MODEL_PATH, do_lower_case=True)
        self.max_length = conf.MAX_LENGTH
        self.padding = "post"
        self.truncate = "post"
        

    def feature_selection(self, input_data):
        """
            This method extracts the needful features from the input dataset.

            Parameters:
            -----------
                input_data : this is the dataset .tsv file having 13 input variables and one target variable.

            Returns:
            -----------
                x_label : input variable (Eg: ['string 1', 'string 2])
                y_label : target variable (Eg: [0, 1])
        """
        dataset = pd.read_csv(input_data, delimiter='\t', names=conf.columns_list)
        x_label = list(dataset['statement'])
        y_label = [conf.label_mappings[l] for l in list(dataset['label'])]
        return x_label, y_label

    def tokenization(self):
        """
            This method converts the input strings into tokens using BERT tokenizer and adds special tokens [CLS] and [SEP],\
            at starting and ending of the token list.

            Returns:
            -----------
            The generated token list.
        """
        return list(map(lambda t: ['[CLS]'] + self.tokenizer.tokenize(t)[:510] + ['[SEP]'], self.statements))

    def tokensToIds(self):
        """
            This method converts the input tokens of variable length to a length of 512 by adding 'post' padding.

            Returns:
            -----------
            The method returns token IDs for each sentence of length 512.
                
        """
        tokens = self.tokenization()
        tokens_ids = pad_sequences(
            list(map(self.tokenizer.convert_tokens_to_ids, tokens)), 
            maxlen = conf.MAX_LENGTH, truncating = self.truncate, padding = self.padding, dtype = "int")
        return tokens_ids

    def createMask(self):
        """
            This method creates masks corresponding to the input variable.  

            Returns:
            -----------
            This method returns masks containing 0 and 1 for every input sequence.
                
        """
        tokens_ids = self.tokensToIds()
        mask = [[float(token > 0) for token in token_id] for token_id in tokens_ids]
        return mask

    def preProcess(self):
        """
            This method combines input variables, corresponding masks and target variables.

            Returns:
            -----------
            This method returns the data loader using the inputs.
        """
        print(" --> Data Pre-processing started")
        tokens_tensor = torch.tensor(self.tokensToIds())
        label_tensor = torch.tensor(np.array(self.labels).reshape(-1, 1)).float()
        masks_tensor = torch.tensor(self.createMask())

        dataset = TensorDataset(tokens_tensor, masks_tensor, label_tensor)
        dataset_sampler = RandomSampler(dataset)
        train_dataloader = DataLoader(dataset, sampler = dataset_sampler, batch_size = conf.BATCH_SIZE)
        print(" --> Data Pre-processing completed")

        return train_dataloader




