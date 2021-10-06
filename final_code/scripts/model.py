# Author : Prateek Chhikara
# Email  : prateekchhikara24@gmail.com

import transformers
import torch.nn as nn
import conf


class BertBinaryClassifier(nn.Module):
    def __init__(self):
        """
            Initialises the BERT embeddings, dropout, fully connected and a sigmoid layer.
        """
        super(BertBinaryClassifier, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(conf.BERT_MODEL_PATH)
        self.dropout = nn.Dropout(conf.DROPOUT)
        self.linear = nn.Linear(conf.EMBEDDING_SIZE, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, tokens, masks=None):
        """
            Creates a BERT model pipeline using pre-trained BERT embeddings and few additional layers.
        """
        _, pooled_output = self.bert(tokens, attention_mask = masks, output_hidden_states = False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        probability_output = self.sigmoid(linear_output)
        return probability_output, pooled_output