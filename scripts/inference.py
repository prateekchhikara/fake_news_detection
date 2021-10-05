# Author : Prateek Chhikara
# Email  : prateekchhikara24@gmail.com

import argparse
import model
import transformers
import pre_processing
import conf
import torch
import pickle
import time
from keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Inferencing')
parser.add_argument('--model_type', default = "sigmoid", type=str)
parser.add_argument('--input_statement', default = "Artificial Intelligence will rule the world in coming future ...!!!", type=str)
args = parser.parse_args()

def extract_embeddings(input_data):
    """
        This function returns the embeddings corresponding to the input text given

        Parameters:
        -----------
            input_data : this is an input string on which inference is to be made

        Returns:
        -----------
            This function return the embeddings of the input text, and the output of the sigmoid
    """

    # Fine-tuned BERT model is loaded
    embedding_model = model.BertBinaryClassifier()
    if torch.cuda.is_available():
        device = "cuda"
        embedding_model.to(device)
        embedding_model.load_state_dict(torch.load(conf.SAVED_MODEL_PATH))
    else:
        device = "cpu"
        embedding_model.to(device)
        embedding_model.load_state_dict(torch.load(conf.SAVED_MODEL_PATH, map_location = torch.device('cpu')))
    embedding_model.eval()

    # The input data is pre-preocessed by using tokenization and then adding special tokens and padding
    tokenizer = transformers.BertTokenizer.from_pretrained(conf.BERT_MODEL_PATH, do_lower_case=True)
    input_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]'], input_data))
    token_ids = pad_sequences(
            list(map(tokenizer.convert_tokens_to_ids, input_tokens)), 
            maxlen = conf.MAX_LENGTH, truncating = "post", padding = "post", dtype = "int")


    masks = [[float(token > 0) for token in token_id] for token_id in token_ids]

    # Token IDs and masks are laoded on the available devices
    token_ids = torch.tensor(token_ids).to(device)
    masks = torch.tensor(masks).to(device)

    # Fine-tuned BERT model used as a feature extractor
    tag, embeddings = embedding_model(token_ids, masks)

    return embeddings, tag

def predict_class(embeddings, model_name):
    """
        This function loads the saved SVM/RF/LR model and use BERT embeddings to predict the output.

        Parameters:
        -----------
            embeddings : embeddings calculated by the fine-tuned BERT model
            model_name : the model which will be used for the predictions
    """
    if model_name == 'svm':
        model = pickle.load(open(conf.SVM_MODEL_PATH, 'rb'))
    elif model_name == 'lr':
        model = pickle.load(open(conf.LR_MODEL_PATH, 'rb'))
    elif model_name == 'rf':
        model = pickle.load(open(conf.RF_MODEL_PATH, 'rb'))


    predictions = model.predict(embeddings.cpu().detach().numpy())
    print(90*'-')
    if predictions[0] == 1:
        print("Output :  The news article is Fake")
    else:
        print("Output : The news article sounds credible")
    print(90*'-')


if __name__ == "__main__":

    if args.model_type not in ['rf', 'lr', 'svm', 'sigmoid']:
        print("Mention correct model from [lr, svm, rf]")
        exit()

    t1 = time.time()
    print(90*'-')
    print('TEXT : ', args.input_statement)
    input_data = [args.input_statement]

    # extracting embeddings and tag from the fine-tuned BERT model.
    embeddings, tag = extract_embeddings(input_data)

    # if we want the base classifier (sigmoid) for the prediction
    if args.model_type == 'sigmoid':
        tag = tag.cpu().detach().numpy()
        print(90*'-')
        if tag[0] > 0.5:
            print("Output :  The news article is Fake")
        else:
            print("Output : The news article sounds credible")
        print(90*'-')
    else: # if we want either of the SVM, RF, LR classifier for the prediction
        predict_class(embeddings, args.model_type)

        
    t2 = time.time()
    print("Total inference time : ", round(t2 - t1, 2), " sec")
    print(90*'-')