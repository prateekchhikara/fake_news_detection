# Author : Prateek Chhikara
# Email  : prateekchhikara24@gmail.com

from tqdm import tqdm
import conf, pre_processing
import model
import torch
import utils
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--model_type', default='sigmoid', type=str)

args = parser.parse_args()


class Tester():

    def __init__(self, model_name):
        """
            Initialises the BERT fine-tuned embedding model, and loading it to the available device.\
            SVM fine-tuned model is also loaded.
        """
        print(" --> Embedding model initialized.")
        self.embedding_model = model.BertBinaryClassifier()
        self.embedding_model.to(self.deviceType())
        if self.deviceType() == "cuda":
            print('using GPU')
            self.embedding_model.load_state_dict(torch.load(conf.SAVED_MODEL_PATH))
        else:
            print('GPU not found, therefore using CPU')
            self.embedding_model.load_state_dict(torch.load(conf.SAVED_MODEL_PATH, map_location = torch.device('cpu')))
        self.embedding_model.eval()
        self.test_data_embeddings = []
        self.actual_labels = []
        self.model_name = model_name
        self.base_model = False

        if self.model_name == 'svm':
            print("Loading SVM model")
            self.model = pickle.load(open(conf.SVM_MODEL_PATH, 'rb'))
        elif self.model_name == 'rf':
            print("Loading RF model")
            self.model = pickle.load(open(conf.RF_MODEL_PATH, 'rb'))
        elif self.model_name == 'lr':
            print("Loading LR model")
            self.model = pickle.load(open(conf.LR_MODEL_PATH, 'rb'))
        else:
            print("Loading base model")
            self.base_model = True

    def deviceType(self):
        """
            Checks for available device type.

            Returns:
            -----------
                "cuda" : if GPU is present and detected
                "cpu" : if CPU is detected or GPU is not detected
        """
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def test(self, data_loader):
        """
            The function performs model testing on the test set. First, embeddings are extracted from text\
            using fine-tuned BERT embedding extractor, and further SVM is used for the prediction.

            Parameters:
            -----------
                data_loader : it contains three attributes; token embeddings, token masks, and target variables
        """
        print("============== Testing on the test dataset ================")
        device = self.deviceType()
        BERT_final_predictions = []
        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
                token_ids = data[0].to(device)
                masks = data[1].to(device)
                label = data[2].to(device)
                final_output, embeddings = self.embedding_model(token_ids, masks)
                
                if self.base_model:
                    final_output = final_output.cpu().detach().numpy()
                    BERT_final_predictions += list(final_output[:, 0] > 0.5)
                else:
                    for embed in range(len(embeddings)):
                        self.test_data_embeddings.append(embeddings[embed].cpu().detach().numpy())
                self.actual_labels.extend([int(i) for i in list(label.cpu().detach().numpy())])

        if self.base_model:
            predictions = BERT_final_predictions
        else:
            self.test_data_embeddings = np.array(self.test_data_embeddings)
            self.actual_labels = np.array(self.actual_labels)
            predictions = self.model.predict(self.test_data_embeddings)

        utils.metrics_calc(self.actual_labels, predictions)



if __name__ == "__main__":
    t1 = time.time()
    d = pre_processing.PreProcessing(conf.TEST_DATA)
    data_loader = d.preProcess()
    t = Tester(args.model_type)
    t.test(data_loader)

    t2 = time.time()
    print(90*'-')
    print("Total Testing time : ", round(t2 - t1, 2), " sec")
    print(90*'-')