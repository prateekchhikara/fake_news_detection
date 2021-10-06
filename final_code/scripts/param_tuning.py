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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle
import warnings
import sys
warnings.filterwarnings("ignore")

class ParamTuning():

    def __init__(self, model_name):
        self.embedding_model = model.BertBinaryClassifier()
        self.embedding_model.to(self.deviceType())
        if self.deviceType() == "cuda":
            self.embedding_model.load_state_dict(torch.load(conf.SAVED_MODEL_PATH))
        else:
            self.embedding_model.load_state_dict(torch.load(conf.SAVED_MODEL_PATH, map_location = torch.device('cpu')))
        self.embedding_model.eval()
        self.model = None
        self.model_name = model_name

    def deviceType(self):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def feature_extraction(self, data_loader):
        device = self.deviceType()
        actual_labels = []
        data_embeddings = []

        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
                token_ids = data[0].to(device)
                masks = data[1].to(device)
                label = data[2].to(device)
                _, embeddings = self.embedding_model(token_ids, masks)
                for embed in range(len(embeddings)):
                    data_embeddings.append(embeddings[embed].cpu().detach().numpy())
                actual_labels.extend([int(i) for i in list(label.cpu().detach().numpy())])

        return np.array(data_embeddings), np.array(actual_labels)
        

    def tuning(self, train_data_loader, val_data_loader):
        train_embeddings, train_labels = self.feature_extraction(train_data_loader)
        val_embeddings, val_labels = self.feature_extraction(val_data_loader)

        final_embeddings = np.concatenate((train_embeddings,val_embeddings), axis=0)
        final_labels = np.concatenate((train_labels, val_labels), axis = 0)
        print(final_embeddings.shape, final_labels.shape)

        if self.model_name == 'svm':
            self.model = SVC()
            clf = GridSearchCV(self.model, conf.svm_params, cv = conf.CROSS_VALIDATION, scoring = 'accuracy')
            clf.fit(final_embeddings, final_labels)
            pickle.dump(clf.best_estimator_, open(conf.SVM_MODEL_PATH, 'wb'))
            print("Best SVM parameters are: ", clf.best_params_)
        elif self.model_name == 'rf':
            self.model = RandomForestClassifier()
            clf = GridSearchCV(self.model, conf.rf_params, cv = conf.CROSS_VALIDATION, scoring = 'accuracy')
            clf.fit(final_embeddings, final_labels)
            pickle.dump(clf.best_estimator_, open(conf.RF_MODEL_PATH, 'wb'))
            print("Best RF parameters are: ", clf.best_params_)
        elif self.model_name == 'lr':
            self.model = LogisticRegression()
            clf = GridSearchCV(self.model, conf.lr_params, cv = conf.CROSS_VALIDATION, scoring = 'accuracy')
            clf.fit(final_embeddings, final_labels)
            pickle.dump(clf.best_estimator_, open(conf.LR_MODEL_PATH, 'wb'))
            print("Best LR parameters are: ", clf.best_params_)
        else:
            print("Please Specify a correct model name")
        
        pred = clf.best_estimator_.predict(final_embeddings)
        utils.metrics_calc(final_labels, pred)
 



if __name__ == "__main__":
    try:
        model_name = sys.argv[1]
        if model_name not in ['svm', 'rf', 'lr']:
            print("Please enter correct model name.")
            exit()
    except IndexError:
        print("Please enter a model name: either svm or lr or rf")


    d_train = pre_processing.PreProcessing(conf.TRAIN_DATA)
    d_val = pre_processing.PreProcessing(conf.VAL_DATA)

    train_data_loader = d_train.preProcess()
    val_data_loader = d_val.preProcess()

    t = ParamTuning(model_name)
    t.tuning(train_data_loader, val_data_loader)

    # t.metrics_calc(t.actual_labels, t.test_data_embeddings)