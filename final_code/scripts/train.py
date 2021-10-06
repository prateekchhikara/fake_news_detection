# Author : Prateek Chhikara
# Email  : prateekchhikara24@gmail.com

from tqdm import tqdm
import conf, pre_processing
import model
import torch
import utils
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import time
import warnings
warnings.filterwarnings("ignore")

class Trainer():

    def __init__(self):
        self.max_norm = 1.0
        self.actual_labels = []
        self.predictions = []

    def loss_fn(self, predicted, actual):
        """
            Binary cross entropy loss function is defined here.

            Parameters:
            -----------
                predicted : model predictions values (length = batch size)
                actual : actual label values (length = batch size)

            Returns:
            -----------
                loss value between the predicted and the actual labels.
        """
        return nn.BCELoss()(predicted, actual)

    def optimizer(self, model):
        """
            Optimizer function is defined here.

            Parameters:
            -----------
                model : the loaded BERT classification model

            Returns:
            -----------
                optimizer attached to the model's parameters
        """
        return Adam(model.parameters(), lr = conf.LEARNING_RATE)

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

    def getModel(self):
        """
            returns a loaded model

            Returns:
            -----------
                fetches the model, load it either on GPU or CPU depending on available device type
        """
        bert_model = model.BertBinaryClassifier()
        if self.deviceType() == "cuda":
            print("=============== Loading model on GPU ==================")
            bert_model = bert_model.cuda()
        else:
            print("=============== Loading model on CPU (GPU not detected or disabled) ==================")
        return bert_model

    def train(self, model, data_loader):
        """
            The function performs model training. The training is performed end to end without freezing any \
            of the model layers.

            Parameters:
            -----------
                model : the loaded BERT classification model
                data_loader : it contains three attributes; token embeddings, token masks, and target variables
        """
        model.train()
        device = self.deviceType()
        best_accuracy = 0
        all_accuracy_values = []
        all_loss_values = []
        for epoch in range(conf.EPOCHS):
            print(f"============== EPOCH {epoch + 1} has started =================")
            train_loss = 0
            accuracy = 0
            predictions = []
            actual_labels = []
            for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
                token_ids = data[0].to(device)
                masks = data[1].to(device)
                label = data[2].to(device)
                pred, _ = model(token_ids, masks)

                loss = self.loss_fn(pred, label)
                train_loss += loss.item()

                model.zero_grad()
                loss.backward()

                clip_grad_norm_(parameters = model.parameters(), max_norm = self.max_norm)
                self.optimizer(model).step()

                predictions.extend([float(i) for i in list(pred.cpu().detach().numpy())])
                actual_labels.extend([int(i) for i in list(label.cpu().detach().numpy())])


            predictions = [1 if i > 0.5 else 0 for i in predictions]
            accuracy = utils.accuracy(actual_labels, predictions)
            print("Current Epoch's accuracy = ", accuracy)
            print("Best accuracy = ", best_accuracy)
            print("Loss is = ", round(train_loss, 3))

            all_accuracy_values.append(accuracy)
            all_loss_values.append(round(train_loss, 3))

            if accuracy > best_accuracy:
                torch.save(model.state_dict(), conf.SAVED_MODEL_PATH)
                best_accuracy = accuracy
                self.predictions = predictions
                self.actual_labels = actual_labels

        utils.plot_data(all_accuracy_values, "accuracy")
        utils.plot_data(all_loss_values, "loss")

        

if __name__ == "__main__":
    t1 = time.time()
    
    d = pre_processing.PreProcessing(conf.TRAIN_DATA)
    t = Trainer()
    
    data_loader = d.preProcess()
    model = t.getModel()
    
    t.train(model, data_loader)

    t2 = time.time()
    print(90*'-')
    print("Total Training time : ", round(t2 - t1, 2), " sec")
    print(90*'-')
    utils.metrics_calc(t.actual_labels, t.predictions)
    
