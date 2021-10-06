# Author : Prateek Chhikara
# Email  : prateekchhikara24@gmail.com

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import conf

def accuracy(actual, pred):
    return round(accuracy_score(actual, pred), 3)

def precision(actual, pred):
    return round(precision_score(actual, pred), 3)

def recall(actual, pred):
    return round(recall_score(actual, pred), 3)

def f1score(actual, pred):
    return round(f1_score(actual, pred), 3)

def metrics_calc(actual, pred):
    """
        prints the classification metric values

        Parameters:
        -----------
            actual : the actual target values
            pred : the values predicted by the model
    """
    print("=============== Data Statistics: ====================")
    print("Accuracy = ", accuracy(actual, pred))
    print("Precision = ", precision(actual, pred))
    print("Recall = ", recall(actual, pred))
    print("F1 score = ", f1score(actual, pred))

def plot_data(data_points, plot_type):
    epochs = [i+1 for i in range(conf.EPOCHS)]
    plt.clf()
    plt.plot(epochs, data_points, color='red', marker='o')
    plt.title(plot_type + ' variation over the epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(plot_type, fontsize=14)
    plt.xticks(epochs)
    plt.grid(True)
    plt.savefig(conf.SAVED_FIGURE_PATH + plot_type + '.png')
