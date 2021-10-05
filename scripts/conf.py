# Author : Prateek Chhikara
# Email  : prateekchhikara24@gmail.com

# hyper-paramters

MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-5
EMBEDDING_SIZE = 768
DROPOUT = 0.2
CROSS_VALIDATION = 5

# paths

TRAIN_DATA = "../data/liar_dataset/train.tsv"
VAL_DATA = "../data/liar_dataset/valid.tsv"
TEST_DATA = "../data/liar_dataset/test.tsv"
BERT_MODEL_PATH = "../data/bert_base_uncased_model"
SAVED_MODEL_PATH = "../output/models/model.pth"
SVM_MODEL_PATH = "../output/models/svm.pkl"
RF_MODEL_PATH = "../output/models/rf.pkl"
LR_MODEL_PATH = "../output/models/lr.pkl"
SAVED_FIGURE_PATH = "../output/figures/"


# pre defined parameters and variable values

columns_list = ['ID', 'label', 'statement', 'subject', 'speaker', 'speaker_job_title',
            'state', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts',
            'mostly_true_counts', 'pants_on_fire_counts', 'statement_location']

label_mappings = {
    'half-true' : 1,
    'false' : 1,
    'barely-true' : 1,
    'pants-fire' : 1,
    'mostly-true' : 0,
    'true' : 0
}

svm_params = {
    'C' : [1, 10, 50, 100],
    'degree' : [2, 3, 4],
    'kernel' : ['linear', 'poly', 'rbf']
}

rf_params = {
    'n_estimators' : [2, 5, 10, 20],
    'max_features' : [300, 400, 500, 600]
}

lr_params = {
    'solver' : ['newton-cg', 'lbfgs', 'liblinear'],
    'penalty' : ['l2'],
    'C' : [10.0, 1.0, 0.1, 0.01]
}
