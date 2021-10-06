
# Fake News Detection

The consumption of news articles on social media platforms is a double-edged sword. Being low cost and
providing fast access to rich content have led people to seek out and consume news from social media. However,
it permits widespread “fake news,” containing low-quality news with intentionally false information. The
widespread of fake news has the potential for highly negative impacts on individuals and society. Therefore,
fake news should be detected before disseminating it on social media platforms.

## Dataset

The LIAR dataset is used for experimentation and analysis purposes. The dataset is in .tsv format and following
are the columns present in the dataset.

    1. Column 1: the ID of the statement ([ID].json).
    2. Column 2: the label.
    3. Column 3: the statement.
    4. Column 4: the subject(s).
    5. Column 5: the speaker.
    6. Column 6: the speaker’s job title.
    7. Column 7: the state info.
    8. Column 8: the party affiliation.
    9. Column 9-13: the total credit history count, including the current statement. \
        (a) Column 9: barely true counts. 
        (b) Column 10: false counts. 
        (c) Column 11: half true counts.
        (d) Column 12: mostly true counts. 
        (e) Column 13: pants on fire counts. 
    10. Column 14: the context (venue / location of the speech or statement).

## Used Scheme

The current problem statement is binary text classification. First, we have to extract features from the
text. There are various traditional text feature extraction approaches, such as the bag of words, TF-IDF, etc.
Presently, deep learning approaches are widely used, such as LSTMs, bi-LSTMs, GRUs, etc. These models
are not bi-directional (even bi-LSTM). The bidirectionality of a model is essential for truly understanding
the meaning of a language. To illustrate, the word “bank” has a different context in both of the mentioned
sentences. (a) Prateek wants to hang out near the bank, and (b) Prateek wants to deposit his salary to the
bank. If we try to predict the context of the word “bank” by only taking either the left or the right context, we will be making errors. BERT solves this problem by considering both the left and the right context before
making a prediction.

BERT can represent text using some sequence as an input; it looks left and right many times to produces a
vector representation for each word as the output. BERT can be used in two ways:

1. Feature Extraction: we use the final output of BERT as an input to another model. This way, we extract features from text using BERT and then use it in a separate model for the actual task at hand.

2. Fine-tuning: we add additional layers on top of BERT and then train the whole network. By this, we train our additional layers and also fine-tune the BERTs weights and parameters.


### Pre-processing

The sentences are pre-processed using the following steps. First, the input sentences are trimmed for white
spaces. For instance, let us take the following sentence of length four characters.
“My name is Prateek”

Then I have used BERT tokenizer to create tokens of the input string. The tokens will be as follows:
```
["my", "name", "is", "pr", "##ate", "##eek"]
```

The token length equals 6. Further, I have added special tokens, [CLS] and [SEP], at the starting and end
of the token list. The token list will now be of length 8. In the dataset, the sentences will be of varying size;
hence we have to make the same length. Therefore, the input token list is added with paddings to make its
length equals to 512. Now, for each sentence, we have a token list of length 512. Further, each token in the
token list is mapped to corresponding embeddings using the lookup table of 30,000. Each token will have an
embedding length of 768. Therefore, for each sentence, the dimensions will be 512x768. I have also created
masks of dimension 512x1, which contains 0 at the place of paddings and 1 on the remaining places. Then, I
made a data loader using the data, its corresponding mask, and the target variables.


### Embedding Extraction

Bidirectional Encoder Representations from Transformers (BERT) is a deep learning model in which every
input element is connected to every input element. Traditionally, language models could only read text input
in a sequence manner, either left-to-right or right-to-left, but couldn’t do both simultaneously. BERT reads the
input data in both directions at once. BERT can be used as a feature extractor from the input texts. Therefore, I
have used BERT to extract features from my input texts. The BERT is trained on millions of sentences from the
Wikipedia dataset. To increase the feature extraction power of BERT for the current classification task, I have
performed fine-tuning the entire network using the dropout and sigmoid layer on top of the BERT architecture.
This way, I got a more enhanced version of the feature extractor. The comparison of fine-tuning and without
fine-tuning architecture is made in the result section.









## Installation

### Environment setup:
```
Create a ‘venv’ or ‘conda’ environment and install the ‘requirements.txt’ containing all the necessary dependencies.
Major dependencies are:
• Keras==2.3.0
• numpy==1.16.6
• scikit-learn==0.24.2
• tensorflow==1.14.0
• torch==1.9.0
• transformers==3.0.0
```
### Inference/Demo:
If you want to predict whether the given text is fake or real then run the following script:
```
python3 inference.py --model type < x > --input statement < textstring >
```
where, x can be replaced with ‘svm’, ‘rf’, or ‘lr’ (default value is sigmoid).
The text string can be replaced with any string that you want to predict (default string is “Artificial Intelligence
will rule the world in coming future ...!!!”).
To run this command, the model weights should be present in their respective folder.
weights location: /final_code/output=models

### BERT fine-tuning:
If you want to fine-tune the pre-trained BERT model on the training dataset then run the following script:
```
python3 train.py
```
To run this command, the pre-trained model weights and dataset should be present in their respective folder.
pre-trained weights location: /final_code/data/bert_base_uncased_model
dataset location: /final_code/data/liar_dataset

### Classifiers fine-tuning:
If you want to fine-tune the classifiers models on top of the BERT embeddings then run the following script:
```
python3 param_tuning.py < x >
```
where, x can be replaced with ‘svm’, ‘rf’, or ‘lr’. 
To run this command, the model weights and dataset should be present in their respective folder.
weights location: /final_code/output_models
dataset location: /final_code/data/liar_dataset

### Testing:

If you want to test any model on the test data then run the following script:
```
python3 test.py --model type < x >
```
where, x can be replaced with ‘svm’, ‘rf’, or ‘lr’ (default value is sigmoid).
To run this command, the model weights and dataset should be present in their respective folder.
weights location: /final_code/output_models
dataset location: /final_code/data/liar_dataset


# Model weights

[Download Weight files](https://drive.google.com/drive/folders/1yH7LiJN09Sla3kIy0bSdcYzxyncpzIaj?usp=sharing)