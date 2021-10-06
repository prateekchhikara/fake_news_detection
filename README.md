
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

