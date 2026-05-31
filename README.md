# more-cwi

Different approaches to complex word identification outside what I did in my Bachelor's thesis.

## NN Classifier

This is a simple neural network classifier following an embedding layer.

- `train_model.py` trains the neural network based on the training data. Set the
  data file path in the python script.
- `use_model.py` to use the model. Adjust the model file path inside the script.

The data example file shows the required training data format.

## Fine-tuned T5 model

Find the code and a notebook with saved cell outputs in the directory `T5_seq2seq_CWI`.
You should be able to view the outputs at [nbviewer](https://nbviewer.org/github/a-klaus/more-cwi/blob/main/T5_seq2seq_CWI/T5_cwidentifier.ipynb).
