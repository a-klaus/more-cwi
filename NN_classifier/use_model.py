"""
Load and use Complex Word Classifier model
"""

import torch

from sentence_transformers import SentenceTransformer, util

from complex_word_classifier import ComplexWordClassifier


HIDDEN_SIZE = 400
DROPOUT = 0.1

PATH = "example.pt"

# Load the Sentence Transformer model
embedding_model_path = "ibm-granite/granite-embedding-97m-multilingual-r2"
embedding_model = SentenceTransformer(embedding_model_path)

model = ComplexWordClassifier(embedding_model, HIDDEN_SIZE, DROPOUT)
model.load_state_dict(torch.load(PATH, weights_only=True))
model.eval()


def classify(word):
    """Simple classification function"""
    with torch.no_grad():
        scores = model([word])
    verdict = scores.argmax(axis=1).item()
    if verdict == 0:
        return "einfach", scores
    else:
        return "komplex", scores


# User command line interface
while True:
    word = input("Wort: ")
    if not word or word in {"exit", "stop", "stopp", ""}:
        break
    print(classify(word))

