"""
Complex Word Identification using an embedding model followed by a simple
neural network.
"""

import json

import matplotlib.pyplot as plt
import torch

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader

from complex_word_classifier import ComplexWordClassifier


DATA_FILE = "data/data.json"

# Hyperparamters
EPOCHS = 10
LEARNING_RATE = 0.0005
BATCH_SIZE = 8
HIDDEN_SIZE = 400
DROPOUT = 0.1

run_name = f"single_hidden_{HIDDEN_SIZE}_{LEARNING_RATE}_{BATCH_SIZE}_{DROPOUT}"
MODEL_SAVE_PATH = run_name + ".pt"


# Load dataset, create dataloaders
with open(DATA_FILE, encoding="utf-8") as file_in:
    data = json.load(file_in)

train_items = [{"item": i["word"], "label": i["label"]} for i in data["train"]]
dev_items = [{"item": i["word"], "label": i["label"]} for i in data["dev"]]
test_items = [{"item": i["word"], "label": i["label"]} for i in data["test"]]

train_ds = Dataset.from_list(train_items)
dev_ds = Dataset.from_list(dev_items)
test_ds = Dataset.from_list(test_items)

train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)


# Load the Sentence Transformer model
embedding_model_path = "ibm-granite/granite-embedding-97m-multilingual-r2"
embedding_model = SentenceTransformer(embedding_model_path)


# Instantiate network
classification_network = ComplexWordClassifier(embedding_model,
                                               HIDDEN_SIZE,
                                               DROPOUT)

# Freeze embedding model layers
for name, param in classification_network.named_parameters():
    if name.startswith("embedding"):
        param.requires_grad = False


# Evaluation metric
def count_correct_answers(pred, label):
    """Simple correct answer count for later accuracy calculation."""
    pred = pred.argmax(axis=1)
    x = pred == label
    return x.sum()


# Set up loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(classification_network.parameters(),
                              lr=LEARNING_RATE)

# Lists for collecting values for learning curve plotting
train_curve_loss = []
train_curve_acc = []
val_curve_loss = []
val_curve_acc = []


# Training loop
for epoch in range(EPOCHS):

    running_loss = 0.
    last_loss = 0.
    correct = 0
    total = 0

    print("-"*100)
    print(f"--- EPOCH {epoch}")
    print("-"*100)

    classification_network.train()

    for i, data in enumerate(train_dataloader):
        inputs = data["item"]
        labels = data["label"]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = classification_network(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        print(f"\rStep {i}: loss {loss.item()}", end="")
        loss.backward()
        correct += count_correct_answers(outputs, labels)
        total += len(labels)

        # Adjust weights
        optimizer.step()

        # Report progress
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100  # Loss per batch
            print(f'\rBatch {i + 1} loss: {last_loss}')
            print(f"+ Acc: {correct / total}\n")
            # Gather data for learning curve
            train_curve_loss.append(last_loss)
            train_curve_acc.append(correct / total)
            # Reset variable
            running_loss = 0.

    # Validation loop
    classification_network.eval()

    with torch.no_grad():

        val_loss = 0
        val_correct = 0
        val_total = 0

        print("\n--- VALIDATION:")
        for i, data in enumerate(val_dataloader):
            # Get items
            inputs = data["item"]
            labels = data["label"]

            # Make predictions for this batch
            outputs = classification_network(inputs)

            # Compute results
            loss = loss_fn(outputs, labels)
            val_correct += count_correct_answers(outputs, labels)
            val_total += len(labels)
            val_loss += loss.item()

        last_loss = val_loss / (i+1)  # Loss per batch
        print(f'Loss: {last_loss}')
        print(f"Acc: {val_correct / val_total}")

        val_curve_loss.append(last_loss)
        val_curve_acc.append(val_correct / val_total)

        val_loss = 0.


# Save model
torch.save(classification_network.state_dict(), MODEL_SAVE_PATH)


# Plot learning curve
fig, ax = plt.subplots(2,1)

ax[0].plot(train_curve_loss)
ax[0].plot(train_curve_acc)
ax[1].plot(val_curve_loss)
ax[1].plot(val_curve_acc)

fig.savefig(f"plots/{run_name}.png")


# Compute test results
classification_network.eval()

with torch.no_grad():

    test_loss = 0
    test_correct = 0
    test_total = 0

    print("-"*100)
    print("--- TEST")
    print("-"*100)

    for i, data in enumerate(test_dataloader):
        inputs = data["item"]
        labels = data["label"]

        # Make predictions for this batch
        outputs = classification_network(inputs)

        # Compute results
        loss = loss_fn(outputs, labels)
        test_correct += count_correct_answers(outputs, labels)
        test_total += len(labels)
        test_loss += loss.item()

    last_loss = test_loss / (i+1)  # Loss per batch
    print(f'Batch {i + 1} loss: {last_loss}')
    print(f"Acc: {test_correct / test_total}")
    test_loss = 0.
    print("-"*100)


# TODO F1 instead of accuracy would be good

