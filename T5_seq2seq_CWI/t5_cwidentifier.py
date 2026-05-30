# -*- coding: utf-8 -*-
"""T5 cwidentifier code only version"""

# Imports
from dataclasses import dataclass
import re
import string
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# Loading the model
model_name = "google-t5/t5-small"

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Set up data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

# Loading the dataset
DATAFILE = "mdr_4_unified.json"

dataset = load_dataset("json", data_files=DATAFILE, field="data")

# Split into train, dev and test
ds = dataset["train"].train_test_split(test_size=0.2)
print(ds)

# Inspect items
for i in ds["train"]:
    print("sentence:", i["sentence"])
    print("target:", i["target"])
    break

# Prepare the model input
prefix = "Identify complex words:"

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["sentence"]]
    model_inputs = tokenizer(inputs)

    labels = tokenizer(text_target=examples["target"])

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

ds_tokenized = ds.map(preprocess_function, batched=True)
ds_tokenized = ds_tokenized.filter(lambda x: len(x["input_ids"]) < 1025)

print(ds_tokenized)

# Create evaluation metrics

# Load metrics
wer = evaluate.load("wer")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

# Pattern for retrieving targets
pattern = r"\[([^ ]+)\]"


def retrieve_targets(items):
    """Find all targets in a given sequence."""
    return [re.findall(pattern, s) for s in items]

def postprocess_for_prec_rec(preds, labels):
    # Well one could do this elegantly with re, but this also works
    preds = [[1 if "]" in word else 0 for word in p.split()] for p in preds]
    labels = [[1 if "]" in word else 0 for word in l.split()] for l in labels]
    return preds, labels


def treat_double_words(decoded_str_list):
    """Treat double occurrences of words by numerating them."""
    returns = []
    for decoded_str in decoded_str_list:
        set_words = {}
        # Most punctuation will not be needed henceforth
        decoded_str = decoded_str.translate(str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@\\^_`{|}~"))

        decoded_str = decoded_str.split()
        for i in range(len(decoded_str)):
            # Word counter
            if decoded_str[i] in set_words:
                set_words[decoded_str[i]] += 1
            else:
                set_words[decoded_str[i]] = 0
            # Treat marked words
            if decoded_str[i].endswith("]"):
                decoded_str[i] = decoded_str[i][:-1] + f"_{set_words[decoded_str[i]]}]"
            else:
                decoded_str[i] = decoded_str[i] + f"_{set_words[decoded_str[i]]}"
        decoded_str = " ".join(decoded_str)
        returns.append(decoded_str)
    return returns


def compute_metrics(eval_preds):
    """Compute metrics for evaluation loop.

    Main component is a score combining WER (to check if the model changes the
    phrase, which it sadly does), precision and recall. For the calculation of
    the latter two, workarounds are needed for cases with no predictions or no
    correct targets.
    The individual scores and the generation length are returned as well

    Returns:
        dict: containing all metrics
    """
    preds, labels = eval_preds
    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Decode labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Treat double occurrences of words
    decoded_labels = treat_double_words(decoded_labels)
    decoded_preds = treat_double_words(decoded_preds)

    # Get targets
    predicted_targets = retrieve_targets(decoded_preds)
    correct_targets = retrieve_targets(decoded_labels)

    # Calculate precision and recall
    precisions, recalls = [], []
    for c_targets, p_targets in zip(correct_targets, predicted_targets):
        tp = [t for t in p_targets if t in c_targets]
        # Treat edge case: precision cannot be calculated
        if len(p_targets) == 0:
            if len(p_targets) == 0:
                precisions.append(1)
            else:
                precisions.append(0)
        else:
            precisions.append(len(tp) / len(p_targets))
        # Treat edge case: recall cannot be calculated
        if len(c_targets) == 0:
            if len(c_targets) == 0:
                recalls.append(1)
            else:
                recalls.append(0)
        else:
            recalls.append(len(tp) / len(c_targets))

    # Postprocessing for WER computation
    decoded_preds = [
        i.translate(str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@\\[]^_`{|}~"))
        for i in decoded_preds
    ]
    decoded_labels = [
        i.translate(str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@\\[]^_`{|}~"))
        for i in decoded_labels
    ]
    # Calculate individual WERs
    wers = [wer.compute(predictions=[d], references=[l]) for d, l in zip(decoded_preds, decoded_labels)]

    # Calculate the mean of all three scores for a unified view
    res = [(p+r+(1-w))/3 for p, r, w in zip(precisions, recalls, wers)]

    result = {
        "3score": np.mean(res),
        "wer": np.mean(wers),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls)
    }

    # Collect prediction length as well
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# Set up training arguments for the fine-tuning
training_args = Seq2SeqTrainingArguments(
    output_dir="german_cwidentifier",
    eval_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.1,
    save_total_limit=3,
    num_train_epochs=25,
    generation_max_length=1024,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_tokenized["train"],
    eval_dataset=ds_tokenized["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Fine-tuning!
trainer.train()

# Test example
model = trainer.model

sentence = "Identify complex words: Aktuell arbeiten wegen der hohen Energiepreise die Brennöfen auch nicht mehr permanent, sondern in Etappen."
input_ids = tokenizer(sentence, return_tensors="pt").to(model.device)

output = model.generate(**input_ids, cache_implementation="static", max_new_tokens=1024)

print("-"*100)
print("Input:", sentence)
print("Model output:")
print(tokenizer.decode(output[0], skip_special_tokens=True))

