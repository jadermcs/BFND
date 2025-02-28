#!/usr/bin/env python
from transformers import pipeline
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from rapidfuzz import fuzz
import spacy
from hashlib import sha256


import torch
from transformers import BertTokenizer, BertForNextSentencePrediction


def predict_next_sentence(sentence_a, sentence_b):
    """
    Use BERT to predict if sentence_b is a continuation of sentence_a.

    Args:
        sentence_a (str): The first sentence
        sentence_b (str): The potential next sentence

    Returns:
        tuple: (is_next_prediction (bool), probability (float))
    """
    # Load pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    # Tokenize the input
    # BERT takes input in format: [CLS] sentence_a [SEP] sentence_b [SEP]
    encoded_input = tokenizer(
            sentence_a, sentence_b, return_tensors='pt', padding=True)
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        encoded_input = {key: val.cuda() for key, val in encoded_input.items()}
    # Set model to evaluation mode
    model.eval()
    # Get prediction
    with torch.no_grad():
        outputs = model(**encoded_input)
        predictions = outputs.logits
    # The model returns logits for two labels:
    # [0] = IsNextSentence, [1] = NotNextSentence
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(predictions, dim=1)
    # Get the prediction and probability
    is_next_prediction = bool(probs[0, 0] > probs[0, 1])
    probability = float(probs[0, 0] if is_next_prediction else probs[0, 1])
    return is_next_prediction, probability


nltk.download('punkt')  # Download the necessary resources
nlp = spacy.load("en_core_web_sm")

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"


sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
sentiment_task("Covid cases are increasing fast!")

dataset = load_dataset("biglam/hmd_newspapers")["train"]


dataset = dataset.filter(lambda x:
                         x["ocr_quality_mean"] and x["ocr_quality_mean"] > 0.8)


def process(example):
    example["text"] = example["text"].replace("\n", " ")
    example["text"] = example["text"].replace("- ", "")
    return example


dataset = dataset.map(process)


dataset.filter(lambda x: "imigrant" in x["text"])[0]


def get_hash(examples):
    examples["hash"] = sha256(examples["text"].encode('utf-8')).hexdigest()[:16]
    return examples


dataset = dataset.map(get_hash)


def sentence_split(examples):
    sentences = sum([sent_tokenize(ex) for ex in examples["text"]], [])
    source = [examples["hash"]] * len(sentences)
    return {"sentence": sentences, "source": source}


dataset = dataset.map(
        sentence_split,
        batched=True,
        remove_columns=dataset.column_names,
        batch_size=100)

dataset = dataset.flatten()

paragraphs = []
previous_sentence = None
running_paragraph = ""
for example in dataset:
    if len(paragraphs) > 5:
        break
    if previous_sentence:
        is_next, probability = predict_next_sentence(previous_sentence, example["sentence"])
    else:
        probability = 1.
    if probability > 0.8:
        running_paragraph += " " + example["sentence"]
    else:
        paragraphs.append(running_paragraph)
        running_paragraph = ""
    previous_sentence = example["sentence"]

print(paragraphs)
exit()

def mentions_nation(example):
    doc = nlp(example["sentence"])
    # Extract nationalities (NORP entities)
    example["nationality"] = [ent.text for ent in doc.ents if ent.label_ == "NORP"]
    return example


# dataset = dataset.map(mentions_nation).filter(lambda x: len(x["nationality"]))

def get_score(example):
    pred = sentiment_task(example["sentence"][:1000])[0]
    if pred["label"] == "positive":
        score = pred["score"]
    elif pred["label"] == "negative":
        score = -pred["score"]
    else:
        score = .0
    example["score"] = score
    return example


# In[82]:


dataset = dataset.map(get_score).filter(lambda x: x["score"] < 0.)


# In[83]:


keywords = set()
for n in dataset["nationality"]:
    lower = [x.lower() for x in n]
    keywords.update(lower)
for k in keywords:
    print(k)


# In[84]:


with open("keywords.txt", "w") as fout:
    for k in keywords:
        fout.write(k+"\n")


# In[85]:


for example in dataset.filter(lambda x: x["score"] < -0.9):
    print(example)


# In[86]:


dataset[0]


# In[22]:


column = "text"
threshold = 90
count = 0
to_drop = []

for index1, row1 in df.iterrows():
    for index2, row2 in df.iloc[index1+1:].iterrows():
        if fuzz.ratio(row1[column], row2[column]) > threshold:
            to_drop.append(index2)
            count += 1


# In[ ]:


df = df.drop(to_drop)
print("Number of duplicated rows:", count)


# In[ ]:


# Aggregate the scores by date and calculate the mean for each day
df.index = pd.to_datetime(df['date'])
df_aggregated = df.resample(rule='YE')['score'].mean().reset_index().fillna(0.)


# In[ ]:


# Plotting the data
plt.figure(figsize=(20, 4))
plt.plot(df_aggregated['date'], df_aggregated['score'], marker='o')
plt.title('Average Score Over Time')
plt.xlabel('Date')
plt.ylabel('Average Score')
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


df_aggregated['smoothed_score'] = df_aggregated['score'].rolling(window=3, min_periods=1).mean()


# In[ ]:


# Plot the original and smoothed data
plt.figure(figsize=(15, 4))
plt.plot(df_aggregated['date'], df_aggregated['score'], label='Original', marker='o', alpha=0.6)
plt.plot(df_aggregated['date'], df_aggregated['smoothed_score'], label='Smoothed (Rolling Avg)', linewidth=2)
plt.title('Average Score Over Time with Smoothing')
plt.xlabel('Date')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


df[df.score < 0]


# In[ ]:


df[df.score < -0.9]["text"].values


# In[ ]:




