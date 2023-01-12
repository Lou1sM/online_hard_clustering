import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from dl_utils.tensor_funcs import numpyify
import json


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base").cuda()


def preprocess_document(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(input_ids=inputs['input_ids'].cuda(),attention_mask=inputs['attention_mask'].cuda())

    breakpoint()
    return outputs.last_hidden_state.squeeze(0).mean(axis=0)

with open('datasets/Tweets') as f:d=f.readlines()
data = [eval(item[:-1]) for item in d]
X_list = []
y_list = []
for i,item in enumerate(data):
    if i%100==0: print(i)
    embedding = numpyify(preprocess_document(item['textCleaned']))
    label = item['clusterNo']
    X_list.append(embedding)
    y_list.append(label)
    item['embedding']=embedding.tolist()

with open('datasets/tweets/tweets.json','w') as f:json.dump(data,f)

X = np.stack(X_list)
y = np.stack(y_list)
np.save('datasets/tweets/roberta_doc_vecs.npy',X)
np.save('datasets/tweets/cluster_labels.npy',y)
