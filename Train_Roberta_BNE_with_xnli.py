"""**0.   Definimos parámetros, instalamos paquetes, importamos bibliotecas**
"""

tam_train=392702
tam_test=5010

MODEL_TYPE = "PlanTL-GOB-ES/roberta-large-bne"
carpeta='/opt/salidas' ##poner la ruta local donde se deja el modelo entrenado
L_RATE = 1e-5
MAX_LEN = 256

NUM_EPOCHS = 4
BATCH_SIZE = 19
NUM_CORES=15

import torch
torch.cuda.empty_cache()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import pandas as pd
import numpy as np
import os
import gc

import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# set a seed value
torch.manual_seed(555)

from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score

import transformers
from transformers import AdamW

print(torch.__version__)

"""**1. Importamos el dataset**"""

import datasets
train_ds = datasets.load_dataset('xnli','es', split=datasets.ReadInstruction(
    'train', from_=1, to=tam_train, unit='abs'))
test_ds = datasets.load_dataset('xnli','es', split=datasets.ReadInstruction(
    'test', from_=1, to=tam_test, unit='abs'))

df_train = pd.DataFrame(train_ds)
df_test = pd.DataFrame(test_ds)

print(df_train.shape)
df_train.head()

print(df_test.shape)
df_test.head()

"""**2. Importamos el modelo y el tokenizer**"""

from transformers import RobertaTokenizer, RobertaForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained(MODEL_TYPE)
model =  RobertaForSequenceClassification.from_pretrained(MODEL_TYPE, num_labels=3)

# Send the model to the device.
model.to(device)

"""**3. Definimos la clase que va a tener los tokens y los labels del dataset**"""

class CompDataset(Dataset):

    def __init__(self, df):
        self.df_data = df

    def __getitem__(self, index):

        sentence1 = self.df_data.loc[index, 'premise']
        sentence2 = self.df_data.loc[index, 'hypothesis']

        encoded_dict = tokenizer.encode_plus(
                    sentence1, sentence2,           # Sentences to encode.
                    add_special_tokens = True,      # Add the special tokens.
                    max_length = MAX_LEN,           # Pad & truncate all sentences.
                    pad_to_max_length = True,
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',          # Return pytorch tensors.
                    padding="max_length", truncation=True
               )
        
        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        
        target = torch.tensor(self.df_data.loc[index, 'label'])

        sample = (padded_token_list, att_mask, target)

        return sample

    def __len__(self):
        return len(self.df_data)

"""**3.1 Creamos los dataloaders**"""

train_data = CompDataset(df_train)
test_data = CompDataset(df_test)


train_dataloader = torch.utils.data.DataLoader(train_data,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                       num_workers=NUM_CORES)

val_dataloader = torch.utils.data.DataLoader(test_data,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                       num_workers=NUM_CORES)


print(len(train_dataloader))
print(len(val_dataloader))

"""**4. Antes de hacer el fine tuning, probamos el accuracy del modelo sobre el dataset de train**"""

b_input_ids, b_input_mask, b_labels = next(iter(train_dataloader))

print(b_input_ids.shape)
print(b_input_mask.shape)
print(b_labels.shape)

# Pass a batch of train samples to the model.

batch = next(iter(train_dataloader))
# Send the data to the device
b_input_ids = batch[0].to(device)
b_input_mask = batch[1].to(device)
b_labels = batch[2].to(device)

# Run the model
outputs = model(b_input_ids, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)

preds = outputs[1].detach().cpu().numpy()

y_true = b_labels.detach().cpu().numpy()
y_pred = np.argmax(preds, axis=1)

# This is the accuracy without fine tuning.

val_acc = accuracy_score(y_true, y_pred)
print(val_acc)

"""**5. Definimos el optimizer y entrenamos**"""

# Define the optimizer
optimizer = AdamW(model.parameters(),
              lr = L_RATE, 
              eps = 1e-8 
            )

 
# Set the seed.
seed_val = 101
 
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
 
loss_values = []
 
for epoch in range(0, NUM_EPOCHS):
     
     print("")
     print('======== Epoch {:} / {:} ========'.format(epoch + 1, NUM_EPOCHS))
     
     stacked_val_labels = []
     targets_list = []
 
     # ========================================
     #               Training
     # ========================================
     
     print('Training...')
     
     # put the model into train mode
     model.train()

     torch.set_grad_enabled(True)
 
     total_train_loss = 0
 
     for i, batch in enumerate(train_dataloader):
         
         train_status = 'Batch ' + str(i) + ' of ' + str(len(train_dataloader))
         print(train_status, end='\r')
 
         b_input_ids = batch[0].to(device)
         b_input_mask = batch[1].to(device)
         b_labels = batch[2].to(device)
 
         model.zero_grad()        
 
         outputs = model(b_input_ids, 
                     attention_mask=b_input_mask,
                     labels=b_labels)
         
         loss = outputs[0]
         
         total_train_loss = total_train_loss + loss.item()
         
         optimizer.zero_grad()
         loss.backward()
         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

         optimizer.step() 
     
     print('Train loss:' ,total_train_loss)
 
     # ========================================
     #               Validation
     # ========================================
     
     print('\nValidation...')
 
     # Put the model in evaluation mode.
     model.eval()
 
     torch.set_grad_enabled(False)
     total_val_loss = 0
     
 
     for j, batch in enumerate(val_dataloader):
         
         val_status = 'Batch ' + str(j) + ' of ' + str(len(val_dataloader))
         print(val_status, end='\r')
 
         b_input_ids = batch[0].to(device)
         b_input_mask = batch[1].to(device)
         b_labels = batch[2].to(device)      

         outputs = model(b_input_ids, 
                 attention_mask=b_input_mask, 
                 labels=b_labels)
         
         loss = outputs[0]

         total_val_loss = total_val_loss + loss.item()

         preds = outputs[1]
         val_preds = preds.detach().cpu().numpy()
         targets_np = b_labels.to('cpu').numpy()
         targets_list.extend(targets_np)
 
         if j == 0:  # first batch
             stacked_val_preds = val_preds
         else:
             stacked_val_preds = np.vstack((stacked_val_preds, val_preds))

     # Calculate the validation accuracy
     y_true = targets_list
     y_pred = np.argmax(stacked_val_preds, axis=1)
     val_acc = accuracy_score(y_true, y_pred)
     
     print('Val loss:' ,total_val_loss)
     print('Val acc: ', val_acc)

     # Save the Model
     model.save_pretrained(carpeta+'/Model'+str(epoch)+'/')
     
     # Use the garbage collector to save memory.
     gc.collect()
