from typing import Tuple, Union
from transformers import BertTokenizer, AutoModel, AutoTokenizer, BertForSequenceClassification, AdamW, BertConfig
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
import numpy as np
import os

# def task1(title: str) -> float:
#     return len(title) % 3 / 2

def task1(descriptions: list) -> float:
    
    val_dataloader = create_dataloader(descriptions)
        
    model_dir = 'distilbert_avito'
    
#     print(os.listdir())
#     print('/n')
#     print(os.listdir(model_dir))

    os.system('7z x distilbert_avito/pytorch_model.z01  -odistilbert_avito/')
    
    model = BertForSequenceClassification.from_pretrained(model_dir)
    
    device = 'cpu'
    model.to(device)
    model.eval()

    # Predict 
    predictions = []
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
        
    flat_predictions_raw = np.concatenate(predictions, axis=0)
    preds = F.softmax(torch.Tensor(flat_predictions_raw))[:,1]
    
    return preds


def create_dataloader(X_val, batch_size=64):
    
    sentences = X_val.tolist()

    input_ids_val = []
    attention_masks_val = []

    model_dir = 'distilbert_avito'
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                     
                            add_special_tokens = True,        
                            pad_to_max_length = True,
                            return_attention_mask = True, 
                            max_length = 512, 
                            return_tensors = 'pt', 
                       )
  
        input_ids_val.append(encoded_dict['input_ids'])
        attention_masks_val.append(encoded_dict['attention_mask'])

    input_ids_val = torch.cat(input_ids_val, dim=0)
    attention_masks_val = torch.cat(attention_masks_val, dim=0)

    dataset_val = TensorDataset(input_ids_val, attention_masks_val)
    validation_dataloader = DataLoader(
                dataset_val,
                sampler = SequentialSampler(dataset_val),
                batch_size = batch_size
            )
    
    return validation_dataloader


def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)
    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size


