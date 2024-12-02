import torch
import configs
import pandas as pd
import torch.nn.functional as F

#Read and preprocess dataset
def get_dataset(csv_file, drop_low_samples=True):
    dataset = pd.read_csv(csv_file)
    #Preprocessing steps: 1. Remove unreadable and NA values 2. standardized identity format 3. (Optional) drop words that are too long
    cleaned_data = dataset[dataset['IDENTITY'] != 'UNREADABLE'].dropna()
    cleaned_data['IDENTITY'] = cleaned_data['IDENTITY'].str.lower()
    cleaned_data = cleaned_data.reset_index(drop=True)
    if drop_low_samples:
        indices = [idx for idx, label in enumerate(
            cleaned_data['IDENTITY'].values) if len(label) > configs.max_length]
        cleaned_data = cleaned_data.drop(index=indices)
        cleaned_data = cleaned_data.reset_index(drop=True)
    return cleaned_data


#Get the integer to character (decoding) mappings and character to integer (encoding) mappings
def get_vocabulary():
    vocabulary = [' ', "'", '-', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    map_integer_to_character = dict(enumerate(vocabulary))
    map_integer_to_character = {k+1: v for k, v in map_integer_to_character.items()}
    map_character_to_integer = {v: k for k, v in map_integer_to_character.items()}

    return map_integer_to_character, map_character_to_integer

#Encoder function
def encode(string):
    _, map_character_to_integer = get_vocabulary()
    #Convert word into token tensor
    word_list = []
    for i in string:
        word_list.append(map_character_to_integer[i])
    token = torch.tensor(word_list)
    #Add padding so all tokens are same dimensions
    pad_token = F.pad(token, pad=(0, configs.MAX_LENGTH-len(token)),
                      mode='constant', value=0)
    return pad_token


#Decoder function
def decode(token):
    map_integer_to_character, _ = get_vocabulary()
    token = token[token != 0]
    string_list = []
    for i in token:
        string_list.append(map_integer_to_character[i.item()])
    return "".join(string_list)