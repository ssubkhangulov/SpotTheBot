import numpy as np
import json

def make_data_from_texts(n, d, lang, texts, vec_type):
    absolute_path = '/home/sshsubkhangulov/sultan_folder/data'
    svd_dict_path = f'{absolute_path}/{lang}/text_clean/dictionaries/{vec_type}_dictionary_{d}.json'

    with open(svd_dict_path) as file:
        svd_dict = json.load(file)

    data = []
    data_str = []
    for text in (texts):
        arr_words = text.split()
        for i in range(len(arr_words)-n+1):
            n_gram = []
            n_gram_str = []
            skip_ngram = False
            for word in arr_words[i:i+n]:
                if word not in svd_dict:
                    skip_ngram = True
                    break
                n_gram += svd_dict[word]
                n_gram_str.append(word)
            if skip_ngram == False:
                data.append(np.array(n_gram))
                data_str.append(tuple(n_gram_str))

    data = np.array(data)
    return data, data_str


def make_data(n, d, lang, data_type, vec_type):
    absolute_path = '/home/sshsubkhangulov/sultan_folder/data'
    texts_path = f'{absolute_path}/{lang}/{data_type}_corpus.txt'
    
    with open(texts_path, 'r') as file:
        texts = file.readlines()

    
    return make_data_from_texts(n, d, lang, texts, vec_type)
