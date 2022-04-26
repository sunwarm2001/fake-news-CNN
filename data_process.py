import pandas as pd
import numpy as np
import re
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def cutsource(s):
    ''' a function to cut out news source in "true" texts
        luckily they are separated by '-' (dash sign)
    '''
    if '- ' in s:
        s1 = s.split('- ')[0]
        s = s[len(s1) + 2:]

    return s


def cutgetty(s):
    ''' a function to cut out 'getty images' in "fake" texts
    '''
    s = re.sub('Getty Images', '', s)

    return s


def cutfactbox(s):
    ''' a function to cut out 'factbox' in "true" titles
    '''
    s = re.sub('factbox', '', s, flags=re.IGNORECASE)

    return s

# 处理数据 —— 对数据进行清理
def get_embedLookup(true_path, fake_path, word_model_path):
    true = pd.read_csv(true_path)
    fake = pd.read_csv(fake_path)
    true['text'] = true['text'].astype(str).apply(cutsource)
    fake['text'] = fake['text'].astype(str).apply(cutgetty)
    true['title'] = true['title'].astype(str).apply(cutfactbox)
    # combine data into 1 dataframe, discarding 'date' and 'subject' fields,
    # removing rows with empty text or title.
    cols = ['title', 'text']
    df = pd.concat([fake[cols], true[cols]], ignore_index=True)
    df['text'] = df['text'].str.strip()
    df['title'] = df['title'].str.strip()
    label = len(fake)*['fake'] + len(true)*['true']
    df['label'] = label
    # drop news shorter than a tweet
    df = df[df['text'].str.len() > 280]
    df = df.replace('', np.nan)
    df.dropna(inplace=True)
    #example = df.iloc[42]
    #print(example['title'] + '\n' + example['text'] + '\n' + example['label'] + '\n')
    print(df)
    print("Class balance:\n{}".format(df['label'].value_counts()))

    # lowercase the texts and remove punctuation
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace('[^\w\s]','', regex=True) # remove punctuation (everything that's not a word(also a number) or whitespace)
    texts_split = df['text'].str.split().tolist()

    # seems like I have 1 letter words which usually don't carry much semantic significance
    # let's get rid of them
    for i, text in enumerate(texts_split):
        texts_split[i] = [word for word in text if len(word)>1]

    # one last preprocessing step is that I would like to cut the length of the texts to have a "light" model
    lens = [len(text) for text in texts_split]
    # I put max_len equal to 400 and cut all the texts up to this number
    max_len = 400
    for i, text in enumerate(texts_split):
        texts_split[i] = text[:max_len]
    print("\n",texts_split[42])


    # loading the model

    embed_lookup = KeyedVectors.load_word2vec_format(word_model_path, binary=True)
    print(len(embed_lookup), 'words in the vocabulary')
    word = 'news'
    # Tokenization: For each news text we represent words as their index in the lookup table
    # unknown words are represnted as 0s, i.e. spaces
    tokenized_news = []
    for text in texts_split:
        ints = []
        for word in text:
            try:
                idx = embed_lookup.key_to_index[word]
            except:
                idx = 0
            ints.append(idx)
        tokenized_news.append(ints)
    print('\n An example of a tokenized text: \n', tokenized_news[42])

    pttexts = np.zeros((len(tokenized_news), max_len), dtype=int)
    for i, tok_text in enumerate(tokenized_news):
        pttexts[i, -len(tok_text):] = tok_text
    # converting labels into 0s and 1s: 0 for true and 1 for fake
    # checking that we haven't lost any news
    labels = np.array([0 if label == 'true' else 1 for label in df['label']])
    print("\n padded and tokenized first 11 texts up to first 10 words \n", pttexts[:11, :10])
    print(len(labels), len(pttexts))

    np.save(r'E:\office应用\毕业设计\fake-news-CNN\processed_data\pttexts.npy',pttexts)
    np.save(r'E:\office应用\毕业设计\fake-news-CNN\processed_data\labels.npy',labels)
    print("存储完成")
    # split data into training, validation, and test data (tokenized+padded texts and labels, x and y)
    #split_frac = 0.8

def split_data(train_size, test_size, random_state):
    pttexts = np.load(r'E:\office应用\毕业设计\fake-news-CNN\processed_data\pttexts.npy')
    labels = np.load(r'E:\office应用\毕业设计\fake-news-CNN\processed_data\labels.npy')
    train_x, rem_x, train_y, rem_y = train_test_split(pttexts, labels, train_size=train_size, random_state=random_state)
    val_x, test_x, val_y, test_y = train_test_split(rem_x, rem_y, test_size=test_size, random_state=random_state)

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    print("\t\t\tDatasets Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))

    print('\nTest Y balance: {:.3f}'.format(np.sum(test_y)/len(test_y)))
    # dataloaders
    batch_size = 64

    # shuffling and batching data
    train_loader = DataLoader(train_data, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    np.save(r'E:\office应用\毕业设计\fake-news-CNN\processed_data\train_loader.npy', train_loader)
    np.save(r'E:\office应用\毕业设计\fake-news-CNN\processed_data\valid_loader.npy', valid_loader)
    np.save(r'E:\office应用\毕业设计\fake-news-CNN\processed_data\test_loader.npy', test_loader)