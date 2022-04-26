import pandas as pd
import numpy as np
import re
from gensim.models import KeyedVectors


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

def get_embedLookup(true_path, fake_path, word_model_path):
    global  embed_lookup
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
