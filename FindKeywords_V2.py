from nltk import ngrams
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize,sent_tokenize
import re
from collections import Counter
from nltk.corpus import stopwords

import pandas as pd
def tokenize(text): 
    return re.findall(r'\w+', text.lower())

tokenized_words = Counter(tokenize(open('train.txt').read()))

def probability(word, N=sum(tokenized_words.values())): 
    return tokenized_words[word] / N

def known(words): 
    return set(w for w in words if w in tokenized_words)

def edit_dist_1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    #Split the word in two halves in all possible ways
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    #Edit distance for deletion
    deletes    = [left + right[1:]               for left, right in splits if right]
    #Edit distance based on transposition
    transposes = [left + right[1] + right[0] + right[2:] for left, right in splits if len(right)>1]
    #Edit distance based on replacing
    replaces   = [left + c + right[1:]           for left, right in splits if right for c in letters]
    #Edit Distance based on insertion
    inserts    = [left + c + right               for left, right in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edit_dist_2(word): 
    #Find words with edit distance : 2
    return (e2 for e1 in edit_dist_1(word) for e2 in edit_dist_1(e1))

def find(text,keyword):
    #Find Keywords in the short text
    words=tokenize(text)
    if(keyword in words):
        return True
    
def preprocess(text):
    preprocessed_text=''
    for word in word_tokenize(text):
        if(word in stop_words or word in delimiters):
            continue
        if(word in tokenized_words):
            preprocessed_text+=word+' '
        else:
            try:
                keyword_2=list(known(edit_dist_2(word)))[0]
            except:
                keyword_2=word
            preprocessed_text+=keyword_2+' '
    return preprocessed_text

stop_words=stopwords.words('english')
delimiters=['.',',',' ','\t']
df=pd.read_csv('output.csv')
text_list_temp=df.loc[:,'Reviews'].tolist()
#Preprocess convert misspelled words
text_list=[]
for text in text_list_temp:
    text_list.append(preprocess(text))

keywords_file=open('keywords.txt')
keywords_checked=[]
for words in keywords_file.readlines():
    keywords_checked.append(words.strip())    
keywords_file.close()

negative_words=['no','not']
keywords=[]
words=[]
temp_keywords=[]

for text in text_list:
    sentences=sent_tokenize(text)
    for sentence in sentences:
        words.append(word_tokenize(sentence))
    
    for temp_sent in words:
        #For User given Keywords
        for temp_keys in keywords_checked:
            if(temp_keys in temp_sent):
                temp_keywords.append(temp_keys)
        trigrams=ngrams(temp_sent,3)
        #For extracted keywords
        for grams in trigrams:
            f=1
            for j in negative_words:
                if(j in grams):
                    f=0
                    break
                else:
                    #print(j)
                    for word,tag in pos_tag(list(grams)):
                        if((tag=='NN' or tag=='JJ') and word not in temp_keywords):
                            temp_keywords.append(word)
            if(f==0):
                break
    keywords.append(temp_keywords)

for textwise_keywords in keywords:
    print(textwise_keywords)
