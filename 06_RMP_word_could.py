# -*- coding: utf-8 -*-
# @Time    : 2019/1/12 下午5:26
# @Author  : mingchun liu
# @Email   : psymingchun@gmail.com
# @File    : 自然语言处理.py
# @Software: PyCharm

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

#######################################
# Without stemming
#######################################

#### ------------------------------ Importing Data -------------------------------

# dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t', quoting=3)
src = r'Hight_Professor_Tag.csv'

# dataset = pd.read_csv('Hight_Professor_Tag.csv',delimiter='\t', quoting=3)
dataset = pd.read_csv(src)

# print(dataset.head())
dataset.shape

# ----------------------------- Cleaning the Data -----------------------------

# Replacing special characters with white spaces.
review = re.sub('[^a-zA-Z]', ' ', dataset['tag_professor'][2])
print(review)

# Converting to lower case.
review = review.lower()
print(review)

# Creating word list by splitting.
review = review.split()
print(review)

# Removing stopwords.
# nltk.download('stopwords')
review = [word for word in review if not word in
                                         set(stopwords.words('english'))]
print(review)

# Stemming the words.
ps = PorterStemmer()
print(ps)
review = [ps.stem(word) for word in review if not word
                    in set(stopwords.words('english'))]
print(review)

# Joining the words back.
review = ' '.join(review)
print(review)

# Performing the same action in entire data and creating a corpus.

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['tag_professor'][i])
    # review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word
              in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

print(corpus)

#词干恢复
newcorpus = []
import nltk.stem.snowball as sb
# 导入nltk.stem用来词型还原
import nltk.stem as ns
sb_stemmer = sb.SnowballStemmer("english")  # 思诺博词干提取器
lemmatizer = ns.WordNetLemmatizer()
for word in corpus:
    # 将名词还原为单数形式
    n_lemma = lemmatizer.lemmatize(word, pos='a')
    # 将动词还原为原型形式
    v_lemma = lemmatizer.lemmatize(word, pos='v')
    print('%8s %8s %8s' % (word, n_lemma, v_lemma))
    newcorpus.append(n_lemma)

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()


new = ' '.join(newcorpus)
print('new',new)

new1 = wnl.lemmatize(new)
####--------------------------Visualization of dataset------------------------

stopwords = set(STOPWORDS)

backgroud_Image = plt.imread('cry2.jpg')
def show_wordcloud(data, title=None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=60,
        scale=1.5,
        collocations=False,
        mask=backgroud_Image,  # 设置背景图片
        #random_state=1  # chosen at random by flipping a coin; it was heads
        color_func=lambda *args, **kwargs: (94, 129, 186)
    ).generate(str(data))


    fig = plt.figure(1, figsize=(5, 5),dpi=100)
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

# show_wordcloud(dataset['tag'])
show_wordcloud(new1)

