# -*- coding: utf-8 -*-
# @Time    : 2019/7/20 下午9:42
# @Author  : mingchun liu
# @Email   : psymingchun@gmail.com
# @File    : RMP评论情感分析.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re

def rmp_sentment_analysis(src):
    df = pd.read_csv(src, usecols=['professor_name','school_name', 'star_rating','comments'])
    high_professor_comment = df[(df['star_rating'] >= 4.0) & (df['star_rating'] <= 5.0)]['comments'].sample(10000).dropna().tolist()
    text = ' '.join(high_professor_comment)
    tokens = [t.lower() for t in re.split(r'[^\w\s]|\s', text) if t != '']
    print(tokens)

    sr = stopwords.words('english')
    add_stopword = ['him.','took', 'one','took','day']
    sr = sr + add_stopword
    clean_tokens = tokens[:]
    for token in tokens:
        if token in stopwords.words('english'):
            clean_tokens.remove(token)

    freq = nltk.FreqDist(clean_tokens)
    for key, val in freq.items():
        print(str(key) + ':' + str(val))
    freq.plot(20, cumulative=False)


    # https://github.com/cjhutto/vaderSentiment#about-the-scoring
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # 高分教授的情感
    high_professor_sentiment = []
    pos_num = 0
    neg_num = 0
    for sentence in high_professor_comment:
        vs = analyzer.polarity_scores(sentence)
        # print("{:-<65} {}".format(sentence, str(vs)))
        # print(vs['compound'])
        high_professor_sentiment.append(vs['compound'])
        if vs['compound'] >= 0.05:
            print('positive')
            pos_num += 1
        else:
            print('negtive')
            neg_num += 1
    print(pos_num, neg_num)
    ratio = pos_num/(pos_num+neg_num)
    print('ratio:',ratio)
    high_mean = np.mean(high_professor_sentiment)
    high_std = np.std(high_professor_sentiment)
    print('平均数：',high_mean,'标准差',high_std)


    low_professor_comment = df[(df['star_rating'] >= 1.0) & (df['star_rating'] <= 2.0)]['comments'].sample(10000).dropna().tolist()
    # 低分教授
    print('低分教授')
    low_professor_sentiment = []
    low_pos_num = 0
    low_neg_num = 0
    for sentence in low_professor_comment:
        vs = analyzer.polarity_scores(sentence)
        # print("{:-<65} {}".format(sentence, str(vs)))
        # print(vs['compound'])
        low_professor_sentiment.append(vs['compound'])
        if vs['compound'] <= -0.05:
            # print('positive')
            low_pos_num += 1
        else:
            # print('negtive')
            low_neg_num += 1
    print(low_pos_num, low_neg_num)
    low_ratio = low_pos_num/(low_pos_num+low_neg_num)
    print('ratio:',low_ratio)
    low_mean = np.mean(low_professor_sentiment)
    low_std = np.std(low_professor_sentiment)
    print('平均数：',low_mean,'标准差',low_std)

if __name__ == "__main__":
    src = r'/Users/liumingchun/【1】科研+实验室/3-科研项目/RateMyProfessor/bigdata/ratemyprofessor.csv'
    rmp_sentment_analysis(src)





