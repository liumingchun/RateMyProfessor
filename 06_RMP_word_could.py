# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 7:33 PM
# @Author  : mingchun liu
# @Email   : psymingchun@gmail.com
# @Software: PyCharm

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


def word_cloud():
    df = pd.read_csv(src, nrows=1000)
    df = df.dropna(0)
    list = df['tag_professor'].dropna().tolist()
    # print(list)
    text = ' '.join(str(x) for x in list).strip()
    text = text.lower()
    tag = re.sub('[^a-zA-Z]', ' ', text)
    print(tag)
    newlist = tag.split(' ')
    # print(newlist)
    newlist1 = []
    for i in newlist:
        if i:
            newlist1.append(i)
    # print(newlist1)

    # Removing stopwords.
    nltk.download('stopwords')
    review = [word for word in newlist1 if not word in
                                             set(stopwords.words('english'))]
    # print(review)

    # Stemming the words.
    ps = PorterStemmer()
    print(ps)
    review = [ps.stem(word) for word in review if not word
                        in set(stopwords.words('english'))]
    print(review)
    # Joining the words back.
    review = ' '.join(review)
    print(review)

    backgroud_Image = plt.imread('cry2.jpg')
    wc = WordCloud(
        background_color='white',  # 设置背景颜色
        max_font_size=80,  # 设置字体最大值
        max_words=200,
        scale=5, # 设置清晰度
        # random_state=40, # 设置有多少种随机生成状态，即有多少种配色方案
        collocations = False, # 添加这行代码，会把good去除，是否包括两个词的搭配
        color_func=lambda *args, **kwargs: (94, 129, 186),
        mask=backgroud_Image

    )
    wc.generate_from_text(review)

    plt.xlabel("The Word Cloud of High Rating Professor", fontsize=12)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.figure(1, figsize=(10, 10), dpi=300)
    plt.show()

if __name__ == "__main__":
    src = r'Low_Professor_Tag.csv' # 对高分教授、低分教授；助理、副教授、教授的tag进行可视化
    word_cloud()
