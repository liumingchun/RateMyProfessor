#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Liu Mingchun
# @date：2018/10/06

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import re
from wordcloud import WordCloud


src = r'/Users/liumingchun/【1】科研+实验室/3-科研项目/RateMyProfessor/bigdata/ratemyprofessor.csv'

def tagProfessorAnalysis():
    ''' 1.分析助理教授，副教授，教授的TAG的词频并卡方检验 assistant professor, associate professor and full professor'''
    df = pd.read_csv(src, usecols=['professor_name','school_name', 'year_since_first_review', 'tag_professor'])
    assistant_professor = df[(df['year_since_first_review'] >= 0) & (df['year_since_first_review'] <= 6)].drop_duplicates(['professor_name','school_name'], 'first', False)
    associate_professor  = df[(df['year_since_first_review'] < 12) & (df['year_since_first_review'] > 6)].drop_duplicates('professor_name', 'first', False)
    full_professor  = df[(df['year_since_first_review'] >= 12)].drop_duplicates('professor_name', 'first', False)
    # print(assistant_professor)
    # print(associate_professor)
    # print(full_professor)

    # assistant_professor TAG
    assistant_professor_list = assistant_professor['tag_professor'].dropna().tolist()
    assistant_professor_tag=[]
    for i in assistant_professor_list:
        pattern = re.split('\(\d+\)', i.lower())
        for j in pattern:
            if j != '':
                assistant_professor_tag.append(j.strip())
    print(assistant_professor_tag)


    # associate TAG
    associate_professor_list = associate_professor['tag_professor'].dropna().tolist()
    associate_professor_tag = []
    for i in associate_professor_list:
        pattern = re.split('\(\d+\)', i.lower())
        for j in pattern:
            if j != '':
                associate_professor_tag.append(j.strip())

    # Full TAG
    full_professor_list = full_professor['tag_professor'].dropna().tolist()
    full_professor_tag = []
    for i in full_professor_list:
        pattern = re.split('\(\d+\)', i.lower())
        for j in pattern:
            if j != '':
                full_professor_tag.append(j.strip())

    colu ={'assistant_professor': assistant_professor_tag, 'associate_professor':associate_professor_tag, 'full_professor':full_professor_tag}
    df2 = pd.DataFrame().from_dict(colu, orient='index')
    df2 = df2.T

    # 计算TAG次数和频率
    df3 =pd.DataFrame()
    df3['assistant_professor_count'] = df2['assistant_professor'].value_counts()
    df3['associate_professor_count'] = df2['associate_professor'].value_counts()
    df3['full_professor_count'] = df2['full_professor'].value_counts()

    df3['assistant professor frequency'] = df3['assistant_professor_count'] / df3['assistant_professor_count'].sum()
    df3['associate professor frequency'] = df3['associate_professor_count'] / df3['associate_professor_count'].sum()
    df3['full professor frequency'] = df3['full_professor_count'] / df3['full_professor_count'].sum()

    # 对TAG卡方检验
    p = stats.chisquare(f_obs=df3[['assistant_professor_count', 'associate_professor_count', 'full_professor_count']], axis=1)
    # print('卡方值："%s,"，p值：%s' % p, end='')
    df3['Chi-Square'] = np.array(p[0])
    df3['p-value'] = np.array(p[1])
    print(df3)
    df3.to_csv('三种教授tag分析.csv')

def tagAnalysis():
    ''' 2.分析高分组和低分组教授TAG的词频并卡方检验'''
    df = pd.read_csv(src, usecols=['professor_name','school_name', 'star_rating', 'tag_professor'])
    high_rating_professor = df[(df['star_rating'] >= 4) & (df['star_rating'] <= 5)].drop_duplicates(['professor_name','school_name'], 'first', False) # 高分组4-5星
    low_rating_professor  = df[(df['star_rating'] <= 2) & (df['star_rating'] >= 1)].drop_duplicates('professor_name', 'first', False) # 低分组1-2星
    # print(high_rating_professor)

    # 高分组TAG
    high_rate_list = high_rating_professor['tag_professor'].dropna().tolist()
    new_high_list=[]
    for i in high_rate_list:
        pattern = re.split('\(\d+\)',i.lower())
        new_high_list += pattern
    new_high_list = [x.strip() for x in new_high_list if x != '']
    # print(new_high_list)

    # 低分组TAG
    low_rate_list =low_rating_professor['tag_professor'].dropna().tolist()
    new_low_list = []
    for i in low_rate_list:
        pattern = re.split('\(\d+\)', i.lower())
        new_low_list += pattern
    new_low_list = [x.strip() for x in new_low_list if x != '']
    # print(new_low_list)

    colu ={'High_rate_professors':new_high_list, 'Low_rate_professors':new_low_list}
    df2 = pd.DataFrame().from_dict(colu, orient='index')
    df2 = df2.T
    # print(df2)

    # 计算TAG次数和频率
    df3 =pd.DataFrame()
    df3['High rate count'] = df2['High_rate_professors'].value_counts()
    df3['Low rate count'] = df2['Low_rate_professors'].value_counts()
    df3['High rate frequency'] = df3['High rate count'] / df3['High rate count'].sum()
    df3['Low rate frequency'] = df3['Low rate count'] / df3['Low rate count'].sum()

    # 对TAG卡方检验
    p = stats.chisquare(f_obs=df3[['High rate count', 'Low rate count']], axis=1)
    # print('卡方值："%s,"，p值：%s' % p, end='')
    df3['Chi-Square'] = np.array(p[0])
    df3['p-value'] = np.array(p[1])
    print(df3)
    df3.to_csv('tag分析.csv')

# 以上两个题目的TAG词云
def wordcloud():
    df = pd.read_csv('Hight_Professor_Tag.csv', nrows=100)
    list = df['tag_professor'].dropna().tolist()
    print(list)

    text = ' '.join(str(x) for x in list).strip()
    text = text.lower()
    tag = re.sub('[^a-zA-Z]', ' ', text)
    print(tag)
    newlist = tag.split(' ')
    print(newlist)
    newlist1 = []
    for i in newlist:
        if i:
            newlist1.append(i)
    print(newlist1)

    newtext = ' '.join(newlist1)

    from scipy.misc import imread
    b_mask = imread('sad1.png')
    wc = WordCloud(
        background_color='white',  # 设置背景颜色
        max_font_size=80,  # 设置字体最大值
        max_words=200,
        scale=5, # 设置清晰度
        # random_state=40, # 设置有多少种随机生成状态，即有多少种配色方案
        collocations = False, # 添加这行代码，会把good去除，是否包括两个词的搭配
        color_func=lambda *args, **kwargs: (94, 129, 186),
        mask=b_mask

    )
    wc.generate_from_text(newtext)
    # plt.figure(dpi=300)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.figure(1, figsize=(5, 5), dpi=300)
    plt.show()
    plt.pause(0)

# 计算课程评分 和 难度、是否再次选课的关系
def corr():
    df = pd.read_csv(src, usecols=['professor_name','school_name', 'star_rating', 'diff_index', 'take_again'])
    df = df[(df['star_rating'] >= 1.0 ) & (df['diff_index'] >= 1.0)].drop_duplicates(['professor_name','school_name'], 'first', False)

    # 计算评分和难度的回归方程
    print('计算diff_index和star_rating的回归方程和R方')
    data = df[['diff_index', 'star_rating']]
    regression = stats.linregress(data)
    print("R square：", regression[2] ** 2)
    print('线性回归方程是 Y= %.3fX + %.3f,rvalue是%.3f,p-values是%s,标准误是%s' % regression)
    g = sns.set("paper",font_scale =1.3)
    g = sns.set_style("white")
    # 随机取1000个点
    data = data.sample(1000)
    data = data.rename(index=str, columns={'diff_index': 'Difficulty Index', 'star_rating': 'Star Rating'})
    g = sns.jointplot('Difficulty Index','Star Rating', data=data,height=6,ratio=7, kind="reg", xlim=(1,5), ylim=(1.0, 5.0),space=0,color='b')
    plt.text(1.2, 1.5, r"Y=-0.50X+5.18", fontsize=12)
    plt.text(1.2, 1.2, r'$R^2$=0.20', fontsize=12)
    plt.show()


    # 计算评分和是否再次选课
    df = df.dropna(0)
    data2 = df[['star_rating', 'take_again']]
    data2['take_again'] = data2['take_again'].str.strip('%').astype(float) / 100
    # 计算回归方程
    regression2 = stats.linregress(data2)
    print("R square：", regression2[2] ** 2)
    print('评分和是否再次选课的线性回归方程是 Y= %.3fX + %.3f,rvalue是%.3f,p-values是%s,标准误是%s' % regression2)
    data2 = data2.sample(1000)
    data2 = data2.rename(columns={'star_rating': 'Star Rating','take_again':'Would Take Again'})
    g = sns.jointplot("Star Rating", "Would Take Again", data=data2, kind="reg", ylim=(0, 1),xlim=(1.0, 5.0), color='b', space=0, ratio=7, height=6)
    plt.text(1.5, 0.85, r"Y=0.26X-0.30", fontsize=12)
    plt.text(1.5, 0.8, r'$R^2$=0.64', fontsize=12)
    plt.show()

    # # 计算所有变量的相关系数矩阵
    # p2 = stats.spearmanr(df[['star_rating','diff_index','prodessor_take_again','num_student', 'student_star','student_difficult','len_comment','word_comment','help_useful','help_not_useful']])
    # print(p2)

#5。计算take Again和评分的关系
def take_again_corr():
    df = pd.read_csv(src)
    # print(df.info())
    df = df[(df['star_rating'] >= 1.0) & (df['diff_index'] >= 1.0)]
    # df = df.sample(1000) #随机选取数据
    df = df.dropna(axis=0)
    star_rating = df['star_rating']
    take_again = df['prodessor_take_again']
    print('take_again 数据条目:', take_again.count())
    print('star_rating数据条目:', star_rating.count())


    # print('～～～～take_again与star_rating～～～～')
    # p = stats.spearmanr(df[['prodessor_take_again', 'star_rating',]])
    # print('take_again与star_rating的相关系数是：%.4f, p值是：%d' % p)
    # print(p)

    # # 计算回归方程
    # regression2 = stats.linregress(df[['prodessor_take_again','star_rating']])
    # print(regression2)
    # print("R square", regression2[2] ** 2)
    # R_square = regression2[2] ** 2
    # print('线性回归方程是 Y= %.3fX + %.3f,rvalue是%.3f,pvalus是%s,标准误是%s' % regression2)

    g = sns.set("paper", font_scale=1.3)
    g = sns.set_style("white")
    # 核密度图
    data = df.rename(index=str, columns={'prodessor_take_again': 'Would Take Again', 'star_rating': 'Star Rating'})
    data = data.sample(10000)
    g = sns.jointplot('Would Take Again', 'Star Rating', data=data, height=6, ratio=7, kind="kde", xlim=(0, 1),
                      ylim=(1.0, 5.0), space=0,
                      color='b',
                      cbar_kws=dict(use_gridspec=False, location="right", anchor=(1.5, 0), shrink=0.9, ticks=None, ),
                      cbar=True, )

    # print(dfSample.info())
    # # 回归
    # g = sns.jointplot(x=dfSample["Star Rating"], y=dfSample["Would Take Again"], data=df, kind="reg", xlim=(1, 5),
    #                   ylim=(0, 1), space=0.1, ratio=3)
    # # #评分和是否在选的回归线及R方
    plt.text(-2.5,1.4, r"Y=2.40X+2.14",fontsize=12)
    plt.text(-2.5,1.2, r'$R^2$=0.64',fontsize=12)
    plt.pause(0)  #防止图闪退

# 是否为了分数
def for_cerdis():
    df = pd.read_csv(src, usecols=['student_star', 'for_credits'])
    # df = df[(df['star_rating'] >= 1.0) & (df['diff_index'] >= 1.0) & (df['for_credits'] == 'Yes') & (df['for_credits'] == 'No')]
    print(df.info())
    df = df.dropna(axis=0)
    df = df.rename(index=str, columns={'for_credits': 'For Credits', 'student_star': 'Star Rating Given by Students'})
    g = sns.set("paper",font_scale =1.3)
    g = sns.set_style("white")
    g = sns.catplot(x="For Credits", y="Star Rating Given by Students", kind='box', width=0.2,
                    data=df[['Star Rating Given by Students', 'For Credits']], order=["Yes", "No"])
    # g = sns.catplot(x="attence", y="student_star",kind='box' , data=df[['student_star', 'attence']])
    plt.show()

def mannwhitneyu():
    print('manteniU检验，是不是为了学分～～～～～～～～～～')
    from math import sqrt
    df = pd.read_csv(src, usecols=['student_star', 'for_credits','attence'])
    df2 = df[(df['for_credits'] == 'Yes') & (df['student_star'] >= 1.0 )] # yes分组
    df3 = df[(df['for_credits'] == 'No') & (df['student_star'] >= 1.0 )]  # no分组

    p = stats.mannwhitneyu(df2['student_star'],df3['student_star'],alternative='two-sided')
    print(p)

    yes_mean = df2['student_star'].mean()  # 平均数
    yes_std = df2['student_star'].std()  # 标准差
    no_mean = df3['student_star'].mean()  # 平均数
    no_std = df3['student_star'].std()  # 标准差
    #test conditions
    cohens_d = (yes_mean - no_mean) / (sqrt((yes_std ** 2 + no_std ** 2) / 2))
    print(cohens_d)

#第5题，比较评论单词字数和长度
def len_comment():
    src1 = r'/Users/liumingchun/【1】科研+实验室/3-科研项目/RateMyProfessor/bigdata/ratemyprofessor1.csv'
    print('评论长度比较～～～～～～～～～～')
    df = pd.read_csv(src1, usecols=['star_rating', 'len_comment','comments'])
    df = df.drop_duplicates(['comments'], 'first', False)
    # 按照1-2星及4-5星分为2组，然后进行比较两者差异，独立样本T检验，方差齐性检验
    df2 = df[(df['star_rating'] <= 2) & (df['star_rating'] >= 1)]  # 低分组
    df3 = df[(df['star_rating'] >= 4) & (df['star_rating'] <= 5)]  # 高分组

    low_len_comment = df2['len_comment']
    high_len_comment= df3['len_comment']

    #方差齐性检验，看是否满足检验要求
    leven_test = stats.levene(high_len_comment,low_len_comment,center='median')
    print('len_comment的方差齐性检验是：%s, p值是：%s' % leven_test)

    low_mean = df2['len_comment'].mean() #平均数
    low_std = df2['len_comment'].std()   #标准差
    low_size = df2['len_comment'].count()#数量

    high_mean = df3['len_comment'].mean() #平均数
    high_std = df3['len_comment'].std()   #标准差
    high_size = df3['len_comment'].count()#数量

    low_rvs  =stats.norm.rvs(loc=low_mean,scale= low_std,size = low_size)
    high_rvs = stats.norm.rvs(loc=high_mean,scale=high_std,size =high_size)

    p = stats.ttest_ind(high_rvs,low_rvs,equal_var=True)
    print('高分组的len_comment平均数是%f，标准差是%f,样本量是%s' % (high_mean,high_std,high_size))
    print('低分组的len_comment平均数是%f，标准差是%f,样本量是%s' % (low_mean,low_std,low_size))
    print('len两组的T检验是：%s，p值是：%s' % p)


    from math import sqrt
    # test conditions
    cohens_d = (high_mean - low_mean) / (sqrt((high_std ** 2 + low_std ** 2) / 2))
    print('len_comment科恩d值',cohens_d)


    # 评价词云
    # list = df3['professor_all_comments'].dropna().tolist()
    # text = ' '.join(str(x) for x in list).strip()
    # text = text.lower()
    # print(text)
    #
    # wc = WordCloud(
    #     background_color='white',  # 设置背景颜色
    #     max_font_size=150,  # 设置字体最大值
    #     random_state=30  # 设置有多少种随机生成状态，即有多少种配色方案
    # )
    # wc.generate_from_text(text)
    # plt.figure()
    # plt.imshow(wc, interpolation="bilinear")
    # plt.axis("off")
    # plt.show()
    # plt.pause(0)

def word_comment():
    src1 = r'/Users/liumingchun/【1】科研+实验室/3-科研项目/RateMyProfessor/bigdata/ratemyprofessor1.csv'
    print('评论单词个数比较～～～～～～～～～～')
    df = pd.read_csv(src1, usecols=['star_rating', 'word_comment','comments'])
    df = df.drop_duplicates(['comments'], 'first', False)

    # 按照1-2星及4-5星分为2组，然后进行比较两者差异，独立样本T检验
    high_words_comments = df[(df['star_rating'] <= 5.0) & (df['star_rating'] >= 4.0)] # 低分组
    low_words_comments = df[(df['star_rating'] >= 1.0) & (df['star_rating'] <= 2.0)]  # 高分组

    high_words_comment = high_words_comments['word_comment']
    low_words_comment =  low_words_comments['word_comment']

    #方差齐性检验
    leven_words_test = stats.levene(high_words_comment,low_words_comment,center='median')
    print('words的方差齐性检验%s, p值是：%s' % leven_words_test)

    high_words_mean = high_words_comment.mean()  # 平均数
    high_words_std = high_words_comment.std()  # 标准差
    high_words_size = high_words_comment.count()  # 数量

    low_words_mean = low_words_comment.mean()  # 平均数
    low_words_std = low_words_comment.std()  # 标准差
    low_words_size = low_words_comment.count()  # 样本

    low_words_rvs =stats.norm.rvs(loc=low_words_mean,scale= low_words_std,size = low_words_size)
    high_words_rvs = stats.norm.rvs(loc=high_words_mean,scale=high_words_std,size =high_words_size)
    p_words = stats.ttest_ind(high_words_rvs,low_words_rvs,equal_var=True)

    print('高分组的words_comment 平均数是%f，标准差是%f,样本数%s' % (high_words_mean,high_words_std,high_words_size))
    print('低分组的words_comment 平均数是%f，标准差是%f,样本数%s' % (low_words_mean,low_words_std,low_words_size))
    print('words_comment两组T检验是%s,p值是%s ' % p_words)

    from math import sqrt
    # test conditions
    cohens_d = (low_words_mean - high_words_mean) / (sqrt((high_words_std ** 2 + low_words_std ** 2) / 2))
    print('words_comment科恩d值：',cohens_d)


def word_comment_kruskal():
    print('评论单词个数比较～～～～～～～～～～')
    df = pd.read_csv(src, nrows=1000)
    print(df.info())
    df1 = df[(df['star_rating'] <= 2) & (df['star_rating'] >= 1)]  # 低分组
    df2 = df[(df['star_rating'] <= 3) & (df['star_rating'] >= 2)]  # 低分组
    df3 = df[(df['star_rating'] <= 4) & (df['star_rating'] >= 3)]  # 低分组
    df4 = df[(df['star_rating'] >= 4) & (df['star_rating'] <= 5)]  # 高分组
    print(df1)

    p = stats.f_oneway(df1['word_comment'],df2['word_comment'],df3['word_comment'],df4['word_comment'])
    print(p)


def one_way_anove():
    df = pd.read_csv(src, usecols=['professor_name', 'school_name','year_since_first_review', 'star_rating']).drop_duplicates(['professor_name','school_name'], 'first', False)
    # print(df)

    assistant_professor  = df[(df['year_since_first_review'] >= 6)].drop_duplicates('professor_name', 'first', False)
    associate_professor  = df[(df['year_since_first_review'] > 6) & (df['year_since_first_review'] < 12)].drop_duplicates('professor_name','first', False)
    full_professor  = df[(df['year_since_first_review'] >= 12)].drop_duplicates('professor_name', 'first', False)

    # 方差齐性检验
    leven_words_test = stats.levene(assistant_professor['star_rating'], associate_professor['star_rating'], full_professor['star_rating'], center='median')
    print('words的方差齐性检验%s, p值是：%s' % leven_words_test)
    bartellrt = stats.bartlett(assistant_professor['star_rating'], associate_professor['star_rating'], full_professor['star_rating'])
    print('sdsd',bartellrt)
    if bartellrt[1] <= 0.5:
        print('小于0.5,不齐性')

    p = stats.f_oneway(assistant_professor['star_rating'], associate_professor['star_rating'], full_professor['star_rating'])
    print(p)


if __name__ == '__main__':

    # tagProfessorAnalysis() # 计算助理教授、副教授和终身教授的tag
    # tagAnalysis() #计算高分组和低分组教授的tag
    # wordcloud() # 以上两个题目的词云分析
    # corr() # 计算难度和评分的关系，计算所有变量的相关矩阵
    # for_cerdis()
    # word_comment()
    # len_comment()
    # mannwhitneyu()
    one_way_anove()