#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Liu Mingchun
# @date：2018/10/06

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import re
from wordcloud import WordCloud

src = r'/Users/liumingchun/【1】科研+实验室/3-科研项目/RateMyProfessor/bigdata/ratemyprofessor.csv'

def TagProfessorAnalysis():
    ''' 1.分析助理教授，副教授，教授的TAG的词频并卡方检验 assistant professor, associate professor and full professor'''
    df = pd.read_csv(src, usecols=['professor_name','school_name', 'year_since_first_review', 'tag_professor'])

    # 去除重复教授
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

    # 对TAG卡方检验，此方法存在问题
    # https: // blog.csdn.net / qq_38214903 / article / details / 82967812
    # p = stats.chisquare(f_obs=df3[['assistant_professor_count', 'associate_professor_count', 'full_professor_count']], axis=1)
    # # print('卡方值："%s,"，p值：%s' % p, end='')
    # df3['Chi-Square'] = np.array(p[0])
    # df3['p-value'] = np.array(p[1])
    print(df3)
    df3.to_csv('three_kind_professor_tags.csv', index = False)

    # 官方文档：https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
    chi2_contingency = stats.chi2_contingency(
        [[116682, 72203, 33979], [105073, 62446, 29567], [94462, 63165, 31738], [94385, 61028, 26614],
              [93568, 59697, 26108], [92607, 69498, 31629], [90521, 49743, 19863], [80107, 54653, 24821],
              [73406, 40653, 18475], [72131, 43760, 20300], [70649, 46349, 19737], [69475, 43136, 20567],
              [68422, 39229, 18229], [64967, 24334, 13102], [50452, 24662, 8355], [44178, 19235, 9911],
              [40735, 16549, 7531], [33070, 17774, 7840], [26663, 16439, 6838], [23495, 11992, 4905], ])
    print(chi2_contingency)

def High_low_professor_tag_analysis():
    ''' 2.分析高分组和低分组教授TAG的词频并卡方检验 higt professor, low professor '''
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

    # # 对TAG卡方检验
    # p = stats.chisquare(f_obs=df3[['High rate count', 'Low rate count']], axis=1)
    # # print('卡方值："%s,"，p值：%s' % p, end='')
    # df3['Chi-Square'] = np.array(p[0])
    # df3['p-value'] = np.array(p[1])
    # print(df3)
    df3.to_csv('high_low_professor_tag.csv', index = False)

    # 卡方检验
    chi2_contingency = stats.chi2_contingency(
        [[142004, 2855], [129289, 1394],
         [128981, 2409], [96900, 2635], [96640, 6788],
         [91958, 730], [91801, 1026], [83990, 1525], [81630, 10595],
         [66802, 10339], [65814, 17274], [63103, 2160], [54658, 10528],
         [50312, 10066], [47347, 2040], [29688, 6061], [26943, 3862], [21286, 6248], [17816, 4063], [15863, 3036]])
    print(chi2_contingency)


# 计算课程评分 和 难度、是否再次选课的关系
def corr_rating_difficulty():
    df = pd.read_csv(src,nrows=2000000, usecols=['professor_name','school_name', 'star_rating', 'diff_index'])
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
    # data = data.sample(1000)
    data = data.rename(index=str, columns={'diff_index': 'Difficulty Index', 'star_rating': 'Star Rating'})
    # g = sns.jointplot('Difficulty Index','Star Rating', data=data,height=6,ratio=7, kind="reg", xlim=(1,5), ylim=(1.0, 5.0),space=0,color='b')
    g = sns.jointplot('Difficulty Index','Star Rating', data=data,height=6,ratio=7, kind="kde", xlim=(1,5), ylim=(1.0, 5.0),space=0,color='b')

    plt.text(-2.8, 1.5, r"Y=-0.50X+5.18", fontsize=12)
    plt.text(-2.8, 1.2, r'$R^2$=0.20', fontsize=12)
    plt.show()

# 计算评分和是否再次选课
def corr_rating_take_again():
    df = pd.read_csv(src, usecols=['professor_name', 'school_name', 'star_rating', 'diff_index','take_again'])
    pd.set_option('display.max_rows', 100, 'display.max_columns', 1000, "display.max_colwidth", 1000, 'display.width',1000)
    df = df[(df['star_rating'] >= 1.0) & (df['diff_index'] >= 1.0)].drop_duplicates(['professor_name', 'school_name'],'first', False)
    df = df.replace(
        r'<span class="would-take-again">Would Take Again: <span class="response">N/A</span></span>\\r\\n                <span class="would-take-again">Would Take Again: <span class="response">N/A</span></span>\\r\\n                <span class="would-take-again">Would Take Again: <span class="response">N/A</span></span>\\r\\n                ',np.nan,regex=True)

    df['professor_take_again'] = df['take_again'].str.strip('%').astype(float) / 100  # 百分数转化成小数

    df = df.dropna(0)
    print(df.info())
    # 计算回归方程
    regression2 = stats.linregress(df[['professor_take_again','star_rating']])
    print(regression2)
    print("R square", regression2[2] ** 2)
    R_square = regression2[2] ** 2
    print('线性回归方程是 Y= %.3fX + %.3f,rvalue是%.3f,pvalus是%s,标准误是%s' % regression2)
    g = sns.set("paper", font_scale=1.3)
    g = sns.set_style("white")
    # 核密度图
    data = df.rename(index=str, columns={'professor_take_again': 'Would Take Again', 'star_rating': 'Star Rating'})
    g = sns.jointplot('Star Rating','Would Take Again', data=data, height=6, ratio=7, kind="kde", xlim=(1, 5),
                      ylim=(0, 1.0), space=0, color='b')

    plt.text(-4.2, 0.2, r"Y=2.45X+2.10", fontsize=12)
    plt.text(-4.2, 0.1, r'$R^2$=0.64', fontsize=12)
    plt.show()


def all_corr():
    df = pd.read_csv(src)
    pd.set_option('display.max_rows', 100, 'display.max_columns', 1000, "display.max_colwidth", 1000, 'display.width',1000)

    df = df.replace(r'<span class="would-take-again">Would Take Again: <span class="response">N/A</span></span>\\r\\n                <span class="would-take-again">Would Take Again: <span class="response">N/A</span></span>\\r\\n                <span class="would-take-again">Would Take Again: <span class="response">N/A</span></span>\\r\\n                ',
        np.nan, regex=True)
    df['professor_take_again'] = df['take_again'].str.strip('%').astype(float) / 100  # 百分数转化成小数
    df = df.dropna(0)

    print(df.info())
    # 计算所有变量的相关系数矩阵
    p = stats.spearmanr(df[['star_rating','diff_index','year_since_first_review','professor_take_again','num_student', 'student_star','student_difficult','help_useful','help_not_useful']])
    print(p)

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


# 是否为了分数, 强制及分数分布
def for_cerdis():
    df = pd.read_csv(src, usecols=['student_star', 'for_credits',"attence", 'grades'])
    df = df.dropna(axis=0)
    print(df.info())
    df = df.rename(index=str, columns={'for_credits': 'For Credits', 'student_star': 'Star Rating Given by Students',"attence":"Attence",'grades':'Student Grade'})
    g = sns.set("paper",font_scale =1.3)
    g = sns.set_style("white")

    # # 是否为了学分
    # g = sns.catplot(x="For Credits", y="Star Rating Given by Students", kind='box', width=0.2,
    #                 data=df[['Star Rating Given by Students', 'For Credits']], order=["Yes", "No"])
    # # 是否强制
    # g = sns.catplot(x="Attence", y="Star Rating Given by Students",kind='box' ,width=0.25, data=df[["Attence", "Star Rating Given by Students"]])

    # 成绩分布
    g = sns.catplot(x="Student Grade", y="Star Rating Given by Students",kind='box' ,width=0.25, data=df[["Student Grade", "Star Rating Given by Students"]], order=['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','D-','F','WD','INC','Not','Audit/No'])
    plt.show()


# 是否为了分数, 强制进行非参数检验
def mannwhitneyu():
    print('manteniU检验，是不是为了学分～～～～～～～～～～')
    from math import sqrt
    df = pd.read_csv(src, usecols=['student_star', 'for_credits','attence'])

    # 检验是否为了学分
    # df2 = df[(df['for_credits'] == 'Yes') & (df['student_star'] >= 1.0 )] # yes分组
    # df3 = df[(df['for_credits'] == 'No') & (df['student_star'] >= 1.0 )]  # no分组

    # 检验是否强制
    df2 = df[(df['attence'] == 'Mandatory') & (df['student_star'] >= 1.0)]  # yes分组
    df3 = df[(df['attence'] == 'Not Mandatory') & (df['student_star'] >= 1.0)]  # no分组


    p = stats.mannwhitneyu(df2['student_star'],df3['student_star'],alternative='two-sided')
    print(p)

    yes_num = df2['student_star'].count()  # 个数
    yes_mean = df2['student_star'].mean()  # 平均数
    yes_std = df2['student_star'].std()  # 标准差
    print('Yes个数',yes_num,'平均数：', yes_mean,'标准差：',yes_std)

    no_num = df3['student_star'].count()  # 个数
    no_mean = df3['student_star'].mean()  # 平均数
    no_std = df3['student_star'].std()  # 标准差
    print('NO个数',no_num,'平均数：', no_mean,'标准差：',no_std)
    #test conditions
    cohens_d = (yes_mean - no_mean) / (sqrt((yes_std ** 2 + no_std ** 2) / 2))
    print('cohens_d:',cohens_d)

# 学生分数检验
def Anova():
    df = pd.read_csv(src, usecols=['student_star', 'grades'])
    df = df.dropna(axis=0)
    student_grade =['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F', 'WD', 'INC', 'Not','Audit/No']

    df_A = df[(df['grades'] == 'A+') & (df['student_star'] >= 1.0)]
    df_A2 = df[(df['grades'] == 'A') & (df['student_star'] >= 1.0)]
    df_A3 = df[(df['grades'] == 'A-') & (df['student_star'] >= 1.0)]

    df_B = df[(df['grades'] == 'B+') & (df['student_star'] >= 1.0)]
    df_B2 = df[(df['grades'] == 'B') & (df['student_star'] >= 1.0)]
    df_B3 = df[(df['grades'] == 'B-') & (df['student_star'] >= 1.0)]

    df_C = df[(df['grades'] == 'C+') & (df['student_star'] >= 1.0)]
    df_C2 = df[(df['grades'] == 'C') & (df['student_star'] >= 1.0)]
    df_C3 = df[(df['grades'] == 'C-') & (df['student_star'] >= 1.0)]

    df_D = df[(df['grades'] == 'D+') & (df['student_star'] >= 1.0)]
    df_D2 = df[(df['grades'] == 'D') & (df['student_star'] >= 1.0)]
    df_D3 = df[(df['grades'] == 'D-') & (df['student_star'] >= 1.0)]

    df_F = df[(df['grades'] == 'F') & (df['student_star'] >= 1.0)]
    df_WD = df[(df['grades'] == 'WD') & (df['student_star'] >= 1.0)]
    df_INC = df[(df['grades'] == 'INC') & (df['student_star'] >= 1.0)]
    df_Not = df[(df['grades'] == 'Not') & (df['student_star'] >= 1.0)]
    df_Audit = df[(df['grades'] == 'Audit/No') & (df['student_star'] >= 1.0)]

    # 方差齐性检验，不齐性
    # # https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.kruskal.html
    # leven_words_test = stats.levene(df_A['student_star'] ,df_A2['student_star'], df_A3['student_star'],
    #                                 df_B['student_star'], df_B2['student_star'], df_B3['student_star'],
    #                                 df_C['student_star'], df_C2['student_star'], df_C3['student_star'],
    #                                 df_D['student_star'], df_D2['student_star'], df_D3['student_star'],
    #                                 df_F['student_star'], df_WD['student_star'], df_INC['student_star'],
    #                                 df_Not['student_star'], df_Audit['student_star']
    #                                 ,center='median')
    # print('方差齐性检验%s, p值是：%s' % leven_words_test)

    h_test = stats.kruskal(df_A['student_star'] ,df_A2['student_star'], df_A3['student_star'],
                                    df_B['student_star'], df_B2['student_star'], df_B3['student_star'],
                                    df_C['student_star'], df_C2['student_star'], df_C3['student_star'],
                                    df_D['student_star'], df_D2['student_star'], df_D3['student_star'],
                                    df_F['student_star'], df_WD['student_star'], df_INC['student_star'],
                                    df_Not['student_star'], df_Audit['student_star'])
    print('Kruskal-Wallis H',h_test)


def word_comment():
    src1 = r'/Users/liumingchun/【1】科研+实验室/3-科研项目/RateMyProfessor/bigdata/ratemyprofessor1.csv'
    print('评论单词个数比较～～～～～～～～～～')
    df = pd.read_csv(src1,usecols=['star_rating', 'word_comment','comments'])
    df = df.drop_duplicates(['comments'], 'first', False)

    # 按照1-2星及4-5星分为2组，然后进行比较两者差异，独立样本T检验
    high_words_comments = df[(df['star_rating'] <= 5.0) & (df['star_rating'] >= 4.0)] # 低分组
    low_words_comments = df[(df['star_rating'] >= 1.0) & (df['star_rating'] <= 2.0)]  # 高分组

    high_words_comment = high_words_comments['word_comment']
    low_words_comment =  low_words_comments['word_comment']

    # 方差齐性检验
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


if __name__ == '__main__':
    # TagProfessorAnalysis() # 计算助理教授、副教授和终身教授的tag
    # High_low_professor_tag_analysis() #计算高分组和低分组教授的tag
    # corr_rating_difficulty() # 计算难度和评分的关系，计算所有变量的相关矩阵
    # for_cerdis()

    # mannwhitneyu()
    # Anova()

    # word_comment()

    # corr_rating_take_again()

    all_corr()