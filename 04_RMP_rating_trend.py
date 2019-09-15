# -*- coding: utf-8 -*-
# @Time    : 2019/7/4 上午11:45
# @Author  : mingchun liu
# @Email   : psymingchun@gmail.com
# @File    : 时间序列分析.py
# @Software: PyCharm
# @Resource: 设置数字标签 https://www.jianshu.com/p/5ae17ace7984
# @Resource: dpi和图片大小关系 https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.stats import sem

def professor_rating_year_trend(src):
    df = pd.read_csv(src, usecols=['professor_name', 'school_name', 'department_name', 'year_since_first_review', 'star_rating'])
    df = df.drop_duplicates(['professor_name','school_name'], 'first', False)

    # 计算平均数，标准差，置信区间
    df_mean = df.groupby('year_since_first_review').star_rating.mean()
    df_se = df.groupby('year_since_first_review').star_rating.apply(sem).mul(1.96)
    count = df.groupby('year_since_first_review').professor_name.count()

    x = df_mean.index.astype(int)
    sum = count.astype(int)
    print(count.astype(int))

    # Plot
    fig, ax1 = plt.subplots(dpi=300, figsize = (10, 8))
    rcParams['font.family'] = 'sans-serif'
    fig.set_figwidth(10)
    fig.set_figheight(8)
    ax1.set_xlim([0, 20])

    ax1.set_ylabel("Average Star Rating", fontsize=12)
    ax1.plot(list(x), list(df_mean), color="black", lw=1)
    ax1.fill_between(x, df_mean - df_se, df_mean + df_se, color="#b7dafc",lw=5) #显示标准区间

    # # # Decorations
    # # Lighten borders
    # plt.gca().spines["top"].set_alpha(0)
    # plt.gca().spines["bottom"].set_alpha(1)
    # plt.gca().spines["right"].set_alpha(0)
    # plt.gca().spines["left"].set_alpha(1)
    # plt.xticks(x[::1], [str(d) for d in x[::1]] , fontsize=10)
    # # plt.title("Star Rating by Teaching", fontsize=18)
    plt.xlabel("Year Since First Review", fontsize=12)

    s, e = plt.gca().get_xlim()
    plt.xlim(s, e)
    plt.axvline(6)  # 助理教授，参考线
    plt.axvline(15) # 副教授， 参考线
    # Draw Horizontal Tick lines
    for y in range(3, 5, 1):
        plt.hlines(y, xmin=s, xmax=e, colors='black', alpha=0.3, linestyles="--", lw=2)

    ax1.text(1, 1.5, r"assistant professor", fontsize=10)
    ax1.text(8, 1.5, r'associate professor', fontsize=10)
    ax1.text(16,1.5, r'full professor', fontsize=10)
    ax1.set_ylim([1, 5])

    # # 直方图，显示教授人数
    ax2 = ax1.twinx()
    plt.ylabel('The number of Professor', fontsize=12)
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(1)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(1)
    plt.xticks(x[::2], [str(d) for d in x[::2]] , fontsize=10)
    # plt.title("Number of Professor by Teaching", fontsize=12)
    plt.xlabel("Year Since First Review", fontsize=12)
    for a,b in zip(x,sum):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=10)

    ax2.set_ylim([1, 180000])
    ax2.plot(list(x), list(sum), color="red") # 教授人数的变化趋势，代替柱状图

    fig.tight_layout()
    # plt.show()
    plt.savefig('professor_year_review_trend.png')

if __name__ == "__main__":
    src = r'/Users/liumingchun/【1】科研+实验室/3-科研项目/RateMyProfessor/bigdata/ratemyprofessor.csv'
    professor_rating_year_trend(src)