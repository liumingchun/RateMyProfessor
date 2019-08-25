# -*- coding: utf-8 -*-
# @Time    : 2019/7/4 上午11:45
# @Author  : mingchun liu
# @Email   : psymingchun@gmail.com
# @File    : 时间序列分析.py
# @Software: PyCharm
'''
设置数字标签 https://www.jianshu.com/p/5ae17ace7984
'''

import pandas as pd

from matplotlib import pyplot as plt
from scipy.stats import sem

src = r'/Users/liumingchun/【1】科研+实验室/3-科研项目/RateMyProfessor/bigdata/ratemyprofessor.csv'
df = pd.read_csv(src, usecols=['professor_name','school_name','department_name','year_since_first_review','star_rating'])
df = df.drop_duplicates(['professor_name','school_name'], 'first', False)

df_mean = df.groupby('year_since_first_review').star_rating.mean()
# print(df_mean)
df_se = df.groupby('year_since_first_review').star_rating.apply(sem).mul(1.96)
# print(df_se)
count = df.groupby('year_since_first_review').professor_name.count()

x = df_mean.index.astype(int)
sum = count.astype(int)
print(count.astype(int))
# # abc = df['professor_name'].duplicated(keep=False)
# abc = df['professor_name'].drop_duplicates()
# print(abc)


# Plot
fig, ax1 = plt.subplots()

fig.set_figwidth(15)
fig.set_figheight(12)


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
plt.axvline(6) # 参考线
plt.axvline(15) # 参考线
# Draw Horizontal Tick lines
for y in range(3, 5, 1):
    plt.hlines(y, xmin=s, xmax=e, colors='black', alpha=0.5, linestyles="--", lw=2)

ax1.text(1, 1.5, r"assistant professor", fontsize=10)
ax1.text(8, 1.5, r'associate professor', fontsize=10)
ax1.text(16,1.5, r'full professor', fontsize=10)
ax1.set_ylim([1, 5])


# # 直方图，显示教授人数
ax2 = ax1.twinx()
plt.ylabel('Number of Professor', fontsize=12)
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(1)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(1)
plt.xticks(x[::2], [str(d) for d in x[::2]] , fontsize=10)
# plt.title("Number of Professor by Teaching", fontsize=12)
plt.xlabel("Year Since First Review", fontsize=10)
for a,b in zip(x,sum):
    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=7)

ax2.set_ylim([1, 180000])
ax2.plot(list(x), list(sum), color="red")

fig.tight_layout()
plt.show()

