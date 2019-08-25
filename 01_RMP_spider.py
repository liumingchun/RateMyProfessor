# -*- coding: utf-8 -*-
'''
基本思路：
1、根据ID遍历http://www.ratemyprofessors.com/网页，并保存到本地
2、该网站反爬能力较弱，因此比较容易爬取
3、最后保存为 pro-data-1901092 的文件
'''

import urllib.request
import time

url1 = "http://www.ratemyprofessors.com/ShowRatings.jsp?tid="

num = 1901092
for i in range(1901092,2901092):
    try:
        url = url1 + str(i)
        pageFile = urllib.request.urlopen(url) #通过URL获取网页信息
        pageHtml = pageFile.read() #读取网页源码
        # print(pageHtml) #打印网页源码
        pageFile.close()

        file = 'pro-data-'+str(num)
        with open(file,'a+') as f: #打开本地文件
            f.write(str(pageHtml)) # 写入html
        f.close() #关闭文件
        print('文件名是：%s' % file)
        num+= 1

        time.sleep(3)

    except:
        print('没有找到网页')