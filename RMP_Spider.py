# -*- coding: utf-8 -*-
import urllib.request
import time

# url = "http://www.ratemyprofessors.com/ShowRatings.jsp?tid=1901092"
url1 = "http://www.ratemyprofessors.com/ShowRatings.jsp?tid="
url2 = 'https://www.ratemyprofessors.com/campusRatings.jsp?sid=630'

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

        time.sleep(2)

    except:
        print('没有找到网页')