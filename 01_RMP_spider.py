# -*- coding: utf-8 -*-
# @Time    : 2018/12/23 下午15:35
# @Author  : mingchun liu
# @Email   : psymingchun@gmail.com
# @File    : RMP.py
# @Software: PyCharm

import urllib.request
import time

def rmp_spider(url):
    """
    1、根据ID遍历http://www.ratemyprofessors.com/网页
    2、以html或文本形式保存在本地，例如 pro-data-1901092
    """
    num = 1901092
    for i in range(1901092,2901092):
        try:
            url1 = url + str(i)
            pageFile = urllib.request.urlopen(url1) # 通过URL获取网页信息
            pageHtml = pageFile.read() # 读取网页源码
            # print(pageHtml) # 打印网页源码
            pageFile.close()

            file = 'pro-data-'+str(num)
            with open(file,'a+') as f: # 打开本地文件
                f.write(str(pageHtml)) # 写入html
            f.close() #关闭文件
            print('文件名是：%s' % file)
            num+= 1

            time.sleep(3)

        except:
            print('没有找到网页')

if __name__ == "__main__":
    url = "http://www.ratemyprofessors.com/ShowRatings.jsp?tid="
    rmp_spider(url)