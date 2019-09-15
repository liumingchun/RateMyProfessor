# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 22:26:48 2019

@author: fuxiaojuan
"""
import csv
from textblob import TextBlob

#get the relation between professor and comments
def textblob(file=r'F:\python\RMP.csv'):    
    csv_data = csv.reader(open(file,'r'))
    rows=[]
    dict={}
    b=[]
    writelist=[]
    for row in csv_data:   
        if row[0] not in rows:
            rows.append(row[0])
            dict[row[0]] = row[22]
        else:
            con=row[22]+dict[row[0]]
            dict[row[0]] = con     

#write into the file
    with open("test7.csv","w",newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["professor_name","polarity","subjectivity"])
        print('The result with textblob us following\n')
        for k in dict.keys():
            blob = TextBlob(dict[k])
            score=blob.sentiment
            b=[k,score[0],score[1]]
            print(b)
            if b not in writelist:
                writelist.append(b)           
        writelist.pop(0)
        print(writelist)
        writer.writerows(writelist)
        




if __name__ == '__main__':
	textblob()


