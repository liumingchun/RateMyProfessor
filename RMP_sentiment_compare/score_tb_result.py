# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 22:26:48 2019

@author: fuxiaojuan
"""
import csv
from textblob import TextBlob

#prepare for the writing
def takeFirst(elem):
    return elem[0]

#find the score and the comments accordingly
def textblob(file=r'F:\python\RMP.csv'):    
    csv_data = csv.reader(open(file,'r'))
    rows=[]
    mydict={}
    writelist=[]
    for row in csv_data:   
        if row[6] not in rows:
            rows.append(row[6])
            mydict[row[6]] = row[22]
        else:
            con=row[22]+mydict[row[6]]
            mydict[row[6]] = con
    
    #write the result into csv
    with open("test3.csv","w",newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["score","polarity","subjectivity"])
        for k in dict.keys(mydict):
            blob = TextBlob(mydict[k])
            score=blob.sentiment
            b=[k,score[0],score[1]]
            if b not in writelist:
                writelist.append(b)           
        writelist.pop(0)
        writelist.sort(key=takeFirst)
        print(writelist)
        writer.writerows(writelist)
          

#call the method
if __name__ == '__main__':
	textblob()
    
