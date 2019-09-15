# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:19:49 2019

@author: fuxiaojuan
"""
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

#get the relation between professor and comments
def vader(file=r'F:\python\RMP.csv'):
    csv_data = csv.reader(open(file,'r'))
    rows=[]
    dict={}
    writelist=[]
    for row in csv_data:   
        if row[0] not in rows:
            rows.append(row[0])
            dict[row[0]] = row[22]
        else:
            con=row[22]+dict[row[0]]
            dict[row[0]] = con     

#initialize the analyse tool
    sid = SentimentIntensityAnalyzer()
    
    #open the file for writing
    with open("test5.csv","w",newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["professor_name","neg","neu","pos",'compound','judge'])

        for k in dict.keys():
            print('The score of professor '+k+' is ')
            comments=dict[k]
            ss = sid.polarity_scores(comments)
            #show the result
            for m in ss:
                print('{0}:{1},'.format(m, ss[m]), end='\n')
            #write the result into the file    
            b=[k]
            b.append(ss['neg'])
            b.append(ss['neu'])
            b.append(ss['pos'])
            b.append(ss['compound'])
            # decide sentiment as positive, negative and neutral 
            if ss['compound'] >= 0.5 : 
                b.append('positive')
            elif ss['compound'] <= -0.5 :
                b.append('negative')  
            else : 
                b.append('nuetral')
            if b not in writelist:
                writelist.append(b)
        print(writelist)
        writelist.pop(0) #get rid of the title which had been written       
        writer.writerows(writelist)            
    
    
#call the function        
if __name__ == '__main__':	
    vader()