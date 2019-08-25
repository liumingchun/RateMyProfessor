
# -*- coding: utf-8 -*-
# @Time    : 2019/1/23 下午12:29
# @Author  : mingchun liu
# @Email   : psymingchun@gmail.com
# @File    : RMP.py
# @Software: PyCharm
'''
从pro-data-xxx文件中提取网页信息，最后输出 RMP_sample_data.csv

'''
import pandas as pd
from pyquery import PyQuery as PQ
import re
import os

path ='/Users/liumingchun/【1】科研+实验室/4-科研项目/RateMyProfessor/data/DownloadedRateMyProfessorData-2/' #原始数据地址
res_path = '/Users/liumingchun/Document/data/' # 保存csv地址
file_list = os.listdir(path)

num = 0
for each in file_list:
    # print(path+each)
    file_path = path+each
    with open(file_path,'r') as f:
        html = f.read()

    professor_names = []
    school_names = []
    department_names=[]
    local_names = []
    state_names = []

    star_ratings = []
    take_agains = []
    tags_professor = []

    diff_indexs = []
    num_students = []
    post_date=[]
    name_onlines =[]

    stu_stars = []
    stu_diffs = []
    attences = []
    for_credits =[]
    would_take_agains =[]
    grades =[]
    comments = []

    try:
        doc = PQ(html)
    except:
        print(f'{file_path} Document is empty')
    else:
        professor_name = doc('h1.profname')
        if professor_name:
            # professor name
            professor_name = doc('h1.profname').text()
            professor = re.sub(r'\\r\\n', "", professor_name)
            professor_name = professor.strip()
            # print(professor_name)

            # area
            area = doc('h2.schoolname').text()
            school_name = area.split(',')[0][3:]
            depart_name = doc('div.result-title').text().split('\\r\\n')[1]
            department_name = re.search(r'.*department', depart_name).group()[18:]

            local_name = area.split(',')[1]
            state_name = area.split(',')[2]
        else:
            professor_name = 'NULL'
            school_name = 'NULL'
            department_name = 'NULL'
            local_name = 'NULL'
            state_name = 'NULL'
        professor_names.append(professor_name)
        school_names.append(school_name)
        department_names.append(department_name)
        local_names.append(local_name)
        state_names.append(state_name)

        rating_div = doc('div.rating-breakdown')
        if rating_div:
            # star_rating would take again level of difficulty
            grade_divs = doc('div.grade')
            for index, grade in enumerate(grade_divs.items()):
                if index == 0:
                    star_rating = grade.text()
                    star_rating = re.sub(r'\\r\\n', "", star_rating)
                    star_rating = star_rating.strip()
                elif index == 1:
                    take_again = grade.text()
                    take_again = re.sub(r'\\r\\n', "", take_again)
                    take_again = take_again.strip()
                elif index == 2:
                    diff_index = grade.text()
                    # print(f'first:{diff_index}')
                    diff_index = re.sub(r'\\r\\n', "", diff_index)
                    diff_index = diff_index.strip()
                    # print(f'second:{diff_index}')
            star_ratings.append(star_rating)
            take_agains.append(take_again)
            diff_indexs.append(diff_index)

            # tags
            tag = doc('div.tag-box span')
            if tag:
                tag = doc('div.tag-box').text()
                tag = re.sub(r'\\r\\n', "", tag)
                tag = tag.strip()
                tags_professor.append(tag)
            else:
                tags_professor.append('NULL')

            # number of students
            num_student = doc('div.table-toggle.rating-count.active').text()
            num_student = re.sub(r'\\r\\n', "", num_student)
            num_student = num_student.strip()
            num_student = num_student.split(' ')[0]
            num_students.append(num_student)

        date = doc('div.date')
        # print(type(date))
        for each in date.items():
            each_date = re.sub(r'\\r\\n', "", each.text()).strip()
            # print(each_date)
            post_date.append(each_date)

        name_online = doc('span.name')
        for each in name_online.items():
            name_onlines.append(each.text().strip())

        for_credit = doc('span.credit')
        for each in for_credit.items():
            credit = each.text().split(' ')[2].strip()
            for_credits.append(credit)

        #学生个人对教授的评分及课程难度评价
        stu_star = doc('div.descriptor-container')
        for each in stu_star.items():
            each_stu_star = re.sub(r'\\r\\n', "", each.text()).strip()
            if 'Overall' in each_stu_star:
                # print(each_stu_star[:3])
                stu_stars.append(each_stu_star[:3])
            elif 'Level of Difficulty' in each_stu_star:
                # print(each_stu_star[:3])
                stu_diffs.append(each_stu_star[:3])
            else:
                pass

        attence = doc('span.attendance')
        for each in attence.items():
            stu_attence = re.sub(r'\\r\\n', "", each.text()).strip()
            # print(stu_attence[12:])
            attences.append(stu_attence[12:])

        take_again = doc('span.would-take-again')
        for each in take_again.items():
            would_tale_again = each.text().split(' ')[3].strip()
            would_take_agains.append(would_tale_again)

        grade = doc('span.grade')
        for each in grade.items():
            recieved_grade = each.text().split(' ')[2].strip()
            grades.append(recieved_grade)

        comment = doc('p.commentsParagraph')
        for each in comment.items():
            each_comment = re.sub(r'\\r\\n',"",each.text()).strip()
            # print(each_comment)
            comments.append(each_comment)

        infos = {'professor_name': professor_names, 'school_name': school_names, 'department_name': department_names,
                 'local_name': local_names, 'state_name': state_names, 'star_rating': star_ratings, 'take_again': take_agains,
                 'diff_index': diff_indexs, 'tag_professor': tags_professor, 'num_student': num_students,
                 'post_date': post_date,'name_onlines':name_onlines,'student_star ':stu_stars, 'student_difficult' :stu_diffs, 'attence':attences, 'for_credits':for_credits,'would_take_agains':would_take_agains,'grades':grades,'comments': comments}
        df = pd.DataFrame().from_dict(infos,orient='index')
        df = df.T

        df.to_csv(res_path+str(num)+'.csv',index=False)
        print('第'+str(num) +'文件')
        num += 1