# -*- coding: utf-8 -*-
# @Modified by: HUI CHEUNGYUEN
# @ProjectName:yolov5-pyqt5

'''
'''
import csv
import base64

def sava_id_info(user, pwd):
    headers = ['name', 'key']
    #user = base64.urlsafe_b64encode(user)
    #pwd = base64.urlsafe_b64encode(pwd)
    values = [{'name':user, 'key':pwd}]
    with open('userInfo.csv', 'a', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, headers)
        writer.writerows(values)

def get_id_info():
    USER_PWD = {}
    with open('userInfo.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            USER_PWD[row[0]] = row[1]
    return USER_PWD

#


