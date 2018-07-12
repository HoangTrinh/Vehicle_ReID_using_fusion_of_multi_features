# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:43:30 2018

@author: Nguyen Thanh Son
@editor: Trinh Man Hoang

1. read file distance.txt --> dist_res, nQuery, nQuery
2. Load all query --> name_query
3. Load all test --> name_test
4. with idQuery in [1..nQuery] extract colum #idQuery of dist_res --> res(nTest, 2)
        res[0]: distance of (imgQueryName, imgIDTest)
        res[1]: imgIDTest: ID of imgTest (init with 0, 1, 2, ... nTest-1)
    sort res by distance
    create 100match_<imgQueryName>.htm
"""

import numpy as np
import os

def show_result(distance_file = 'dist_triple_VGG16_VeRi_vcolor_googlenet_1024.txt', result_folder = 'results'):
    szListDir = "../Data/VeRi"
    szHtmDir = result_folder
    temp_r =  "file:///%s/" + result_folder
    szHtmAsolute = temp_r % os.getcwd()
    szDataDir = "Distance_files"
    szImgTestDir = "../../Data/VeRi/image_test"
    szImgQueryDir = "../../Data/VeRi/image_query"

    #1. load distances --> dist_res
    temp_s = "%s/" + distance_file
    szTmp = temp_s % szDataDir
    szTestFile = "control_result.htm"
    dist_res = np.loadtxt(szTmp)
    nTest = len(dist_res)
    nQuery = len(dist_res[0])
    print("Size of matrix %d x %d" % (nTest, nQuery))

    #2. load all query name
    szTmp = "%s/name_query.txt" % szListDir
    name_query = open(szTmp,"r").readlines()
    if nQuery != len(name_query):
        print("No Queries NOT EQUAL no fusion distance matrix")
        quit()
    for idQuery in range(nQuery):
        name_query[idQuery] = name_query[idQuery].strip('\n')

    #3. load all test name
    szTmp = "%s/name_test.txt" % szListDir
    name_test = open(szTmp,"r").readlines()
    for idTest in range(nTest):
        name_test[idTest]=name_test[idTest].strip('\n')


    def checkQuery(idQuery=0):
        "Create htm/Fusion_<name_query[idQuery]>.htm; idQuery in [0..nQuery-1]"
        res = np.zeros((nTest, 2))
        res[:, 0] = dist_res[:, idQuery]
        res[:, 1] = np.arange(0, nTest, 1)
        res = res[res[:, 0].argsort()]
        szOutFile = "%s/Fusion_match_%s.htm" % (szHtmDir, name_query[idQuery])
        fileout = open(szOutFile, "w")
        fileout.write("<H1>ID: %s</H1>\n" % name_query[idQuery].split('_')[0])

        #view imgQueryName
        szFullPath = "%s/%s" % (szImgQueryDir, name_query[idQuery])
        szLine = "<IMG SRC=%s WIDTH='200' HEIGHT='200'>\n<BR>" % szFullPath
        fileout.write(szLine)
        #view top 50 best matched
        for i in range(50):
            idTest = int(res[i][1])
            szFullPath = "%s/%s" % (szImgTestDir, name_test[idTest])
            test_name = name_test[idTest].split('_')[0]
            #szLine = "<IMG SRC=%s WIDTH='150' HEIGHT='150'>  %lf\n" % (szFullPath, res[i][0])
            szLine = "<IMG SRC=%s WIDTH='150' HEIGHT='150'>  %s\n" % (szFullPath, test_name)
            fileout.write(szLine)
            if ((i+1)%5 == 0):
                fileout.write("<BR>")
        fileout.close()
        return name_query[idQuery]


    filetest = open(szTestFile, "w")
    filetest.write("<H1>View image and 50_nearest </H1>\n")
    for idQuery in range(nQuery):
        filetest.write("<a href=\"%s/Fusion_match_%s.htm\">%s</a><BR>\n" % (szHtmAsolute, checkQuery(idQuery), name_query[idQuery]))
    filetest.close()

if __name__ == '__main__':
    show_result()