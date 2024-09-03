import numpy as np
import os
import random
import pandas as pd
from sklearn import preprocessing
from dtaidistance import dtw


def attention_add(a, b, w_ab):
    len_a = len(a)
    len_b = len(b)

    if len_a > len_b:
        distance, paths = dtw.warping_paths(a, b)
        best_path = dtw.best_path(paths)
        # print(best_path)
        for i in best_path:
            a[i[0]] += w_ab * b[i[1]]

        # for i in range(0, len(a) - len(b)):
        #     temp_dis = dtw.distance(a[i:i+len(np.array(b))], b)
        #     if temp_dis < dis:
        #         dis = temp_dis
        #         startidx = i
        #         endidx = i + len(np.array(b))
        # output = a[:startidx].tolist() + (a[startidx:endidx] + w_ab * b).tolist() + a[endidx+1:]
    elif len_a < len_b:
        distance, paths = dtw.warping_paths(a, b)
        best_path = dtw.best_path(paths)
        # print(best_path)
        for i in best_path:
            a[i[0]] += w_ab * b[i[1]]

        # for i in range(0, len(b) - len(a)):
        #     temp_dis = dtw.distance(b[i:i+len(np.array(a))], a)
        #     if temp_dis < dis:
        #         dis = temp_dis
        #         startidx = i
        #         endidx = i + len(np.array(a))
        # output = (a + w_ab * b[startidx:endidx]).tolist()
    else:
        a = (a + w_ab * b)
        # print(a)

    # print(dis)
    # print(startidx, endidx)
    return a