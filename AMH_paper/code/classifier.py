# -*- coding:utf8 -*-
# @Author  :Dodo

import os
import sys
import collections
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial.distance import squareform
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

plt.style.use('bmh')

try:
    from IPython.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False

class KnnDtw(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays

    Arguments
    ---------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for KNN

    max_warping_window : int, optional (default = infinity)
        Maximum warping window allowed by the DTW dynamic
        programming function

    subsample_step : int, optional (default = 1)
        Step size for the timeseries array. By setting subsample_step = 2,
        the timeseries length will be reduced by 50% because every second
        item is skipped. Implemented by x[:, ::subsample_step]
    """

    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step

    def fit(self, x, l):
        """Fit the model using x as training data and l as class labels

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer

        l : array of shape [n_samples]
            Training labels for input into KNN classifier
        """

        self.x = x
        self.l = l

    def _dtw_distance(self, ts_a, ts_b, d = lambda x,y: abs(x-y)):
        """Returns the DTW similarity distance between two 2-D
        timeseries numpy arrays.

        Arguments
        ---------
        ts_a, ts_b : array of shape [n_samples, n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared

        d : DistanceMetric object (default = abs(x-y))
            the distance measure used for A_i - B_j in the
            DTW dynamic programming function

        Returns
        -------
        DTW distance between A and B
        """

        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = sys.maxsize * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(max(1, i - self.max_warping_window),
                           min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        # Return DTW distance given window
        return cost[-1, -1]

    def _dist_matrix(self, x, y):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]

        y : array of shape [n_samples, n_timepoints]

        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """

        # Compute the distance matrix
        dm_count = 0

        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if(np.array_equal(x, y)):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)

            p = ProgressBar(dm.shape[0])

            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = self._dtw_distance(x[i, ::self.subsample_step],
                                                      y[j, ::self.subsample_step])

                    dm_count += 1
                    p.animate(dm_count)

            # Convert to squareform
            dm = squareform(dm)
            return dm

        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0]))
            dm_size = x_s[0]*y_s[0]

            p = ProgressBar(dm_size)

            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    dm[i, j] = self._dtw_distance(x[i, ::self.subsample_step],
                                                  y[j, ::self.subsample_step])
                    # Update progress bar
                    dm_count += 1
                    p.animate(dm_count)

            return dm

    def predict(self, x, y):
        """Predict the class labels or probability estimates for
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified

        Returns
        -------
          2 arrays representing:
              (1) the predicted class labels
              (2) the knn label count probability
        """

        dm = self._dist_matrix(x, self.x)

        acc = []

        # Identify the k nearest neighbors
        # argsort() 从小到大排列，返回index大小的顺序
        for i in range(1, 401):
            knn_idx = dm.argsort()[:, :i]

            # Identify k nearest labels
            knn_labels = self.l[knn_idx]

            # Model Label
            mode_data = mode(knn_labels, axis=1)
            mode_label = mode_data[0]
            mode_proba = mode_data[1]/i
            acc.append(cal_acc(y, mode_label))

        best = acc.index(max(acc))
        print('best k value:', best)
        print('\n Acc:', acc[best])

        knn_idx = dm.argsort()[:, :best]

        # Identify k nearest labels
        knn_labels = self.l[knn_idx]

        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1]/best

        return mode_label.ravel(), mode_proba.ravel(), best, max(acc)

    def test_predict(self, x):

        dm = self._dist_matrix(x, self.x)

        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.l[knn_idx]

        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1]/self.n_neighbors

        return mode_label.ravel(), mode_proba.ravel()

class ProgressBar:
    """This progress bar was taken from PYMC
    """
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        print ('\r', self,)
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
                        (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)

def load_data(csv_path):

    br = []
    deltaBp = []
    hr = []
    hrSnr = []
    hrv = []
    relax = []
    stress = []
    stressSnr = []

    df_br = pd.read_csv(csv_path, encoding="gbk",
                        usecols=["br", "deltaBp", "hr", "hrSnr", "hrv", "relax", "stress", "stressSnr"])
    # print(df_br)

    for index, row in df_br.iterrows():
        br.append(row['br'])
        deltaBp.append(row['deltaBp'])
        hr.append(row['hr'])
        hrSnr.append(row['hrSnr'])
        hrv.append(row['hrv'])
        relax.append(row['relax'])
        stress.append(row['stress'])
        stressSnr.append(row['stressSnr'])

    avg_br = np.mean(br)
    avg_deltaBp = np.mean(deltaBp)
    avg_hr = np.mean(hr)
    avg_hrSnr = np.mean(hrSnr)
    avg_hrv = np.mean(hrv)
    avg_relax = np.mean(relax)
    avg_stress = np.mean(stress)
    avg_stressSnr = np.mean(stressSnr)

    # print(index)
    for i in range(699-index):
        br.append(avg_br)
        deltaBp.append(avg_deltaBp)
        hr.append(avg_hr)
        hrSnr.append(avg_hrSnr)
        hrv.append(avg_hrv)
        relax.append(avg_relax)
        stress.append(avg_stress)
        stressSnr.append(avg_stressSnr)

    # X.reshape(index,1)
    # print(X)
    # print(type(X))
    # X = np.array(X)
    # print(len(X))
    # print(np.array(br).shape)
    return br, deltaBp, hr, hrSnr, hrv, relax, stress, stressSnr

def cal_acc(y, y_hat):
    return sum(yi == yi_hat for yi, yi_hat in zip(y, y_hat)) / len(y)

if __name__ == '__main__':

    time = np.linspace(0,20,1000)

    amplitude_a = 5*np.sin(time)
    amplitude_b = 3*np.sin(time + 1)

    # m = KnnDtw()
    # distance = m._dtw_distance(amplitude_a, amplitude_b)
    # fig = plt.figure(figsize=(12,4))
    # _ = plt.plot(time, amplitude_a, label='A')
    # _ = plt.plot(time, amplitude_b, label='B')
    # _ = plt.title('DTW distance between A and B is %.2f' % distance)
    # _ = plt.ylabel('Amplitude')
    # _ = plt.xlabel('Time')
    # _ = plt.legend()
    # plt.show()

    # dist_matrix = m._dist_matrix(np.random.random((4,50)), np.random.random((4,50)))
    # print(dist_matrix)

    # initialize

    X_br = []
    X_deltaBp = []
    X_hr = []
    X_hrSnr = []
    X_hrv = []
    X_relax = []
    X_stress = []
    X_stressSnr = []

    y = []

    dataset_path = './dataset'
    json_suffix = '.json'
    csv_suffix = '.csv'
    dataset_class = os.listdir(dataset_path)

    labels = {0:'unhealthy', 1:'healthy'}

    for sample_class in dataset_class:
        sample_class_path = dataset_path + '/' + sample_class
        sample_file = os.listdir(sample_class_path)

        for detail in sample_file:
            detail_path = sample_class_path + '/' + detail
            # print(detail_path)
            sample_detail = os.listdir(detail_path)
            json_path = detail_path + '/' + detail + '_emotion.json'
            # print(json_path)
            csv_path = json_path.replace(json_suffix, '') + csv_suffix
            # print(csv_path)

            if sample_class == 'healthy':
                # print('healthy')
                # load_data(csv_path)
                br, deltaBp, hr, hrSnr, hrv, relax, stress, stressSnr = load_data(csv_path)

                X_br.append(br)
                X_deltaBp.append(deltaBp)
                X_hr.append(hr)
                X_hrSnr.append(hrSnr)
                X_hrv.append(hrv)
                X_relax.append(relax)
                X_stress.append(stress)
                X_stressSnr.append(stressSnr)

                y.append(1)

            elif sample_class == 'unhealthy':
                br, deltaBp, hr, hrSnr, hrv, relax, stress, stressSnr = load_data(csv_path)
                # temp = np.array(temp)
                # X.append(temp)

                X_br.append(br)
                X_deltaBp.append(deltaBp)
                X_hr.append(hr)
                X_hrSnr.append(hrSnr)
                X_hrv.append(hrv)
                X_relax.append(relax)
                X_stress.append(stress)
                X_stressSnr.append(stressSnr)

                y.append(0)

            else :
                print('Another data file exist')
                continue

    X_br = np.array(X_br)
    X_deltaBp = np.array(X_deltaBp)
    X_hr = np.array(X_hr)
    X_hrSnr = np.array(X_hrSnr)
    X_hrv = np.array(X_hrv)
    X_relax = np.array(X_relax)
    X_stress = np.array(X_stress)
    X_stressSnr = np.array(X_stressSnr)

    y = np.array(y)

    print(X_br.shape)
    print(y.shape)

    X_br_train, X_br_test, y_br_train, y_br_test = train_test_split(X_br, y,
                                                                    test_size=0.5, random_state=42)
    X_br_train = np.array(X_br_train)
    y_br_train = np.array(y_br_train)
    X_br_test = np.array(X_br_test)
    y_br_test = np.array(y_br_test)

    X_deltaBp_train, X_deltaBp_test, y_deltaBp_train, y_deltaBp_test = train_test_split(X_deltaBp, y,
                                                                                        test_size=0.5, random_state=42)
    X_deltaBp_train = np.array(X_deltaBp_train)
    y_deltaBp_train = np.array(y_deltaBp_train)
    X_deltaBp_test = np.array(X_deltaBp_test)
    y_deltaBp_test = np.array(y_deltaBp_test)

    X_hr_train, X_hr_test, y_hr_train, y_hr_test = train_test_split(X_hr, y, test_size=0.5, random_state=42)
    X_hr_train = np.array(X_hr_train)
    y_hr_train = np.array(y_hr_train)
    X_hr_test = np.array(X_hr_test)
    y_hr_test = np.array(y_hr_test)

    X_hrSnr_train, X_hrSnr_test, y_hrSnr_train, y_hrSnr_test = train_test_split(X_hrSnr, y,
                                                                                test_size=0.5, random_state=42)
    X_hrSnr_train = np.array(X_hrSnr_train)
    y_hrSnr_train = np.array(y_hrSnr_train)
    X_hrSnr_test = np.array(X_hrSnr_test)
    y_hrSnr_test = np.array(y_hrSnr_test)

    X_hrv_train, X_hrv_test, y_hrv_train, y_hrv_test = train_test_split(X_hrv, y,
                                                                        test_size=0.5, random_state=42)
    X_hrv_train = np.array(X_hrv_train)
    y_hrv_train = np.array(y_hrv_train)
    X_hrv_test = np.array(X_hrv_test)
    y_hrv_test = np.array(y_hrv_test)

    X_relax_train, X_relax_test, y_relax_train, y_relax_test = train_test_split(X_relax, y,
                                                                                test_size=0.5, random_state=42)
    X_relax_train = np.array(X_relax_train)
    y_relax_train = np.array(y_relax_train)
    X_relax_test = np.array(X_relax_test)
    y_relax_test = np.array(y_relax_test)

    X_stress_train, X_stress_test, y_stress_train, y_stress_test = train_test_split(X_stress, y,
                                                                                    test_size=0.5, random_state=42)
    X_stress_train = np.array(X_stress_train)
    y_stress_train = np.array(y_stress_train)
    X_stress_test = np.array(X_stress_test)
    y_stress_test = np.array(y_stress_test)

    X_stressSnr_train, X_stressSnr_test, y_stressSnr_train, y_stressSnr_test = train_test_split(X_stressSnr, y,
                                                                                     test_size=0.5, random_state=42)
    X_stressSnr_train = np.array(X_stressSnr_train)
    y_stressSnr_train = np.array(y_stressSnr_train)
    X_stressSnr_test = np.array(X_stressSnr_test)
    y_stressSnr_test = np.array(y_stressSnr_test)

    # print(x_train.shape)
    # print(y_test.shape)
    # print(x_train)

    # 打印波形
    # plt.figure(figsize=(11,7))
    # colors = ['#D62728','#2C9F2C','#FD7F23','#1F77B4','#9467BD',
    #           '#8C564A','#7F7F7F','#1FBECF','#E377C2','#BCBD27']

    # for i, r in enumerate([0,65,100,145,172,203,240,260]):
    #
    #     plt.subplot(4,2,i+1)
    #     plt.plot(x_train[r][:700], label=labels[y_train[r]], color=colors[i], linewidth=2)
    #     plt.legend(loc='upper left') #图例
    #     plt.tight_layout()
    # plt.show()

    best_train_K = []
    Acc_train = []

    #br classifier
    classfier_br = KnnDtw(n_neighbors=1, max_warping_window=10)
    classfier_br.fit(X_br_train, y_br_train)
    label_br_train, proba, temp_k, temp_acc = classfier_br.predict(X_br_train, y_br_train)
    best_train_K.append(temp_k)
    Acc_train.append(temp_acc)

    #deltaBp classifier
    classfier_deltaBp = KnnDtw(n_neighbors=1, max_warping_window=10)
    classfier_deltaBp.fit(X_deltaBp_train, y_deltaBp_train)
    label_deltaBp_train, proba, temp_k, temp_acc = classfier_deltaBp.predict(X_deltaBp_train, y_deltaBp_train)
    best_train_K.append(temp_k)
    Acc_train.append(temp_acc)

    #hr classifier
    classfier_hr = KnnDtw(n_neighbors=1, max_warping_window=10)
    classfier_hr.fit(X_hr_train, y_hr_train)
    label_hr, proba, temp_k, temp_acc = classfier_hr.predict(X_hr_train, y_hr_train)
    best_train_K.append(temp_k)
    Acc_train.append(temp_acc)

    #hrSnr classifier
    classfier_hrSnr = KnnDtw(n_neighbors=1, max_warping_window=10)
    classfier_hrSnr.fit(X_hrSnr_train, y_hrSnr_train)
    label_hrSnr, proba, temp_k, temp_acc = classfier_hrSnr.predict(X_hrSnr_train, y_hrSnr_train)
    best_train_K.append(temp_k)
    Acc_train.append(temp_acc)

    #hrv classifier
    classfier_hrv = KnnDtw(n_neighbors=1, max_warping_window=10)
    classfier_hrv.fit(X_hrv_train, y_hrv_train)
    label_hrv, proba, temp_k, temp_acc = classfier_hrv.predict(X_hrv_train, y_hrv_train)
    best_train_K.append(temp_k)
    Acc_train.append(temp_acc)

    #relax classifier
    classfier_relax = KnnDtw(n_neighbors=1, max_warping_window=10)
    classfier_relax.fit(X_relax_train, y_relax_train)
    label_relax, proba, temp_k, temp_acc = classfier_relax.predict(X_relax_train, y_relax_train)
    best_train_K.append(temp_k)
    Acc_train.append(temp_acc)

    #stress classifier
    classfier_stress = KnnDtw(n_neighbors=1, max_warping_window=10)
    classfier_stress.fit(X_stress_train, y_stress_train)
    label_stress, proba, temp_k, temp_acc = classfier_stress.predict(X_stress_train, y_stress_train)
    best_train_K.append(temp_k)
    Acc_train.append(temp_acc)

    #stressSnr classifier
    classfier_stressSnr = KnnDtw(n_neighbors=1, max_warping_window=10)
    classfier_stressSnr.fit(X_stressSnr_train, y_stressSnr_train)
    label_stressSnr, proba, temp_k, temp_acc = classfier_stressSnr.predict(X_stressSnr_train, y_stressSnr_train)
    best_train_K.append(temp_k)
    Acc_train.append(temp_acc)

    print('Best K value of eight classifier:\n', best_train_K)
    print('Accuracy:\n', Acc_train)





