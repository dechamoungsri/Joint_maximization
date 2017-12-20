
import sys

sys.path.append('/home/h1/decha/Dropbox/python_workspace/Utility/')

from tool_box.util.utility import Utility

import numpy as np
import matplotlib.pyplot as plt

import re

from numpy.linalg import inv

class PoGUtility(object):

    @staticmethod
    def read_file_to_W(label_file):

        pattern = re.compile(r"""(?P<start>.+)\s(?P<end>.+)\s.+/A:.+/D:.+\-(?P<phone_num>.+)\+.+/E:.+""",re.VERBOSE)

        phone_list = []

        for line in Utility.read_file_line_by_line(label_file):
            # print line
            match = re.match(pattern, line)
            if match:
                phone_num = match.group('phone_num')
                # print phone_num
                if phone_num == 'x' : 
                    phone_list.append(1)
                else:
                    phone_list.append(int(phone_num))

        row = len(phone_list)
        column = sum(phone_list)

        # print row, column

        w = []

        cur = 0

        for i in phone_list:
            r = np.zeros(column)
            r[cur:cur+i] = 1
            w.append(r)
            cur = cur+i

        w = np.array(w)
        # print w

        # for idx, i in enumerate(phone_list):
        #     print i, w[idx]

        return w
        pass

    @staticmethod
    def read_mean_and_cov_of_predictive_distribution(predictive_dist_path):
        mean_path = '{}/mean.npy'.format(predictive_dist_path)
        cov_path = '{}/cov.npy'.format(predictive_dist_path)

        mean = np.load(mean_path)
        cov = np.load(cov_path)

        # print mean
        # print cov

        return (mean, cov)

    @staticmethod
    def cal_mean_variance_of_product_of_gaussian(W, syl_mean, syl_cov, ph_mean, ph_cov, alpha=1, beta=1):

        P_inv = np.dot( 
                np.dot( np.transpose(W) , inv(syl_cov) ) 
                , W )
            
        r = np.dot( 
                np.dot( np.transpose(W) , inv(syl_cov) ) 
                , syl_mean )

        cov_inv = beta * P_inv + alpha * inv(ph_cov)
        cov = inv(cov_inv)

        mean = np.dot(cov, ( (beta * r) + (alpha * np.dot( inv(ph_cov), ph_mean )) ) )

        # print mean, cov

        return (mean, cov)

        pass

    """docstring for PoGUtility"""
    def __init__(self, arg):
        super(PoGUtility, self).__init__()
        
