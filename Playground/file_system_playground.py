
import sys

# sys.path.append('/Users/dechamoungsri/Dropbox_dechamoungsri/Dropbox/python_workspace/Utility/')
sys.path.append('/home/h1/decha/Dropbox/python_workspace/Utility/')

from tool_box.util.utility import Utility

import numpy as np

if __name__ == '__main__':

    npy_file_path = '/work/w16/decha/decha_w16/spec_com_work_space/Speech_synthesis/05_GPR_abcdefghijklmnopqrst/testrun/out/tsc/a-t/speech_param/a-t/demo/seed-00/M-1024/B-1024/num_iters-5/dur_old/param_mean/tscsdj01.npy'

    mean = np.load(npy_file_path)
    # print mean

    param_path = '/work/w16/decha/decha_w16/spec_com_work_space/Speech_synthesis/05_GPR_abcdefghijklmnopqrst/testrun/out/tsc/a-t/speech_param/a-t/demo/seed-00/M-1024/B-1024/num_iters-5/dur_old/param_mean/tscsdj01.param'

    param = Utility.load_obj(param_path)
    print param

    pass
