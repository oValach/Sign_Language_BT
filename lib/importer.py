import numpy as np
import pickle as pkl
import os


def import_abs_data(filepath):
    filename = filepath[0:12]
    path_converted = 'C:/Users/User/BP/Projekt/data_converted'

    dict_file = os.path.join(path_converted, 'dictionary_'+filename+'.pickle')
    pkl_dict = open(dict_file, "rb")
    dictionary = pkl.load(pkl_dict)

    traj_file = os.path.join(path_converted, 'trajectory_'+filename+'.pickle')
    pkl_traj = open(traj_file, "rb")
    trajectory = pkl.load(pkl_traj)

    return dictionary, trajectory

if __name__ == "__main__":
    file_joints = open('C:/Users/User/BP/Projekt/data/joint_list.txt', 'r')
    joints = file_joints.readlines()
    joints = [f.rstrip() for f in joints]

    dictionary, trajectory = import_abs_data('16_05_20_a_R')
    print(trajectory[100][2])
    print(trajectory[100][31])