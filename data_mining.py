from lib import bvh2glo_simple, SL_dict
import os
import sys
import collections
import similaritymeasures
import numpy as np
import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal, interpolate, spatial
from dtaidistance import dtw
from dtaidistance import dtw_ndim
from dtw import dtw as dtw_slower
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean
from sklearn.metrics.pairwise import manhattan_distances
from timeit import default_timer as timer

def load(paths):
    try:
        with open(paths, 'r') as pth:
            paths_list = pth.readlines()
    except:
        print('Path list directory not found.')

    bvh_dir = paths_list[0].rstrip("\n") # all bvh files takes and dictionaries
    bvh_dict = paths_list[1].rstrip("\n") # bvh files with separate words signed
    source_dir = paths_list[2].rstrip("\n") # data converted from angular to global positions
    path_jointlist = paths_list[3].rstrip("\n") # path to the joint_list.txt file 
    path_chosen_joints = paths_list[4].rstrip("\n") # path to the chosen joints indexes from joint_list.txt file
    path_dictionary = paths_list[5].rstrip("\n") # path to the ultimate_dictionary2.txt file
    path_metadata = paths_list[6].rstrip("\n") # path to the meta.pkl file
    path_trajectory = paths_list[7].rstrip("\n") # path to the traj.pkl file
    path_output = paths_list[8] # path to the output_files folder

    with open(path_jointlist, 'r') as f:
        joint_list = f.readlines()      # the list of markers (tracked body parts)
    for i in range(len(joint_list)):
        joint_list[i] = joint_list[i].rstrip("\n")

    with open(path_metadata, 'rb') as pf:
        meta = pk.load(pf)              # metadata: meaning, the initial file, anotation
    with open(path_trajectory, 'rb') as pf:
        traj = pk.load(pf)              # trajektory [item, frame, joint, channel]

    return bvh_dir,bvh_dict,source_dir,path_jointlist,path_chosen_joints,path_dictionary,path_metadata,path_trajectory,path_output,joint_list,meta,traj


def mine_data(in_directory, out_directory):
    """
    converts data from angular BVH to global positions (npy matrix)
    :param in_directory:
    :param out_directory:
    :return:
    """
    bvhfile_list = [l for l in os.listdir(in_directory) if '.bvh' in l]
    for tmp_file_name in bvhfile_list:
        tmp_file_base_name = os.path.splitext(tmp_file_name)[0]
        tmp_data_glo = bvh2glo_simple.calculate(os.path.join(bvh_dir, tmp_file_name))
        np.save(os.path.join(out_directory,tmp_file_base_name + '.npy'), tmp_data_glo[1])


def create_trajectory_matrix(dictionary_data):
    """
    creates sign numpy dictionary
    :param dictionary_data:
    :return:
    """
    trajectory_list = []
    metadata_list = []

    for i, item in enumerate(dictionary_data):
        # vyřazuje položky ve slovníku, které nejsou zpracovane
        if all(['annotation_Filip_bvh_frame' in item.keys(), item['sign_id'] != '', '!' not in item['sign_id']]):
            anot = item['annotation_Filip_bvh_frame']
            npy_name = (os.path.splitext(item['src_mocap'])[0] + '.npy')
            if 'predlozky' in npy_name:  # bastl ! oprava rozdilneho nazvu souboru "predlozky_a_spojky" -> "predlozky_spojky"
                npy_name = npy_name.replace('_a_', '_')

            # tmp_trajectory = np.load(os.path.join(source_dir, os.path.splitext(item['src_mocap'])[0] + '.npy'))
            tmp_trajectory = np.load(os.path.join(source_dir, npy_name))[anot[0]:anot[1], :, :]
            metadata_list.append([item['sign_id'], item['src_mocap'], item['annotation_Filip_bvh_frame']])
            trajectory_list.append(tmp_trajectory)
            print('{:.2f} %'.format(float(i) / len(dictionary_data) * 100))

    with open(path_metadata, 'wb') as pf:
        pk.dump(metadata_list, pf)
    with open(path_trajectory, 'wb') as pf:
        pk.dump(trajectory_list, pf)


def get_chosen_joints(path_chosen_joints):

    with open(path_chosen_joints, 'r') as pth: # loads chosen joints idxs from the chosen_joints.txt file
        selected_joints_idxs = pth.readlines()
    selected_joints_idxs = [int(item.rstrip("\n")) for item in selected_joints_idxs]
    
    return selected_joints_idxs


def get_jointlist(path_jointlist):
    """Returns joint list
    Args:
        path_jointlist (string) = path to the joint_list.txt file in pc

    Returns:
        [list]: List of joints
    """
    file_joints = open(path_jointlist, 'r')
    joints = file_joints.readlines()
    joints = [f.rstrip() for f in joints]
    return joints


def prepare_trajectories(word1, word2, path_chosen_joints):
    """Prepares 2 signals for comparison
    Args:
        word1 (list): First trajectory signal to compare in format [frame,joint,channel]
        word2 (list): Second trajectory signal to compare in format [frame,joint,channel]
        path_chosen_joints (string): A path to the chosen_joints.txt file, for example '/data/chosen_joints.txt'

    Returns:
        [list]: A list of prepared values for each chosen joint separately in format [channel,frame]
    """
    chosen_joints = get_chosen_joints(path_chosen_joints)
    data_prepared = {}
    for i in range(len(chosen_joints)):
        seq1 = np.transpose(np.array(word1[:,chosen_joints[i],:]) - np.array(word1[:,1,:]))
        seq2 = np.transpose(np.array(word2[:,chosen_joints[i],:]) - np.array(word2[:,1,:]))
        data_prepared[chosen_joints[i]] = [seq1,seq2]

    return data_prepared


def compute_dtw(data_prepared, alg_type):
    """Computes 2 types of DTW algorithm on given data from words_preparation fcn
    Args:
        data_prepared (dictionary) = words_preparation fcn output

    Returns:
        [double]: Mean of distances for separate joints counted between 2 instances of words
    """

    dtw_dist = list()

    for _, val in data_prepared.items():

        if alg_type == 'softdtw':
            D = SquaredEuclidean(val[0].T, val[1].T)
            sdtw = SoftDTW(D, gamma=1.0)
            dist = sdtw.compute()
            dtw_dist.append(dist)
        else:
            dtw_dist.append(dtw_ndim.distance_fast(np.transpose(val[0]),np.transpose(val[1])))

    return np.mean(dtw_dist)


def one_word_dtw(word, path_jointlist, number_of_mins, alg_type = 'dtw', graph = 1):
    """Computes DTW distance between 1 word and all other words
    Args:
        word (string) = a given word
        path_jointlist (string) = a path to the joint_list.txt file in pc
        number_of_mins (int) = a number of minimum values that are given as output
        alg_type (string) = dtw or softdtw algorithm
        graph (boolean) = yes or no to display a graph of distance occurences in sorted words data set

    Returns:
        [list]: List of [number_of_mins] minimum values found with information
    """

    sign_name_list = [m[0] for m in meta]
    try:
        idx = sign_name_list.index(word)
    except:
        print('Slovo nenalezeno.')
        sys.exit()

    occurences = sign_name_list.count(word)

    print('{} vyskytu slova "{}"'.format(occurences, word))
    word_index = input('Index instance slova na testovani (0,{}): '.format(occurences-1))
    word_traj = traj[idx+int(word_index)]

    distance = np.zeros((len(traj)))
    for i in range(len(traj)):
        prepared_trajectories = prepare_trajectories(traj[i], word_traj, path_chosen_joints)
        distance[i] = (compute_dtw(prepared_trajectories, alg_type))

    sorted_distances = (distance.argsort())
    sorted_words = [meta[item][0] for item in sorted_distances]
    bestentered = sorted_distances[:number_of_mins-1]

    best100_occurences = sorted_words[:99].count(word)
    best500_occurences = sorted_words[:499].count(word)

    if graph:
        hist_data = [int(item == word) for item in sorted_words]
        hist_data_plot = list()
        idx1 = 0
        idx2 = 36
        while idx2<=len(hist_data):
            hist_data_plot.append(hist_data[idx1:idx2].count(1))
            idx1 += 36
            idx2 += 36
        plt.barh(np.arange(len(hist_data_plot)).tolist(),hist_data_plot)
        plt.xlabel('Vyskytu znaku "{}" v oblasti'.format(word))
        plt.ylabel('Seřazený dataset vzdáleností rozdělen do 37 oblastí')
        plt.xticks(np.arange(0, max(hist_data_plot), 1))
        plt.title('Rozložení znaků s významem "{}" v seřazeném datasetu'.format(word))
        plt.grid()
        plt.show()

    print('\nSerazene vysledky {} slova "{}":'.format(alg_type, word))
    print('Vyskytu slova v {} nejmensich vysledcich: {}'.format(100,best100_occurences))
    print('Vyskytu slova v {} nejmensich vysledcich: {}'.format(500,best500_occurences))
    print('Nejlepších {} shod s {}.instanci slova: {}'.format(number_of_mins,word_index,word))
    for item in bestentered:
        print('{}: {}'.format(meta[item], distance[item]))
    
    return bestentered


def compute(path_output, path_trajectory, path_chosen_joints, alg_type = 'dtw', order = 'notImportant', resample_method = 'interpolation', int_method = 'linear', distance_method = 'euclidean', graph = 0, word_amount = None, occurence_lower_limit = None):
    """Computes distance between given number of words and all others
    Args:
        word_amount [int]: a number of words to count the distance, takes all if the number equals -1
        alg_type [string]: 'dtw', 'softdtw' or 'method_combination' options, continues with this type of algorithm
        resample_method [string]: 'interpolation' or 'fourier' resample
        int_method [string]: if 'interpolation' resample_method is chosen, options are 'linear', 'quadratic' and 'cubic' interpolation
        distance_method [string]: type of metrics to count distance by: 'euclidean', 'hamming', 'minkowsky', 'mahalanobis', 'pearson', 'correlationDistance', 'canberra', 'braycurtis', 'chebychev', 'fréchet'
        graph [boolean]: yes or no to display a cmap='hot' graph of all words distances

    Returns:
        [list]: A list of all distances between the given number of words and all others
    """
    with open(path_trajectory, 'rb') as pf:
        traj = pk.load(pf)              # trajektory [item, frame, joint, channel]
    """
    # GUI to choose used method and metric
    print('CHOOSE ALGORITHM TYPE:')
    while(True):
        alg_type = input('Enter 1: DTW, 2: SoftDTW, 3: Resample and comparison: ')

        if alg_type == '1':
            print('Computing DTW ... ...')
            alg_type = 'dtw'
            break

        elif alg_type == '2':
            print('Computing SoftDTW ... ...')
            alg_type = 'softdtw'
            break

        elif alg_type == '3':
            alg_type = 'method_combination'
            print('CHOOSE RESAMPLE METHOD:')
            while(True):
                resample_method = input('Enter 1: Interpolation, 2: Fourier transform: ')

                if resample_method == '1':
                    resample_method = 'interpolation'
                    print('CHOOSE INTERPOLATION METHOD:')
                    while(True):
                        int_method = input('Enter 1: Linear, 2: Quadratic, 3: Cubic: ')

                        if int_method == '1':
                            int_method = 'linear'
                            break
                        elif int_method == '2':
                            int_method = 'quadratic'
                            break
                        elif int_method == '3':
                            int_method = 'cubic'
                            break
                        else:
                            print('Please enter one of the offered options.')
                    break
                elif resample_method == '2':
                    resample_method = 'fourier'
                    break
                else:
                    print('Please enter one of the offered options.')
            print('CHOOSE METRICS:')
            while(True):
                distance_method = input('Enter 1: Euclidean\n 2: Hamming\n 3: Minkowsky\n 4: Mahalanobis\n 5: Canberra\n 6: Chebychev\n 7: BrayCurtis\n 8:Pearson correlation coefficient\n 9:Area between curves\n')
                if distance_method == '1':
                    distance_method = 'euclidean'
                    break
                elif distance_method == '2':
                    distance_method = 'hamming'
                    break
                elif distance_method == '3':
                    distance_method = 'minkowsky'
                    break
                elif distance_method == '4':
                    distance_method = 'mahalanobis'
                    break
                elif distance_method == '5':
                    distance_method = 'canberra'
                    break
                elif distance_method == '6':
                    distance_method = 'chebychev'
                    break
                elif distance_method == '7':
                    distance_method = 'braycurtis'
                    break
                elif distance_method == '8':
                    distance_method = 'pearson'
                    break
                elif distance_method == '9':
                    distance_method = 'area'
                    break
                else:
                    print('Please enter one of the offered options.')
            
            if resample_method == 'interpolation':
                clear = lambda: os.system('cls')
                clear()
                print('Computing {} {}, {} ... ...'.format(int_method, resample_method, distance_method))
            elif resample_method == 'fourier':
                clear = lambda: os.system('cls')
                clear()
                print('Computing {}, {} ... ...'.format(resample_method, distance_method))
            break
        else:
            print('Please enter one of the offered options.')
    """
    if word_amount == None: # computes with all words that appears to have more occurences than given limit
        distance = np.zeros((int(count_limit(occurence_lower_limit)[0]), len(traj)))
    elif word_amount == -1: # computes with all words
        distance = np.zeros((len(traj), len(traj)))
        DTW_differences = np.zeros((len(traj), len(traj)))
    elif word_amount != None: # computes with given number of words
        distance = np.zeros((int(word_amount), len(traj)))

    start = timer()
    for i in range(len(distance)):
        if (i+1)%100 == 0:
            print_time = timer()
            print('Currently computing {}. row of distance matrix, time: {}'.format(i+1,print_time))
        for j in range(len(distance[0])):
            if i == j:
                distance[i, j] = 0
            else:
                if alg_type == 'dtw': # Classic DTW algorithm
                    prepared_trajectories = prepare_trajectories(traj[i], traj[j], path_chosen_joints)
                    distance[i, j] = (compute_dtw(prepared_trajectories, 'dtw'))
                    try:
                        distance[j, i] = distance[i, j]
                    except:
                        pass

                elif alg_type == 'softdtw':# Differentiable SoftDTW version of DTW
                    prepared_trajectories = prepare_trajectories(traj[i], traj[j], path_chosen_joints)
                    distance[i, j] = (compute_dtw(prepared_trajectories, 'softdtw'))
                    try:
                        distance[j, i] = distance[i, j]
                    except:
                        pass

                elif alg_type == 'method_combination': # Signal resample and distance computation separately

                    if resample_method == 'interpolation':
                        prepared_trajectories = prepare_trajectories(traj[i], traj[j], path_chosen_joints)
                        resampled_trajectories, DTW_difference = resample(path_chosen_joints, prepared_trajectories, order, resample_method, int_method, graph=0)
                        dist_output = compare(resampled_trajectories, dist = distance_method)

                    elif resample_method == 'fourier':
                        prepared_trajectories = prepare_trajectories(traj[i], traj[j], path_chosen_joints)
                        resampled_trajectories, DTW_difference = resample(path_chosen_joints, prepared_trajectories, order, resample_method, graph=0)
                        dist_output = compare(resampled_trajectories, dist = distance_method)

                    else:
                        print('Wrong resample method entered.')

                    distance[i, j] = dist_output
                    DTW_differences[i, j] = DTW_difference
                    try:
                        distance[j, i] = distance[i, j]
                        DTW_differences[i, j] = DTW_difference
                    except:
                        pass
                else:
                    print('Wrong algorithm type entered.')

    # Save output data, graph and time info
    end = timer()
    time = end-start
    DTW_validator = np.mean(DTW_differences)

    if (alg_type == 'dtw') or (alg_type == 'softdtw'):
        with open(os.path.join(path_output, 'time_{}.txt'.format(alg_type)),"w") as file:
            file.write(str(time))
        with open(os.path.join(path_output, 'out_matrix_{}.pkl'.format(alg_type)), 'wb') as pk_out:
            pk.dump(distance, pk_out)

        if graph:
            plt.imshow(distance, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.savefig(os.path.join(path_output, 'Figure_{}.eps'.format(alg_type)), dpi=300)
            plt.savefig(os.path.join(path_output, 'Figure_{}.png'.format(alg_type)), dpi=300)
    else:
        if resample_method == 'fourier': # If the Fourier transform is computed in the name of output file will be fourier, else there will be the type of interpolation
            int_method = 'fourier'
        with open(os.path.join(path_output, 'time_{}_{}.txt'.format(int_method, distance_method)),"w") as file:
            file.write(str(time))
            file.write("\n{} efficiency: {}".format(int_method, DTW_validator))
        with open(os.path.join(path_output, 'out_matrix_{}_{}.pkl'.format(int_method, distance_method)), 'wb') as pk_out:
            pk.dump(distance, pk_out)

        if graph:
            plt.imshow(distance, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.savefig(os.path.join(path_output, 'Figure_{}_{}.eps'.format(int_method, distance_method)), dpi=300)
            plt.savefig(os.path.join(path_output, 'Figure_{}_{}.png'.format(int_method, distance_method)), dpi=300)
    return distance


def resample(path_chosen_joints, data_prepared, order = 'notImportant', method = 'interpolation', int_method = 'linear', graph = 0):
    """Resamples word2 signal to the length of the word1 signal
    Args:
        data_prepared [dictionary]: prepared data from prepare_trajectories fcn
        method [string]: the method of resampling used, 'interpolation', 'fourier'
        int_method [string]: the method of interpolation if the method 'interpolation' is selected
        graph [boolean]: yes or no to display graph with comparison of old and resampled signal

    Returns:
        [dictionary]: Dictionary with the same format as data_prepared with resampled lists
    """
    if method == 'fourier':
        
        data_out = {}
        chosen_joints = get_chosen_joints(path_chosen_joints)

        len_words = [len(data_prepared[chosen_joints[0]][0][0]), len(data_prepared[chosen_joints[0]][1][0])]
        word_to_resample = -1
        word_second = -1

        if order == 'toLonger':         # shorter to longer
            if len_words[0] <= len_words[1]:
                word_to_resample = 0
                word_second = 1

            if len_words[0] > len_words[1]:
                word_to_resample = 1
                word_second = 0

        elif order == 'toShorter':      # longer to shorter
            if len_words[0] >= len_words[1]:
                word_to_resample = 0
                word_second = 1

            if len_words[0] < len_words[1]:
                word_to_resample = 1
                word_second = 0
        else:
            word_to_resample = 0
            word_second = 1

        # Computation
        DTW_difference = []
        for key, val in data_prepared.items():
            item_new = np.zeros(shape=(2,3,len_words[word_second]))
            item_old = val[word_to_resample] # for resample validation by DTW alg.
            for i in range(3):
                item_new[word_to_resample][i] = signal.resample(val[word_to_resample][i], len_words[word_second])
                item_new[word_second][i] = val[word_second][i]
            
            # Resample validation
            DTW_difference.append(dtw_ndim.distance_fast(np.transpose(item_old),np.transpose(item_new[word_to_resample])))

            data_out[key] = item_new

        if graph:
            x_initial = np.linspace(0, len_words[word_second], len_words[word_to_resample])
            x_resampled = np.linspace(0, len_words[word_second], len_words[word_second])

            data_to_graph_initial = data_prepared[chosen_joints[0]] # graphing only first chosen joint for example
            data_to_graph_resampled = data_out[chosen_joints[0]]

            x_resampled_line = np.linspace(0, len_words[word_second], len_words[word_second]*10)
            data_to_graph_line = np.zeros(shape=(3,len_words[word_second]*10))
            for i in range(3):
                data_to_graph_line[i] = signal.resample(data_prepared[chosen_joints[0]][word_to_resample][i], len_words[word_second]*10)

            mpl.style.use('seaborn')
            fig, ax = plt.subplots(3, 1, sharex=True)
            ax[0].plot(x_initial, data_to_graph_initial[word_to_resample][0], marker='D', color = 'k', linewidth=0, markersize=4)
            ax[0].plot(x_resampled_line[0:1], data_to_graph_line[0][0:1], marker='*', color = 'r', linewidth=0.3, markersize=6)
            ax[0].plot(x_resampled_line, data_to_graph_line[0], color = 'r', linewidth=0.3, markersize=0)
            ax[0].plot(x_resampled, data_to_graph_resampled[word_to_resample][0], marker='*', color = 'r', linewidth=0, markersize=6)

            ax[1].plot(x_initial, data_to_graph_initial[word_to_resample][1],marker='D', color = 'k', linewidth=0, markersize=4)
            ax[1].plot(x_resampled_line[0:1], data_to_graph_line[1][0:1], marker='*', color = 'r', linewidth=0.3, markersize=6)
            ax[1].plot(x_resampled_line, data_to_graph_line[1], color = 'r', linewidth=0.3, markersize=0)
            ax[1].plot(x_resampled, data_to_graph_resampled[word_to_resample][1], marker='*', color = 'r', linewidth=0, markersize=6)

            ax[2].plot(x_initial, data_to_graph_initial[word_to_resample][2],marker='D', color = 'k', linewidth=0, markersize=4)
            ax[2].plot(x_resampled_line[0:1], data_to_graph_line[2][0:1], marker='*', color = 'r', linewidth=0.3, markersize=6)
            ax[2].plot(x_resampled_line, data_to_graph_line[2], color = 'r', linewidth=0.3, markersize=0)
            ax[2].plot(x_resampled, data_to_graph_resampled[word_to_resample][2], marker='*', color = 'r', linewidth=0, markersize=6)

            ax[2].set_title('Interpolovaná osa Z')
            ax[2].set_xlabel('čas [snímek]')
            ax[1].set_ylabel('vzdálenost od počátku [cm]')
            fig.legend(['Původní signál','Interpolovaný signál'], loc='upper right')
            plt.show()

        return data_out, np.mean(DTW_difference)

    if method == 'interpolation':

        data_out = {}
        chosen_joints = get_chosen_joints(path_chosen_joints)

        len_words = [len(data_prepared[chosen_joints[0]][0][0]), len(data_prepared[chosen_joints[0]][1][0])]
        word_to_resample = -1
        word_second = -1

        if order == 'toLonger':         # shorter to longer
            if len_words[0] <= len_words[1]:
                word_to_resample = 0
                word_second = 1

            if len_words[0] > len_words[1]:
                word_to_resample = 1
                word_second = 0

        elif order == 'toShorter':      # longer to shorter
            if len_words[0] >= len_words[1]:
                word_to_resample = 0
                word_second = 1

            if len_words[0] < len_words[1]:
                word_to_resample = 1
                word_second = 0
        else:
            word_to_resample = 0
            word_second = 1
            
        # Computation
        DTW_difference = []
        for key, val in data_prepared.items():
            item_new = np.zeros(shape=(2,3,len_words[word_second]))
            item_old = val[word_to_resample] # for interpolation validation by DTW alg.
            for i in range(3):
                item_new[word_to_resample][i] = interpolate_signal(val[word_to_resample][i], len_words[word_second], int_method)
                item_new[word_second][i] = val[word_second][i]

            # Interpolation validation
            DTW_difference.append(dtw_ndim.distance_fast(np.transpose(item_old),np.transpose(item_new[word_to_resample])))

            data_out[key] = item_new

        if graph:
            x_initial = np.linspace(0, len_words[word_second], len_words[word_to_resample])
            x_resampled = np.linspace(0, len_words[word_second], len_words[word_second])
            data_to_graph_initial = data_prepared[chosen_joints[0]] # graphing only first chosen joint for example
            data_to_graph_resampled = data_out[chosen_joints[0]]

            mpl.style.use('seaborn')
            fig, ax = plt.subplots(3,1, sharex=True)
            ax[0].plot(x_initial, data_to_graph_initial[word_to_resample][0], marker='D', color = 'k', linewidth=0, markersize=4)
            ax[0].plot(x_initial[0:1], data_to_graph_initial[word_to_resample][0][0:1], marker='*', color = 'r', linewidth=0.3, markersize=6)
            ax[0].plot(x_initial, data_to_graph_initial[word_to_resample][0], color = 'r', linewidth=0.3, markersize=0)
            ax[0].plot(x_resampled, data_to_graph_resampled[word_to_resample][0], marker='*', color = 'r', linewidth=0, markersize=6)

            ax[1].plot(x_initial, data_to_graph_initial[word_to_resample][1],marker='D', color = 'k', linewidth=0, markersize=4)
            ax[1].plot(x_initial[0:1], data_to_graph_initial[word_to_resample][1][0:1], marker='*', color = 'r', linewidth=0.3, markersize=6)
            ax[1].plot(x_initial, data_to_graph_initial[word_to_resample][1], color = 'r', linewidth=0.3, markersize=0)
            ax[1].plot(x_resampled, data_to_graph_resampled[word_to_resample][1], marker='*', color = 'r', linewidth=0, markersize=6)

            ax[2].plot(x_initial, data_to_graph_initial[word_to_resample][2],marker='D', color = 'k', linewidth=0, markersize=4)
            ax[2].plot(x_initial[0:1], data_to_graph_initial[word_to_resample][2][0:1], marker='*', color = 'r', linewidth=0.3, markersize=6)
            ax[2].plot(x_initial, data_to_graph_initial[word_to_resample][2], color = 'r', linewidth=0.3, markersize=0)
            ax[2].plot(x_resampled, data_to_graph_resampled[word_to_resample][2], marker='*', color = 'r', linewidth=0, markersize=6)

            ax[2].set_title('Interpolovaná osa Z')
            ax[2].set_xlabel('čas [snímek]')
            ax[1].set_ylabel('vzdálenost od počátku [cm]')
            fig.legend(['Původní signál','Interpolovaný signál'], loc='upper right')
            plt.show()

        return data_out, np.mean(DTW_difference)


def interpolate_signal(signal, final_length, int_method = 'linear'):
    """Interpolates given signal to given final_length
    Args:
        signal [list]: Signal that is being interpolated
        final_length [int]: A final length to interpolate signal to
        int_method [string]: A given method of interpolation, 'linear', 'quadratic', 'cubic' etc. 

    Returns:
        [list]: An interpolated signal with values in list
    """
    x = np.r_[0:len(signal)-1:complex(len(signal),1)]
    f = interpolate.interp1d(x,signal,kind=int_method)

    to_interpolate = np.r_[0:len(signal)-1:complex(final_length,1)]
    signal_interpolated = f(to_interpolate)
    return signal_interpolated


def compare(data_prepared, dist = 'euclidean'):
    """Counts the distance between 2 3D signals using one of implemented metrics
    Args:
        word1 [list]: The first signal for the computation
        word2 [list]: The second signal for the computation
        dist [string]: A metrics used for distance computation

    Returns:
        [double]: The distance between 2 given signals
    """
    distances = np.zeros(shape=len(data_prepared))
    joint_counter = -1

    for _, val in data_prepared.items():

        joint_counter += 1

        if (dist != 'pearson') and (dist != 'area'):

            for i in range(len(val[0])):
                if dist == 'euclidean':
                    distances[joint_counter] += spatial.distance.euclidean(val[0][:,i],val[1][:,i])
                elif dist == 'hamming':
                    distances[joint_counter] += spatial.distance.hamming(val[0][:,i],val[1][:,i])
                elif dist == 'minkowsky':
                    distances[joint_counter] += spatial.distance.minkowski(val[0][:,i],val[1][:,i], p=3)
                elif dist == 'mahalanobis':
                    V = np.cov(np.array([val[0][:,i],val[1][:,i]]).T)
                    IV = np.linalg.pinv(V)
                    distances[joint_counter] += spatial.distance.mahalanobis(val[0][:,i],val[1][:,i], IV)
                elif dist == 'canberra':
                    distances[joint_counter] += spatial.distance.canberra(val[0][:,i],val[1][:,i])
                elif dist == 'braycurtis':
                    distances[joint_counter] += spatial.distance.braycurtis(val[0][:,i],val[1][:,i])
                elif dist == 'chebyshev':
                    distances[joint_counter] += spatial.distance.chebyshev(val[0][:,i],val[1][:,i])
                else:
                    print('Wrong distance metrics entered.')

        elif dist == 'pearson':
            correlation = np.zeros(shape=len(val[0]))
            for i in range(len(val[0])):
                correlation[i] = np.corrcoef(val[0][i],val[1][i])[0][1]
            distances[joint_counter] = np.mean(correlation)
        elif dist =='area':
            distances[joint_counter] = similaritymeasures.area_between_two_curves(val[0],val[1])
        else:
            print('Wrong distance metrics entered.')

    return np.mean(distances)


def count_limit(limit):
    """Counts the sum of words instances that appears in more that given number
    Args:
        limit [int]: a minimum value for the word to be

    Returns:
        [int]: The sum of instances of chosen words
        [dictionary]: The number of instances for each chosen word separately
    """
    temp = []
    for i in range(len(meta)):
        temp.append(meta[i][0])

    meta_series = pd.Series(temp)
    counts = meta_series.value_counts().to_dict()

    sum_counts = sum([item for key, item in counts.items() if item >= limit])

    return [sum_counts,counts]


def analyze_result(tested_metrics, method_matrix, noOfminimuminstances, graph = 0):
    """Analysis of given output matrix from one algorithm type
    Args:
        method_matrix [list]: an output matrix from fcn compute
        noOfminimuminstances [int]: how many minimum instances of one word have to exist to include the word into analysis
        graph [boolean]: Yes or No to display a graph of analysis

    Returns:
        [words_data]: The result for all words separately
    """
    [noOfWords, words_counts_dict] = count_limit(noOfminimuminstances)

    matrix_chosen = method_matrix[0:noOfWords]

    top_counts = [1,3,5,10,20,30]

    words_data = np.empty(shape=(noOfWords,len(top_counts)), dtype=object)

    if (tested_metrics == 'pearson') and (tested_metrics == 'braycurtis'): # bigger value -> better result
        matrix_sorted = (-matrix_chosen).argsort()
    else: # smaller value -> better result
        matrix_sorted = matrix_chosen.argsort()
    #matrix_sorted = matrix_chosen.argsort()

    counts_dict = {}
    
    for i in range(noOfWords):
        tops = np.zeros(len(top_counts))

        tested = matrix_sorted[i].tolist()
        if tested[0] == i: # pop if first is the same instance to each other
            tested.pop(0)

        for j in range(len(tested)):
            word_main = meta[i][0]
            word_compared = meta[tested[j]][0]

            if word_main == word_compared:
                if j < top_counts[5]:
                    tops[5]+=1
                    if j < top_counts[4]:
                        tops[4]+=1
                        if j < top_counts[3]:
                            tops[3]+=1
                            if j < top_counts[2]:
                                tops[2]+=1
                                if j < top_counts[1]:
                                    tops[1]+=1
                                    if j < top_counts[0]:
                                        tops[0]+=1

        if not word_main in counts_dict.keys():
            counts_dict[word_main] = tops
        elif word_main in counts_dict.keys():
            counts_dict[word_main] =  [sum(x) for x in zip(counts_dict[word_main], tops)]

        words_data[i] = tops

    method_results = np.zeros(len(top_counts))
    for key, val in counts_dict.items():
        counts_dict[key] = (np.array(val)/words_counts_dict[key]).tolist()
        for i in range(len(method_results)):
            method_results[i] = method_results[i] + counts_dict[key][i]

    method_results = np.array(method_results)/len(counts_dict)

    if graph:
        mpl.style.use('seaborn')
        graph_data1 = []
        graph_data2 = []
        x = []
        for l in range(len(top_counts)):
            graph_data1.append(method_results[l]/top_counts[l]*100)
            graph_data2.append(100)
            x.append(str(top_counts[l]))
        plt.figure()
        plt.grid(True)
        plt.plot(0,0, markersize=0, linewidth=0)
        plt.bar(x,graph_data2, color='red', alpha=1, width=0.4, edgecolor='black', linewidth=1)
        plt.bar(x,graph_data1, color='green', alpha=1, width=0.4, edgecolor='black', linewidth=1)
        plt.xlabel('Počet nejbližších projevů [znak]')
        plt.ylabel('Zastoupení projevu se stejným významem [%]')
        for index,data in enumerate(graph_data1):
            plt.text(x=index-0.18, y=data+1, s="{:.2f} %".format(data) , fontdict=dict(fontsize=11), fontweight='bold')
        plt.legend(['','Rozdíl významu', 'Shoda významu'], bbox_to_anchor=(-0.1,1.02,0.6,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
        #plt.show()

        return words_data


if __name__ == '__main__':

    paths = 'Sign_Language_BP/data/paths.txt'

    bvh_dir,bvh_dict,source_dir,path_jointlist,path_chosen_joints,path_dictionary,path_metadata,path_trajectory,path_output,joint_list,meta,traj = load(paths)

    # converts data from angular BVH to global positions (npy matrix)
    mine = False
    if mine:
        mine_data(bvh_dir, source_dir)

    # creates sign numpy dictionary
    create = False
    if create:
        dict_items = SL_dict.read_dictionary(path_dictionary, 'dictionary_items')
        dict_takes = SL_dict.read_dictionary(path_dictionary, 'dictionary_takes')
        create_trajectory_matrix(dict_items + dict_takes)

    flexing = False
    if flexing:  # access to data examples
        print('Ukázka, jak vypadá položka v proměnné "meta": {}'.format(meta[0]))
        print('Proměnná "traj" je list o délce {}, což je počet všech dat nehledě na význam'.format(len(traj)))
        print('první 3 položky v traj mají následující dimenze:\n{}\n{}\n{}'.format(np.shape(traj[0]), np.shape(traj[1]), np.shape(traj[2])))
        print('Takže dimenze jsou [frame, joint, kanál]')

        # Výběr kloubu pro vizualizaci
        joint = 'RightHand\n'
        # tohle vrací celý list všech nalezených jointů, obsahujících řetězec joint, takže to vrací jednorozměrné pole, proto je tam ta [0], jakože vyberu první prvek z toho pole (abych neměl list, ale int)
        joint_id = [i for i, n in enumerate(joint_list) if joint in n][0]
        print(joint, joint_id)
        # výběr znaku podle id (takže to vrátí všechny se stejným id)
        query = 'teplo'
        list_of_same_signs = [e for e, s in enumerate(meta) if s[0] == query]
        plt.plot(traj[list_of_same_signs[0]][:, joint_id, :])
        plt.title('Takhle zhruba vypadá trajektorie znaku')
        plt.figure()
        plt.plot(traj[list_of_same_signs[0]][:, joint_id, 1])
        plt.title('Takhle to vypadá v zoomu pro jeden kanál')
        plt.show()

    sorting = False
    if sorting: # example of data sort
        sign_name_list = [m[0] for m in meta]
        unique_sign_list_unordered = set(sign_name_list)
        unique_count = []
        for item in unique_sign_list_unordered:
            matches = [m for m in sign_name_list if m == item]
            unique_count.append([item, len(matches)])

        unique_count.sort(key=lambda x: x[1], reverse=True)
        print((unique_count))

    # DTW of one word to all others
    test_dtw_one_word = False
    if test_dtw_one_word: 
        word = 'n_2'
        one_word_dtw(word, path_jointlist, 1, 'softdtw', graph=1)
    
    # Testing fcn of resample only
    test_resample = False
    if test_resample:
        joint = 3
        word1 = traj[191]
        word1_meta = meta[191]

        word2 = traj[250]
        word2_meta = meta[250]
        prepared_trajectories = prepare_trajectories(word1,word2,path_chosen_joints)
        resample_out = resample(path_chosen_joints, prepared_trajectories, 'toLonger', 'interpolation', 'cubic', graph=1)

    # Testing fcn of signal comparison
    test_signal_comparison = False
    if test_signal_comparison:
        joint = 3
        word1 = traj[900]
        word1_meta = meta[900]

        word2 = traj[200]
        word2_meta = meta[200]

        prepared_trajectories = prepare_trajectories(word1, word2, path_chosen_joints)
        resample_out = resample(path_chosen_joints, prepared_trajectories, 'toShorter', 'interpolation', 'linear', graph=0)
        kind = 'chebyshev'
        distance = compare(resample_out, dist = kind)

        print('{} counted over \'{}\' and \'{}\': {}'.format(kind, word1_meta[0], word2_meta[0], distance))

    # Analyze different results for different algorithm methods
    test_interps = False
    if test_interps: # testovaci skript
        with open("Sign_Language_BP/output_files/final/DTW/DTW.pkl", 'rb') as pickle_file:
            linear = pk.load(pickle_file)
        with open("Sign_Language_BP/output_files/final/SoftDtw/SoftDTW.pkl", 'rb') as pickle_file:
            quadr = pk.load(pickle_file)
        with open("Sign_Language_BP/output_files/final/Interp-Cubic,Euclidean/cubic+interpolation+euclidean.pkl", 'rb') as pickle_file:
            cubic = pk.load(pickle_file)

        sorted_lin = linear.argsort() # serazene 2D pole vzdalenosti s indexy
        sorted_quadr = quadr.argsort()
        sorted_cubic = cubic.argsort()

        linVSquadr = sorted_lin==sorted_quadr # porovnani 2 metod vuci sobe, True/False matice
        linVScubic = sorted_lin==sorted_cubic
        quadrVScubic = sorted_quadr==sorted_cubic

        unique, linVSquadrCounts = np.unique(linVSquadr, return_counts=True) # pocet shod a rozdilu dvou danych metod 
        dict(zip(unique, linVSquadrCounts)) # 27 612 False, 972 388 True

        unique, linVScubicCounts = np.unique(linVScubic, return_counts=True)
        dict(zip(unique, linVScubicCounts)) # 27 628 False, 972 372 True

        unique, quadrVScubicCounts = np.unique(quadrVScubic, return_counts=True)
        dict(zip(unique, quadrVScubicCounts)) # 736 False, 999 264 True

        for i in range(len(linVSquadr)): # hledam mista, kde se vyskytuje vice rozdilu za sebou a zjistuji, ze jsou to vetsinou pouze prehozene dve slova vuci sobe
            for j in range(len(linVSquadr[i])):
                if linVSquadr[i,j] == False and linVSquadr[i,j-1] == False and linVSquadr[i,j+1] == False:# and linVSquadr[i,j+2] == False and linVSquadr[i,j+3] == False and linVSquadr[i,j+4] == False:
                    print(sorted_lin[i,j:j+5])     # print oblasti indexu s prohozenymi znaky
                    print(sorted_quadr[i,j:j+5])
                    words = [meta[item][0] for item in sorted_lin[i,j:j+5]] 
                    print(words)                    # print vyznamu prohozenych znaku
                    words = [meta[item][0] for item in sorted_quadr[i,j:j+5]]
                    print(words)
                    print([linear[i][item] for item in sorted_lin[i,j:j+5]])   # print vzdalenostnich hodnot pro dana slova
                    print([quadr[i][item] for item in sorted_quadr[i,j:j+5]])

    # Analysis of one method output matrix from compute fcn
    method_analyze = True
    if method_analyze:
        
        tested_metrics1 = 'chebyshev'
        tested_metrics2 = 'correlationDistance'

        with open("Sign_Language_BP/output_files/final/Lin,Chebyshev/out_matrix.pkl", 'rb') as pickle_file:
            output_1 = pk.load(pickle_file)
        with open("Sign_Language_BP/output_files/final/Lin,Euclidean/out_matrix.pkl", 'rb') as pickle_file:
            output_2 = pk.load(pickle_file)

        minOf_instances = 20
        analyze_result(tested_metrics1, output_1, minOf_instances, graph=1)
        analyze_result(tested_metrics2, output_2, minOf_instances, graph=1)
        plt.show()

    # Compute one algorithm option on optional data size
    compute_main = True
    if compute_main:
        alg_type = 'method_combination'
        resample_method = 'interpolation'
        int_method = 'linear'
        distance_method = 'braycurtis'
        order = 'toShorter'
        start = timer()
        distance_matrix = compute(path_output, path_trajectory, path_chosen_joints, alg_type=alg_type, order=order, resample_method=resample_method, int_method=int_method, distance_method=distance_method, graph = 1, word_amount=-1)
