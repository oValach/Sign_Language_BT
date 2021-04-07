from lib import bvh2glo_simple, SL_dict
import os
import sys
import dcor
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

            # tmp_trajectory = np.load(os.path.join(glo_dir, os.path.splitext(item['src_mocap'])[0] + '.npy'))
            tmp_trajectory = np.load(os.path.join(glo_dir, npy_name))[anot[0]:anot[1], :, :]
            metadata_list.append([item['sign_id'], item['src_mocap'], item['annotation_Filip_bvh_frame']])
            trajectory_list.append(tmp_trajectory)
            print('{:.2f} %'.format(float(i) / len(dictionary_data) * 100))

    with open(path_metadata, 'wb') as pf:
        pk.dump(metadata_list, pf)
    with open(path_trajectory, 'wb') as pf:
        pk.dump(trajectory_list, pf)


def words_preparation(word1, word2, path_jointlist):
    """Prepares 2 signals for dtw to be counted on them
    Args:
        word1 (list): First signal to count in dtw fcn
        word2 (list): Second signal to count in dtw fcn
        path_jointlist (string): A path to the joint_list.txt file, for example 'Sign_Language_BP/data/joint_list.txt'

    Returns:
        [list]: A list of prepared values for each joint separately
    """

    jointlist = get_jointlist(path_jointlist)
    data_prepared = {}

    # přepočítání 3 dimenzionálních dat na 1 dimenzi porovnáváním po částech těla
    for i in range(len(jointlist)):  # joint
        seq1 = np.zeros(shape=(3, len(word1)), dtype=np.double)
        seq2 = np.zeros(shape=(3, len(word2)), dtype=np.double)

        for j in range(len(word1)):  # pocet snimku prvniho slova
            # skip nechtenych joints
            if any(char.isdigit() for char in jointlist[i]) or ('Head' in jointlist[i]) or ('Shoulder' in jointlist[i]) or ('Hips' in jointlist[i]):
                break
            if ('Spine' in jointlist[i]):
                seq1[0][j] = word1[j][i][0]
                # souradnice prvniho slova za podminky ramene
                seq1[1][j] = word1[j][i][1]
                seq1[2][j] = word1[j][i][2]
            else:
                seq1[0][j] = word1[j][i][0] - Spine[0][j]
                # souradnice prvniho slova s odectenim souradnic ramen
                seq1[1][j] = word1[j][i][1] - Spine[1][j]
                seq1[2][j] = word1[j][i][2] - Spine[2][j]

        for j in range(len(word2)):  # pocet snimku druheho slova
            # skip nechtenych joints
            if any(char.isdigit() for char in jointlist[i]) or ('Head' in jointlist[i]) or ('Shoulder' in jointlist[i]) or ('Hips' in jointlist[i]):
                break
            if ('Spine' in jointlist[i]):
                seq2[0][j] = word2[j][i][0]
                # souradnice druheho slova za podminky ramene
                seq2[1][j] = word2[j][i][1]
                seq2[2][j] = word2[j][i][2]
            else:
                seq2[0][j] = word2[j][i][0] - Spine[3][j]
                # souradnice druheho slova s odectenim souradnic ramen
                seq2[1][j] = word2[j][i][1] - Spine[4][j]
                seq2[2][j] = word2[j][i][2] - Spine[5][j]

        # skip nechtenych joints
        if any(char.isdigit() for char in jointlist[i]) or ('Head' in jointlist[i]) or ('Shoulder' in jointlist[i]) or ('Hips' in jointlist[i]):
            continue

        # "zbaveni se" ucinku pohybu ramen na ruce - ulozeni jejich souradnic pro nasledne odecteni
        if ('Spine' in jointlist[i]):
            Spine = np.array([seq1[0], seq1[1], seq1[2],seq2[0], seq2[1], seq2[2]], dtype=object)
            continue

        data_prepared[jointlist[i]] = [seq1, seq2]

    return data_prepared


def distance_computation_dtw(data_prepared, alg_type):
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
        words_prepared = words_preparation(traj[i], word_traj, path_jointlist)
        distance[i] = (distance_computation_dtw(words_prepared, alg_type))

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


def compute(word_amount, alg_type = 'dtw', resample_method = 'interpolation', int_method = 'linear', distance_method = 'euclidean', graph = 0):
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

    #Check of methods and metrics used
    if alg_type == 'dtw':
        print('Computing DTW ... ...')
    elif alg_type == 'softdtw':
        print('Computing SoftDTW ... ...')
    else:
        print('Resample method: {}'.format(resample_method))
        if resample_method == 'interpolation':
            print('Interpolation type: {}'.format(int_method))
        print('Distance computation by metrics: {}'.format(distance_method))
        print('Computing ... ...')

    if word_amount == -1:
        distance = np.zeros((len(traj), len(traj)))
    else:
        distance = np.zeros((int(word_amount), len(traj)))
    
    for i in range(len(distance)):
        for j in range(len(distance[0])):
            if i == j:
                distance[i, j] = 0
            else:
                if alg_type == 'dtw': #Classic DTW algorithm

                    words_prepared = words_preparation(traj[i], traj[j], path_jointlist)
                    distance[i, j] = (distance_computation_dtw(words_prepared, 'dtw'))
                    try:
                        distance[j, i] = (distance_computation_dtw(words_prepared, 'dtw'))
                    except:
                        pass

                elif alg_type == 'softdtw':#Differentiable SoftDTW version of DTW

                    words_prepared = words_preparation(traj[i], traj[j], path_jointlist)
                    distance[i, j] = (distance_computation_dtw(words_prepared, 'softdtw'))
                    try:
                        distance[j, i] = (distance_computation_dtw(words_prepared, 'softdtw'))
                    except:
                        pass

                elif alg_type == 'method_combination': #Signal resample and distance computation separately
                    
                    selected_joints_idxs = [3,4,5,32,33,34] # RightArm, RightForeArm, RightHand, LeftArm, LeftForeArm, LeftHand in jointlist
                    distances_joints = []

                    for k in range(len(selected_joints_idxs)):
                        resample_out = resample(traj[i][:,selected_joints_idxs[k],:], traj[j][:,selected_joints_idxs[k],:], resample_method, int_method, graph=0)
                        one_joint_distance = (compare(resample_out[1],resample_out[2], dist = distance_method))
                        distances_joints.append(one_joint_distance)

                    joints_distance_mean = np.mean(distances_joints)
                    distance[i, j] = joints_distance_mean
                    try:
                        distance[j, i] = joints_distance_mean
                    except:
                        pass
                else:
                    print('Nedefinován typ algoritmu.')
    if graph:
        plt.imshow(distance, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

    return distance


def resample(word1, word2, method = 'interpolation', int_method = 'linear', graph = 0):
    """Resamples word2 signal to the length of the word1 signal
    Args:
        word1 [list]: signal, to ots length is being the first one resampled
        word2 [list]: signal that is being resampled
        method [string]: the method of resampling used, 'interpolation', 'fourier'
        int_method [string]: the method of interpolation if the method 'interpolation' is selected
        graph [boolean]: yes or no to display graph with comparison of old and resampled signal

    Returns:
        [list]: List of lists, initial word2 signal restructualized, resampled word2 signal, initial word1 signal restructualized
    """
    if method == 'fourier':
        joint_list = get_jointlist(path_jointlist)

        word1_restruct = np.zeros(shape=(3),dtype = object)
        word2_restruct = np.zeros(shape=(3),dtype = object)
        word_resampled = np.zeros(shape=(3),dtype = object)

        for i in range(3):
            word1_restruct[i] = [frame[i] for frame in word1]
            word2_restruct[i] = [frame[i] for frame in word2]
        x = np.linspace(0, len(word1_restruct[0]), len(word2_restruct[0]))

        for i in range(3):
            word_resampled[i] = signal.resample(word2_restruct[i], len(word1_restruct[i]))
        xresampled = np.linspace(0, len(word1_restruct[0]), len(word_resampled[0]))
        
        if graph:
            """
            #3 different figures
            channels = ['x','y','z']
            for i in range(3):
                plt.figure()
                plt.plot(word2_restruct[i],'g')
                plt.plot(word_resampled[i],'r')
                plt.title('Resample kanálu {} trajektorie kloubu {} ve slově "{}" '.format(channels[i],joint_list[joint],word1_meta[0]))
                plt.legend(['Initial data signal','Interpolated data signal'], loc='best')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.show()
            """
            interpolated_for_graph = np.zeros(shape=(3),dtype = object)
            for j in range(len(word_resampled)):
                for i in range(len(word_resampled[j])):
                    if i%4 == 0:
                        interpolated_for_graph[j] = np.append(interpolated_for_graph[j],word_resampled[j][i])

            interpolated_for_graph[0] = np.delete(interpolated_for_graph[0],0)
            interpolated_for_graph[1] = np.delete(interpolated_for_graph[1],0)
            interpolated_for_graph[2] = np.delete(interpolated_for_graph[2],0)

            xinterpgraph = np.linspace(0, len(word1_restruct[0]), len(interpolated_for_graph[0]))
            
            mpl.style.use('seaborn')
            fig, ax = plt.subplots(3,1, sharex=True)
            ax[0].plot([0.0,9.21428571], [-7.969536608198854,-7.974704499955581], marker="*", markersize=6, linewidth=0.3, color = 'r')
            ax[0].plot(x, word2_restruct[0], marker='D', color = 'k', linewidth=0.3, markersize=4)
            ax[0].plot(xresampled, word_resampled[0], color = 'r', linewidth=0.5, markersize=4)
            ax[0].plot(xinterpgraph, interpolated_for_graph[0],'*',markersize=6, color = 'r')
            ax[0].set_title('Převzorkovaná osa X')
            ax[1].plot(x, word2_restruct[1], marker='D', color = 'k', linewidth=0.3, markersize=4)
            ax[1].plot(xresampled, word_resampled[1], color = 'r', linewidth=0.5, markersize=4)
            ax[1].plot(xinterpgraph, interpolated_for_graph[1],'*',markersize=6, color = 'r')
            ax[1].set_title('Převzorkovaná osa Y')
            ax[2].plot(x, word2_restruct[2], marker='D', color = 'k', linewidth=0.3, markersize=4)
            ax[2].plot(xresampled, word_resampled[2], color = 'r', linewidth=0.5, markersize=4)
            ax[2].plot(xinterpgraph, interpolated_for_graph[2],'*',markersize=6, color = 'r')
            ax[2].set_title('Převzorkovaná osa Z')
            ax[2].set_xlabel('čas [snímek]')
            ax[1].set_ylabel('vzdálenost od počátku [cm]')
            fig.legend(['Převzorkovaný signál','Původní signál'], loc='upper right')
            #plt.suptitle('Furierova transformace trajektorie kloubu {} na délku slova "{}" z délky slova "{}"'.format(joint_list[joint],word1_meta[0],word2_meta[0]),fontsize=15)
            plt.show()
            """
            #3D graph
            ax = plt.axes(projection='3d')
            ax.plot3D(word_resampled[0],word_resampled[1],word_resampled[2], 'r.')
            ax.plot3D(word2_restruct[0],word2_restruct[1],word2_restruct[2], 'black')
            ax.plot3D(word1_restruct[0],word1_restruct[1],word1_restruct[2], 'blue')
            ax.plot3D(word2_restruct[0][0], word2_restruct[1][0], word2_restruct[2][0], 'k*')
            plt.legend(['resampled data','initial data', 'data from longer dataframe', 'starting point','hips location'], loc='best')
            plt.title('Trajektorie kloubu {}'.format(joint_list[joint]))
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()
            """
        return [word2_restruct,word_resampled,word1_restruct]

    if method == 'interpolation':
        joint_list = get_jointlist(path_jointlist)

        word1_restruct = np.zeros(shape=(3),dtype = object)
        word2_restruct = np.zeros(shape=(3),dtype = object)
        word_interpolated = np.zeros(shape=(3),dtype = object)
        if graph:
            word_interpolatedx = np.zeros(shape=(3),dtype = object)
            word_interpolatedxx = np.zeros(shape=(3),dtype = object)

        for i in range(3):
            word1_restruct[i] = [frame[i] for frame in word1]
            word2_restruct[i] = [frame[i] for frame in word2]
        x = np.linspace(0, len(word1_restruct[0]), len(word2_restruct[0]))

        for i in range(3):
            word_interpolated[i] = interpolate_signal(word2_restruct[i],len(word1_restruct[i]), int_method)
            if graph:
                word_interpolatedx[i] = interpolate_signal(word2_restruct[i],len(word1_restruct[i]*10), 'linear')
                word_interpolatedxx[i] = interpolate_signal(word2_restruct[i],len(word1_restruct[i]), 'linear')
        if graph:
            xinterp = np.linspace(0, len(word1_restruct[0]), len(word_interpolated[0]))
            xinterpx = np.linspace(0, len(word1_restruct[0]), len(word_interpolatedx[0]*10))
        
        if graph:
            
            interpolated_for_graph = np.zeros(shape=(3),dtype = object)
            for j in range(len(word_interpolated)):
                for i in range(len(word_interpolated[j])):
                    if i%4 == 0:
                        interpolated_for_graph[j] = np.append(interpolated_for_graph[j],word_interpolated[j][i])

            interpolated_for_graph[0] = np.delete(interpolated_for_graph[0],0)
            interpolated_for_graph[1] = np.delete(interpolated_for_graph[1],0)
            interpolated_for_graph[2] = np.delete(interpolated_for_graph[2],0)

            xinterpgraph = np.linspace(0, len(word1_restruct[0]), len(interpolated_for_graph[0]))
            
            interpolated_for_graphx = np.zeros(shape=(3),dtype = object)
            for j in range(len(word_interpolatedxx)):
                for i in range(len(word_interpolatedxx[j])):
                    if i%4 == 0:
                        interpolated_for_graphx[j] = np.append(interpolated_for_graphx[j],word_interpolatedxx[j][i])

            interpolated_for_graphx[0] = np.delete(interpolated_for_graphx[0],0)
            interpolated_for_graphx[1] = np.delete(interpolated_for_graphx[1],0)
            interpolated_for_graphx[2] = np.delete(interpolated_for_graphx[2],0)

            xinterpgraphx = np.linspace(0, len(word1_restruct[0]), len(interpolated_for_graphx[0]))
            
            mpl.style.use('seaborn')
            fig, ax = plt.subplots(3,1, sharex=True)
            ax[0].plot(x, word2_restruct[0],marker='D', color = 'k', linewidth=0.3, markersize=4)
            ax[0].plot([0.0,4.03125], [-4.362433016513284,-4.306118578837304],marker="*", markersize=6, linewidth=0.7, color = 'r')
            ax[0].plot(xinterp, word_interpolated[0],'r', linewidth=0.7)
            ax[0].plot(xinterpgraph, interpolated_for_graph[0],'*',markersize=6, color = 'r')
            ax[0].set_title('Interpolovaná osa X')

            ax[1].plot(x, word2_restruct[1],marker='D', color = 'k', linewidth=0.3, markersize=4)
            ax[1].plot(xinterp, word_interpolated[1],'r', linewidth=0.7)
            ax[1].plot(xinterpgraph, interpolated_for_graph[1],'*', markersize=6, color = 'r')
            ax[1].set_title('Interpolovaná osa Y')

            ax[2].plot(x, word2_restruct[2],marker='D', color = 'k', linewidth=0.3, markersize=4)
            ax[2].plot(xinterp, word_interpolated[2],'r', linewidth=0.7)
            ax[2].plot(xinterpgraph, interpolated_for_graph[2],'*', markersize=6, color = 'r')
            ax[2].set_title('Interpolovaná osa Z')
            ax[2].set_xlabel('čas [snímek]')
            ax[1].set_ylabel('vzdálenost od počátku [cm]')
            fig.legend(['Původní signál','Interpolovaný signál'], loc='upper right')
            plt.show()

            #Rozdil linearni a kubicke interpolace
            fig = plt.figure()
            plt.plot(x, word2_restruct[0],marker='D', color = 'k', linewidth=0.3, markersize=7)
            plt.plot([0,4.03125], [-4.362433016513284,-4.306118578837304], marker=".", markersize=10, linewidth=0.7, color = 'r')
            plt.plot([0,4.03125], [-4.362433016513284,-4.306118578837304], marker=".", markersize=10, linewidth=0.7, color = 'b')
            plt.plot(xinterp, word_interpolated[0],'r', linewidth=0.7)
            plt.plot(xinterpgraph, interpolated_for_graph[0],'.',markersize=10, color = 'r')
            plt.plot(xinterpx, word_interpolatedx[0],'b', linewidth=0.7)
            plt.plot(xinterpgraphx, interpolated_for_graphx[0],'.',markersize=10, color = 'b')

            plt.xlabel('čas [snímek]')
            plt.ylabel('vzdálenost od počátku [cm]')
            fig.legend(['Původní signál','Kubicky interpolovaný signál', 'Lineárně interpolovaný signál'], loc='upper right')
            plt.show()

        return [word2_restruct,word_interpolated,word1_restruct]


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


def compare(word1, word2, dist = 'euclidean'):
    """Counts the distance between 2 3D signals using one of implemented metrics
    Args:
        word1 [list]: The first signal for the computation
        word2 [list]: The second signal for the computation
        dist [string]: A metrics used for distance computation

    Returns:
        [double]: The distance between 2 given signals
    """
    distance = 0
    word1 = np.array([word1[0],word1[1],word1[2]])
    word2 = np.array([word2[0],word2[1],word2[2]])
    if dist == 'euclidean':
        for i in range(len(word1[1])):
            distance += spatial.distance.euclidean(word1[:,i],word2[:,i])
    elif dist == 'hamming':
        for i in range(len(word1[1])):
            distance += spatial.distance.hamming(word1[:,i],word2[:,i])
    elif dist == 'minkowsky':
        for i in range(len(word1[1])):
            distance += spatial.distance.minkowski(word1[:,i],word2[:,i], p=3)
    elif dist == 'mahalanobis':
        for i in range(len(word1[1])):
            V = np.cov(np.array([word1[:,i],word2[:,i]]).T)
            IV = np.linalg.pinv(V)
            distance += spatial.distance.mahalanobis(word1[:,i],word2[:,i], IV)
    elif dist == 'pearson':
        return np.corrcoef(word1, word2)
    elif dist == 'correlationDistance':
        return dcor.distance_correlation(word1, word2)
    elif dist == 'canberra':
        for i in range(len(word1[1])):
            distance += spatial.distance.canberra(word1[:,i],word2[:,i])
    elif dist == 'braycurtis':
        for i in range(len(word1[1])):
            distance += spatial.distance.braycurtis(word1[:,i],word2[:,i])
    elif dist == 'chebyshev':
        for i in range(len(word1[1])):
            distance += spatial.distance.chebyshev(word1[:,i],word2[:,i])
    elif dist =='fréchet':
        distance = similaritymeasures.area_between_two_curves(word1, word2)
    else:
        for i in range(len(word1[1])):
            distance += spatial.distance.euclidean(word1[:,i],word2[:,i])
    return distance

if __name__ == '__main__':
    #source_dir = '/home/jedle/data/Sign-Language/_source_clean/'
    source_dir = 'Sign_Language_BP/'
    # bvh_dir = os.path.join(source_dir, 'bvh/')  # all bvh files takes and dictionaries
    bvh_dir = 'Sign_Language_BP/data_bvh/'
    bvh_dict = 'Sign_Language_BP/bvh_dict/'
    #glo_dir = 'source_data/'
    glo_dir = 'Sign_Language_BP/source_data/'
    #word_dir = 'source_words/'
    word_dir = 'Sign_Language_BP/source_words/'
    #path_jointlist = 'data/joint_list.txt'
    path_jointlist = 'Sign_Language_BP/data/joint_list.txt'
    #path_metadata = 'data/meta.pkl'
    path_metadata = 'Sign_Language_BP/data/meta.pkl'
    #path_trajectory = 'data/traj.pkl'
    path_trajectory ='Sign_Language_BP/data/traj.pkl'
    #dict_file = os.path.join(source_dir, 'ultimate_dictionary2.txt')
    dict_file = 'Sign_Language_BP/data/ultimate_dictionary2.txt'

    # converts data from angular BVH to global positions (npy matrix)
    mine = False
    if mine:
        mine_data(bvh_dir, glo_dir)

    # creates sign numpy dictionary
    create = False
    if create:
        dict_items = SL_dict.read_dictionary(dict_file, 'dictionary_items')
        dict_takes = SL_dict.read_dictionary(dict_file, 'dictionary_takes')
        create_trajectory_matrix(dict_items + dict_takes)

    with open(path_jointlist, 'r') as f:
        joint_list = f.readlines()      # to je jenom pořadí markerů
    with open(path_metadata, 'rb') as pf:
        meta = pk.load(pf)              # metadata: nazev, puvod data (soubor), anotace
    with open(path_trajectory, 'rb') as pf:
        traj = pk.load(pf)              # trajektorie [item, frame, joint, channel]

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
    if sorting:
        sign_name_list = [m[0] for m in meta]
        unique_sign_list_unordered = set(sign_name_list)
        unique_count = []
        for item in unique_sign_list_unordered:
            matches = [m for m in sign_name_list if m == item]
            unique_count.append([item, len(matches)])

        unique_count.sort(key=lambda x: x[1], reverse=True)
        print((unique_count))

    test_dtw_one_word = False
    if test_dtw_one_word:
        word = 'zitra'
        one_word_dtw(word, path_jointlist, 20, 'softdtw', graph=1)
    
    test_resample = False
    if test_resample:
        joint = 5
        word1 = traj[191][:, joint, :] #0. znak, vsechny snimky pro [joint]. joint, vsechny dimenze
        word1_meta = meta[191]

        word2 = traj[700][:, joint, :]
        word2_meta = meta[700]

        [word2_restruct,word_interpolated,word1_restruct] = resample(word2, word1,'interpolation', 'cubic', graph=1)
    
    test_signal_comparison = False
    if test_signal_comparison:
        joint = 3
        word1 = traj[900][:, joint, :]
        word1_meta = meta[900]

        word2 = traj[200][:, joint, :]
        word2_meta = meta[200]

        resample_out = resample(word2, word1, 'interpolation', 'linear', graph=0) #returns reorganized word1 and resampled word2
        kind = 'fréchet'
        distance = compare(resample_out[1],resample_out[2], dist = kind)

        print('{} counted over \'{}\' and \'{}\': {}'.format(kind, word1_meta[0], word2_meta[0], distance))

    compute_dtw_more_words = True
    if compute_dtw_more_words:
        
        alg_type = 'method_combination' # 'dtw', 'softdtw', 'method_combination'

        # Used only if 'method_combination' is selected:
        resample_type = 'interpolation' # 'interpolation', 'fourier'
        int_type = 'cubic' # 'linear', 'quadratic', 'cubic'
        distance_method = 'hamming' # 'euclidean', 'hamming', 'minkowsky', 'mahalanobis', 'pearson', 'correlationDistance', 'canberra', 'braycurtis', 'chebychev', 'fréchet'

        if alg_type == 'method_combination':
            start = timer()
            distance_matrix = compute(-1, alg_type, resample_type, int_type, distance_method, graph = 1)
            end = timer()

            print('Duration: {}'.format(end-start))
            pk_out = open("Sign_Language_BP/output_files/{}+{}+{}.pkl".format(int_type, resample_type, distance_method), 'wb')
            pk.dump(distance_matrix, pk_out)
            pk_out.close()
        else:
            start = timer()
            distance_matrix = compute(-1, alg_type, graph = 1) # DTW or SoftDTW
            end = timer()

            print('Duration: {}'.format(end-start))
            pk_out = open("Sign_Language_BP/output_files/{}.pkl".format(alg_type), 'wb')
            pk.dump(distance_matrix, pk_out)
            pk_out.close()