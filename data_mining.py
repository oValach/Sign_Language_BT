from lib import bvh2glo_simple, SL_dict
import os
import sys
import collections
import numpy as np
import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, interpolate, spatial
from dtaidistance import dtw
from dtaidistance import dtw_ndim
from dtw import dtw as dtw_slower


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
    """Counts dtw with euclidean distance between given word trajectory coordinate signals
    Args:
        word1 (list): First trajectory to count in dtw fcn
        word2 (list): Second trajectory to count in dtw fcn
        path_jointlist (string): A path to the joint_list.txt file, for example 'Sign_Language_BP/data/joint_list.txt'

    Returns:
        [list]: A list of frames values for each joint separately
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


# VICEDIM VERZE DTW, zkusit na testovacich slovech, jinem dtw algoritmus
def distance_computation_dtw(data_prepared):

    dtw_dist = list()

    for key, val in data_prepared.items():

        dtw_dist.append(dtw_ndim.distance_fast(np.transpose(val[0]),np.transpose(val[1])))

    return np.mean(dtw_dist)


def get_jointlist(path_jointlist):
    file_joints = open(path_jointlist, 'r')
    joints = file_joints.readlines()
    joints = [f.rstrip() for f in joints]
    return joints
    

def one_word_dtw(word, path_jointlist, number_of_mins,graph = 1):

    sign_name_list = [m[0] for m in meta]
    try:
        idx = sign_name_list.index(word)
    except:
        print('Slovo nenalezeno.')
        sys.exit()

    occurences = sign_name_list.count(word)

    print('{} vyskytu slova {}'.format(occurences, word))
    word_index = input('Index instance slova na testovani (0,{}): '.format(occurences-1))
    word_traj = traj[idx+int(word_index)]

    distance = np.zeros((len(traj)))
    for i in range(len(traj)):
        words_prepared = words_preparation(traj[i], word_traj, path_jointlist)
        distance[i] = (distance_computation_dtw(words_prepared))

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
        plt.title('Rozložení identických znaků ke znaku "{}" v seřazeném datasetu'.format(word))
        plt.grid()
        plt.show()

    print('Vyskytu slova v {} nejmensich vysledcich: {}'.format(100,best100_occurences))
    print('Vyskytu slova v {} nejmensich vysledcich: {}'.format(500,best500_occurences))
    print('Nejlepších {} shod s {}.instanci slova: {}'.format(number_of_mins,word_index,word))
    for item in bestentered:
        print('{}: {}'.format(meta[item], distance[item]))
    
    return bestentered


def compute_dtw(word_amount, graph = 0):

    if word_amount == -1:
        distance = np.zeros((len(traj), len(traj)))
    else:
        distance = np.zeros((int(word_amount), len(traj)))
    
    for i in range(len(distance)):
        for j in range(len(distance[0])):
            if i == j:
                distance[i, j] = 0
            else:
                words_prepared = words_preparation(traj[i], traj[j], path_jointlist)
                distance[i, j] = (distance_computation_dtw(words_prepared))
                try:
                    distance[j, i] = (distance_computation_dtw(words_prepared))
                except:
                    pass
    if graph:
        plt.imshow(distance, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

    return distance


def word_resample(word1,word2,method,graph = 0):
    if method == 'fourier':
        joint_list = get_jointlist(path_jointlist)

        word1_restruct = np.zeros(shape=(3),dtype = object)
        word2_restruct = np.zeros(shape=(3),dtype = object)
        word_resampled = np.zeros(shape=(3),dtype = object)

        for i in range(3):
            word1_restruct[i] = [frame[i] for frame in word1]
            word2_restruct[i] = [frame[i] for frame in word2]
        x = np.linspace(0, len(word1_restruct[0]), len(word2_restruct[0]), endpoint=False)

        for i in range(3):
            word_resampled[i] = signal.resample(word2_restruct[i], len(word1_restruct[i]))
        xresampled = np.linspace(0, len(word1_restruct[0]), len(word_resampled[0]), endpoint=False)
        
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
            fig, ax = plt.subplots(3,1, sharex=True)
            ax[0].plot(x, word2_restruct[0],'g')
            ax[0].plot(xresampled, word_resampled[0],'*r--', linewidth=0.5, markersize=5)
            ax[0].set_title('Resampled X axis')
            ax[1].plot(x, word2_restruct[1],'g')
            ax[1].plot(xresampled, word_resampled[1],'*r--', linewidth=0.5, markersize=5)
            ax[1].set_title('Resampled Y axis')
            ax[2].plot(x, word2_restruct[2],'g')
            ax[2].plot(xresampled, word_resampled[2],'*r--', linewidth=0.5, markersize=5)
            ax[2].set_title('Resampled Z axis')
            fig.legend(['Initial data signal','Resampled data signal'], loc='upper right')
            plt.suptitle('Fourier resample of {} to len of word "{}" from len of word "{}"'.format(joint_list[joint],word1_meta[0],word2_meta[0]),fontsize=15)
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
        return [word_resampled,word1_restruct]

    if method == 'poly':
        joint_list = get_jointlist(path_jointlist)

        word1_restruct = np.zeros(shape=(3),dtype = object)
        word2_restruct = np.zeros(shape=(3),dtype = object)
        word_resampled = np.zeros(shape=(3),dtype = object)

        for i in range(3):
            word1_restruct[i] = [frame[i] for frame in word1]
            word2_restruct[i] = [frame[i] for frame in word2]
        x = np.linspace(0, len(word1_restruct[0]), len(word2_restruct[0]), endpoint=False)

        for i in range(3):
            word_resampled[i] = signal.resample_poly(word2_restruct[i], len(word1_restruct[i]), len(word2_restruct[i]))
        xresampled = np.linspace(0, len(word1_restruct[0]), len(word_resampled[0]), endpoint=False)
        
        if graph:
            fig, ax = plt.subplots(3,1, sharex=True)
            ax[0].plot(x, word2_restruct[0],'g')
            ax[0].plot(xresampled, word_resampled[0],'*r--', linewidth=0.5, markersize=5)
            ax[0].set_title('Resampled X axis')
            ax[1].plot(x, word2_restruct[1],'g')
            ax[1].plot(xresampled, word_resampled[1],'*r--', linewidth=0.5, markersize=5)
            ax[1].set_title('Resampled Y axis')
            ax[2].plot(x, word2_restruct[2],'g')
            ax[2].plot(xresampled, word_resampled[2],'*r--', linewidth=0.5, markersize=5)
            ax[2].set_title('Resampled Z axis')
            fig.legend(['Initial data signal','Interpolated data signal'], loc='upper right')
            plt.suptitle('Fourier resample of {} to len of word "{}" from len of word "{}"'.format(joint_list[joint],word1_meta[0],word2_meta[0]),fontsize=15)
            plt.show()

        return [word_resampled,word1_restruct]

    if method == 'interpolation':
        joint_list = get_jointlist(path_jointlist)

        word1_restruct = np.zeros(shape=(3),dtype = object)
        word2_restruct = np.zeros(shape=(3),dtype = object)
        word_interpolated = np.zeros(shape=(3),dtype = object)

        for i in range(3):
            word1_restruct[i] = [frame[i] for frame in word1]
            word2_restruct[i] = [frame[i] for frame in word2]
        x = np.linspace(0, len(word1_restruct[0]), len(word2_restruct[0]), endpoint=False)

        for i in range(3):
            word_interpolated[i] = interpolate_signal(word2_restruct[i],len(word1_restruct[i]))
        xinterp = np.linspace(0, len(word1_restruct[0]), len(word_interpolated[0]), endpoint=False)
        
        if graph:
            fig, ax = plt.subplots(3,1, sharex=True)
            ax[0].plot(x, word2_restruct[0],'g')
            ax[0].plot(xinterp, word_interpolated[0],'*r--', linewidth=0.5, markersize=5)
            ax[0].set_title('Interpolated X axis')
            ax[1].plot(x, word2_restruct[1],'g')
            ax[1].plot(xinterp, word_interpolated[1],'*r--', linewidth=0.5, markersize=5)
            ax[1].set_title('Interpolated Y axis')
            ax[2].plot(x, word2_restruct[2],'g')
            ax[2].plot(xinterp, word_interpolated[2],'*r--', linewidth=0.5, markersize=5)
            ax[2].set_title('Interpolated Z axis')
            fig.legend(['Initial data signal','Interpolated data signal'], loc='upper right')
            plt.suptitle('Interpolation of {} to len of word "{}" from len of word {}'.format(joint_list[joint],word1_meta[0],word2_meta[0]),fontsize=15)
            plt.show()

        return [word_interpolated,word1_restruct]


def interpolate_signal(signal, final_length):
    x = np.r_[0:len(signal)-1:complex(len(signal),1)]
    f = interpolate.interp1d(x,signal,kind='linear')

    to_interpolate = np.r_[0:len(signal)-1:complex(final_length,1)]
    signal_interpolated = f(to_interpolate)
    return signal_interpolated


def compare(word1,word2,dist = 'euclidean'):
    distance = 0
    if dist == 'euclidean':
        for i in range(3):
            distance += spatial.distance.euclidean(word1[i], word2[i])
    elif dist == 'hamming':
        for i in range(3):
            distance += spatial.distance.hamming(word1[i], word2[i])
    elif dist == 'minkowsky':
        for i in range(3):
            distance += spatial.distance.minkowski(word1[i], word2[i])
    elif dist == 'mahalanobis':
        for i in range(3):
            V = np.cov(np.array([word1, word2]).T)
            IV = np.linalg.inv(V)
            distance += spatial.distance.mahalanobis(word1[i], word2[i], IV)
    elif dist == 'correlation':
        corr = []
        for i in range(3):
            corr[i] = spatial.distance.correlation(word1[i], word2[i])
        return corr

    return distance

if __name__ == '__main__':
    #source_dir = '/home/jedle/data/Sign-Language/_source_clean/'
    source_dir = 'Sign_Language_BP/'
    # bvh_dir = os.path.join(source_dir, 'bvh/')  # all bvh files takes and dictionaries
    bvh_dir = 'data_bvh/'
    bvh_dict = 'bvh_dict/'
    #glo_dir = 'source_data/'
    glo_dir = 'source_data/'
    #word_dir = 'source_words/'
    word_dir = 'source_words/'
    #path_jointlist = 'data/joint_list.txt'
    path_jointlist = 'data/joint_list.txt'
    #path_metadata = 'data/meta.pkl'
    path_metadata = 'data/meta.pkl'
    #path_trajectory = 'data/traj.pkl'
    path_trajectory ='data/traj.pkl'
    #dict_file = os.path.join(source_dir, 'ultimate_dictionary2.txt')
    dict_file = 'data/ultimate_dictionary2.txt'

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

    computing_one_word_dtw = False
    if computing_one_word_dtw:
        word = 'zitra'
        one_word_dtw(word, path_jointlist, 20, graph=1)
    
    resample = False
    if resample:
        joint = 5
        word1 = traj[0][:, joint, :] #0. znak, vsechny snimky pro [joint]. joint, vsechny dimenze
        word1_meta = meta[0]

        word2 = traj[110][:, joint, :]
        word2_meta = meta[110]

        word2_resampled = word_resample(word2,word1,'fourier',graph=1)[0]

    compute_dtw = False
    if compute_dtw:
        compute_dtw(5,1)

    compare_signals = True
    if compare_signals:
        joint = 5
        word1 = traj[0][:, joint, :]
        word1_meta = meta[0]

        word2 = traj[900][:, joint, :]
        word2_meta = meta[900]

        resample_out = word_resample(word2,word1,'interpolation',graph=1) #returns reorganized word1 and resampled word2
        kind = 'euclidean'
        distance = compare(resample_out[0],resample_out[1], dist = kind)

        print('{} counted over {} and {}: {}'.format(kind, word1_meta[0], word2_meta[0], distance))