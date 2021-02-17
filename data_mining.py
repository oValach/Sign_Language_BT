from lib import bvh2glo_simple, SL_dict
import os
import sys
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from dtaidistance import dtw


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
        if all(['annotation_Filip_bvh_frame' in item.keys(), item['sign_id'] is not '', '!' not in item['sign_id']]):
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
        [list]: A list of counted dtw values for each joint separately
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

    xdist = list()
    ydist = list()
    zdist = list()

    for key, val in data_prepared.items():

        dtwx = dtw.distance_fast(val[0][0], val[1][0], use_pruning=True)
        dtwy = dtw.distance_fast(val[0][1], val[1][1], use_pruning=True)
        dtwz = dtw.distance_fast(val[0][2], val[1][2], use_pruning=True)

        xdist.append(dtwx)
        ydist.append(dtwy)
        zdist.append(dtwz)

    xavg = np.mean(xdist)
    yavg = np.mean(ydist)
    zavg = np.mean(zdist)

    # samozrejme docasne reseni, secteni dtw vzdalenosti pro x,y a z pres joints
    output = np.mean([xavg,yavg,zavg])

    return output


def get_jointlist(path_jointlist):
    file_joints = open(path_jointlist, 'r')
    joints = file_joints.readlines()
    joints = [f.rstrip() for f in joints]
    return joints
    
    
def compute_one_word(word, path_jointlist, number_of_mins):

    sign_name_list = [m[0] for m in meta]
    try:
        idx = sign_name_list.index(word)
    except:
        print('Slovo nenalezeno.')
        sys.exit()
    word_traj = traj[idx]

    distance = np.zeros((len(traj)))
    for i in range(len(traj)):
        words_prepared = words_preparation(traj[i], word_traj, path_jointlist)
        distance[i] = (distance_computation_dtw(words_prepared))

    best = (distance.argsort()[:number_of_mins][::-1])

    print('Nejlepších {} shod se slovem: {}'.format(number_of_mins,word))
    for item in best[::-1]:
        print('{}: {}'.format(meta[item], distance[item]))
    
    return best


def resample_to_longer_fourier(word1,word2,graph = 0):
    joint_list = get_jointlist(path_jointlist)

    if len(word1) <= len(word2):
        longer_word_temp = word2
        shorter_word_temp = word1
    else:
        longer_word_temp = word1
        shorter_word_temp = word2

    shorter_word = np.zeros(shape=(3),dtype = object)
    longer_word = np.zeros(shape=(3),dtype = object)
    word_resampled = np.zeros(shape=(3),dtype = object)

    for i in range(3):
        shorter_word[i] = [frame[i] for frame in shorter_word_temp]
        longer_word[i] = [frame[i] for frame in longer_word_temp]
    for i in range(3):
        word_resampled[i] = signal.resample(shorter_word[i], len(longer_word[i])+1)
        word_resampled[i] = word_resampled[i][:-1]
    
    if graph:
        ax = plt.axes(projection='3d')
        ax.plot3D(word_resampled[0],word_resampled[1],word_resampled[2], 'r.')
        ax.plot3D(shorter_word[0],shorter_word[1],shorter_word[2], 'black')
        ax.plot3D(longer_word[0],longer_word[1],longer_word[2], 'blue')
        ax.plot3D(shorter_word[0][0], shorter_word[1][0], shorter_word[2][0], 'k*')
        plt.legend([ 'resampled data','initial data', 'data from longer dataframe', 'starting point','hips location'], loc='best')
        plt.title('Trajektorie kloubu {}'.format(joint_list[joint]))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    return word_resampled


def interpolate_signal(signal, final_length):
    x = np.r_[0:len(signal)-1:complex(len(signal),1)]
    f = interpolate.interp1d(x,signal)

    to_interpolate = np.r_[0:len(signal)-1:complex(final_length,1)]
    signal_interpolated = f(to_interpolate)
    return signal_interpolated


def resample_to_longer_interpolation(word1,word2, graph = 0):
    joint_list = get_jointlist(path_jointlist)

    if len(word1) <= len(word2):
        longer_word_temp = word2
        shorter_word_temp = word1
    else:
        longer_word_temp = word1
        shorter_word_temp = word2

    shorter_word = np.zeros(shape=(3),dtype = object)
    longer_word = np.zeros(shape=(3),dtype = object)
    word_interpolated = np.zeros(shape=(3),dtype = object)

    for i in range(3):
        shorter_word[i] = [frame[i] for frame in shorter_word_temp]
        longer_word[i] = [frame[i] for frame in longer_word_temp]

    for i in range(3):
        word_interpolated[i] = interpolate_signal(shorter_word[i],len(longer_word[i]))

    if graph:
        ax = plt.axes(projection='3d')
        ax.plot3D(word_interpolated[0],word_interpolated[1],word_interpolated[2], 'r.')
        ax.plot3D(shorter_word[0],shorter_word[1],shorter_word[2], 'black')
        ax.plot3D(longer_word[0],longer_word[1],longer_word[2], 'blue')
        ax.plot3D(shorter_word[0][0], shorter_word[1][0], shorter_word[2][0], 'k*')
        plt.legend(['interpolated data','initial data', 'data from longer dataframe', 'starting point','hips location'], loc='best')
        plt.title('Trajektorie kloubu {}'.format(joint_list[joint]))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    return word_interpolated

    
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

    computing = False
    if computing:
        distance = np.zeros((len(traj), len(traj)))
        for i in range(len(traj)):
            for j in range(i):
                if i == j:
                    distance[i, j] = 0
                else:
                    words_prepared = words_preparation(traj[i], traj[j], path_jointlist)
                    distance[i, j] = distance[j, i] = (distance_computation_dtw(words_prepared))

        print(np.shape(distance))
        plt.imshow(distance, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()
        pass

        for i in range(np.size(distance, 0)):
            # vyberu jednu rádku confussion matice a najdu top několik nejlepších shod
            tmp_slice = distance[i, :]
            best10 = (tmp_slice.argsort()[:10][::-1])
            print('Nejlepší tři shody: {}'.format(best10))
            print('vybráno: {}'.format(meta[i]))
            for b in best10:
                print(b)
                print(tmp_slice[b])
                print(meta[b])
            break

    computing_one_word = False
    if computing_one_word:
        word = 'během'
        compute_one_word(word, path_jointlist, 20)

    resample = True
    if resample:
        joint = 5
        word1 = traj[0][:, joint, :] #0. znak, vsechny snimky pro [joint]. joint, vsechny dimenze
        word1_meta = meta[0]

        word2 = traj[110][:, joint, :]
        word2_meta = meta[110]

        word2_resampled = resample_to_longer_fourier(word1,word2,1)