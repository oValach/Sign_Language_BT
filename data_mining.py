from lib import bvh2glo_simple, SL_dict
import os
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt


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
        np.save(os.path.join(out_directory, tmp_file_base_name + '.npy'), tmp_data_glo[1])


def create_trajectory_matrix(dictionary_data):
    """
    creates sign numpy dictionary
    :param dictionary_data:
    :return:
    """
    trajectory_list = []
    metadata_list = []

    for i, item in enumerate(dictionary_data):
        if all(['annotation_Filip_bvh_frame' in item.keys(), item['sign_id'] is not '', '!' not in item['sign_id']]):  # vyřazuje položky ve slovníku, které nejsou zpracovane
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

def distance_computation(word1, word2):
    computed_distance = np.size(word1, 0) - np.size(word2, 0)         # tohle je samozřejmě blbost, ale aby to něco vracelo
    return abs(computed_distance)

if __name__ == '__main__':
    #source_dir = '/home/jedle/data/Sign-Language/_source_clean/'
    source_dir = 'Sign_Language_BP/'
    #bvh_dir = os.path.join(source_dir, 'bvh/')  # all bvh files takes and dictionaries
    bvh_dir = 'Sign_Language_BP/data_bvh/'
    #glo_dir = 'source_data/'
    glo_dir = 'Sign_Language_BP/source_data/'
    #word_dir = 'source_words/'
    word_dir = 'Sign_Language_BP/source_words/'
    #path_jointlist = 'data/joint_list.txt'
    path_jointlist = os.path.join(source_dir, 'data/joint_list.txt')
    #path_metadata = 'data/meta.pkl'
    path_metadata = os.path.join(source_dir, 'data/meta.pkl')
    #path_trajectory = 'data/traj.pkl'
    path_trajectory = os.path.join(source_dir, 'data/traj.pkl')
    #dict_file = os.path.join(source_dir, 'ultimate_dictionary2.txt')
    dict_file = os.path.join(source_dir, 'data/ultimate_dictionary2.txt')

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
    if flexing: # access to data examples
        print('Ukázka, jak vypadá položka v proměnné "meta": {}'.format(meta[0]))
        print('Proměnná "traj" je list o délce {}, což je počet všech dat nehledě na význam'.format(len(traj)))
        print('první 3 položky v traj mají následující dimenze:\n{}\n{}\n{}'.format(np.shape(traj[0]), np.shape(traj[1]), np.shape(traj[2])))
        print('Takže dimenze jsou [frame, joint, kanál]')

        # Výběr kloubu pro vizualizaci
        joint = 'RightHand\n'
        joint_id = [i for i, n in enumerate(joint_list) if joint in n][0]  # tohle vrací celý list všech nalezených jointů, obsahujících řetězec joint, takže to vrací jednorozměrné pole, proto je tam ta [0], jakože vyberu první prvek z toho pole (abych neměl list, ale int)
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

    computing = True
    if computing:
        distance = np.zeros((len(traj), len(traj)))
        for i in range(len(traj)):
            for j in range(i):
                if i == j:
                    distance[i, j] = 0
                else:
                    distance[i, j] = distance[j, i] = (distance_computation(traj[i], traj[j]))

        print(np.shape(distance))
        plt.imshow(distance, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()
        pass

        for i in range(np.size(distance, 0)):
            tmp_slice = distance[i, :]   # vyberu jednu rádku confussion matice a najdu top několik nejlepších shod
            best10 = (tmp_slice.argsort()[:10][::-1])
            print('Nejlepší tři shody: '.format(best10))
            print('vybráno: {}'.format(meta[i]))
            for b in best10:
                print(b)
                print(tmp_slice[b])
                print(meta[b])
            break