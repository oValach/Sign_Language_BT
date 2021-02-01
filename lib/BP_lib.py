def get_trajectory(trajectory, start, end):
    """Returns the trajectory coordinates between given frames

    Args:
        trajectory (list): Trajektory of input record in absolute coordinates
        start (int): Starting frame of output selection
        end (int): Ending frame of output selection

    Returns:
        [list]: Trajektory cut between start and end frame
    """
    word_trajectory = trajectory[start:end, :, :]
    return word_trajectory

def find_word(word, amount, path_bvh, path_converted):
    """Finds first [amount] of occurences of word across all files

    Args:
        word (string): A word to be found
        amount (int): A number of occurencer of the word to find
        path (string): A path to the folder with .bvh data, for example 'Sign_Language_BP/data_bvh'
        path_converted (string): A path to the folder with trajectory and dictionary in .pickle for each .bvh file

    Returns:
        [list]: List of trajetories of found occurences of the given word
    """

    import os
    import pickle as pk

    file_list = os.listdir(path_bvh)

    # vyhnuti se slozkam a ostatnim souborum
    file_list = [f for f in file_list if ('.bvh' in f)]

    if amount != 1:  # pozaduji jeden vyskyt -> nechci vysledek appendovat do pole
        word_trajectory = []
    current_amount = 0
    test_file = 0
    for filepath in file_list:  # iterování přes jednotlivé soubory
        test_file += 1
        # nahrani prepoctenych dat z angularnich - dictionary, trajectory
        [dictionary, trajectory] = import_abs_data(filepath, path_converted)

        number = 0
        for _, val in enumerate(dictionary):

            if current_amount == amount:  # nalezen pozadovany pocet
                break

            number += 1
            for tmp_key in val.keys():
                if val[tmp_key] == '':  # znak bez vyznamu
                    break
                # pokud byl nalezen hledany znak
                if tmp_key == 'sign_id' and val[tmp_key] == word and (amount != 1):

                    current_amount += 1
                    start = val['annotation_Filip_bvh_frame'][0]
                    end = val['annotation_Filip_bvh_frame'][1]

                    # fce jejiz vysledek ulozi do spolecneho pole trajektorii, nazev souboru a start a end snimek
                    word_trajectory.append([get_trajectory(trajectory, start, end), filepath, start, end])

                # chci vratit pouze jeden vyskyt znaku
                elif tmp_key == 'sign_id' and val[tmp_key] == word and (amount == 1):
                    current_amount += 1
                    start = val['annotation_Filip_bvh_frame'][0]
                    end = val['annotation_Filip_bvh_frame'][1]
                    word_trajectory = [get_trajectory(trajectory, start, end), filepath, start, end]

        if amount == -1 and test_file < 33:
            continue
        elif amount == -1 and test_file == 33:
            print('Cetnost slova "' + str(word)+'" je: ' + str(current_amount))
            #pk_out = open("Sign_Language_BP/data_trajectory/"+str(word)+".pickle", 'wb')
            #pk.dump(word_trajectory, pk_out)
            #pk_out.close()
            break
        elif current_amount < amount:  # nenalezen pozadovany pocet vyskytu - > pokracuj
            continue
        elif current_amount < amount and test_file == 33:  # neexistuje pozadovany pocet vyskytu zadaneho znaku
            print('Pozadovana prilis vysoka hodnota. Cetnost slova ' + str(word)+': ' + str(current_amount))
            break
        else:  # nalezen pozadovany pocet znaku
            print('Nalezeno pozadovane mnozstvi slov.')
            break
    return(word_trajectory)


def count_words(lower_limit, graph, path, path_converted):
    """Counts the frequency of words in speeches

    Args:
        lower_limit (int): A minimum amount of occurences for the word to be returned in an output list
        graph (boolean): Yes/No for the graph to show
        path (string): A path to the folder with .bvh data, for example 'Sign_Language_BP/data_bvh'
        path_converted (string): A path to the folder with trajectory and dictionary in .pickle for each .bvh file

    Returns:
        [dictionary]: key = given word, value = number of occurences
    """

    import os
    from collections import OrderedDict
    import matplotlib.pyplot as plt
    import operator
    import numpy as np

    lower_limit -= 1
    file_list = os.listdir(path)
    # vyhnuti se slozkam a ostatnim souborum
    file_list = [f for f in file_list if ('bvh' in f)]

    test_file = 0
    word_dict = {}  # slovnik znaku
    for filepath in file_list:  # iterování přes jednotlivé soubory
        test_file += 1
        # print('Prohledavani souboru cislo ' + str(test_file))

        # nahrani prepoctenych dat z angularnich - dictionary, trajectory
        [dictionary, _] = import_abs_data(filepath, path_converted)

        for _, val in enumerate(dictionary):
            for tmp_key in val.keys():
                if val[tmp_key] == '':  # znak bez vyznamu
                    break
                if tmp_key == 'sign_id' and (val[tmp_key] not in word_dict.keys()):
                    word_dict[val[tmp_key]] = 1  # novy znak
                elif tmp_key == 'sign_id' and (val[tmp_key] in word_dict.keys()):
                    word_dict[val[tmp_key]] += 1  # opakujici se znak

    # serazeni slovniku cetnosti sestupne
    word_counts_sorted = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)
    word_counts_sorted_dict = OrderedDict()
    for k, v in word_counts_sorted:
        word_counts_sorted_dict[k] = v
    # vykresleni histogramu, kde je cetnost >= lower_limit
    temp_dict = dict((k, v) for k, v in word_counts_sorted_dict.items() if v >= lower_limit)
    if graph == True:
        y_pos = np.arange(len(temp_dict))
        plt.barh(y_pos, temp_dict.values(), align='center', color='g')
        plt.yticks(y_pos, temp_dict.keys())
        plt.xticks(np.arange(2, 66, 2.0))
        plt.gca().xaxis.grid(True)
        plt.show()

    return(word_counts_sorted_dict)


def dtw_dist(word1, word2, path_jointlist):
    """Counts dtw with euclidean distance between given word trajectory coordinate signals

    Args:
        word1 (list): First trajectory to count in dtw fcn
        word2 (list): Second trajectory to count in dtw fcn
        path_jointlist (string): A path to the joint_list.txt file, for example 'Sign_Language_BP/data/joint_list.txt'

    Returns:
        [list]: A list of counted dtw values for each joint separately
    """
    from dtaidistance import dtw
    import numpy as np

    file_joints = open(path_jointlist, 'r')
    joints = file_joints.readlines()
    joints = [f.rstrip() for f in joints]
    dtw_dist = {}
    # přepočítání 3 dimenzionálních dat na 1 dimenzi porovnáváním po částech těla
    for i in range(len(joints)):  # joint
        seq1x = np.zeros(shape=(len(word1)),dtype=np.double)
        seq1y = np.zeros(shape=(len(word1)),dtype=np.double)
        seq1z = np.zeros(shape=(len(word1)),dtype=np.double)
        seq2x = np.zeros(shape=(len(word2)),dtype=np.double)
        seq2y = np.zeros(shape=(len(word2)),dtype=np.double)
        seq2z = np.zeros(shape=(len(word2)),dtype=np.double)

        for j in range(len(word1)):  # pocet snimku prvniho slova
            # skip nechtenych joints
            if any(char.isdigit() for char in joints[i]) or ('Head' in joints[i]) or ('Spine' in joints[i]) or ('Hips' in joints[i]):
                break
            if ('Shoulder' in joints[i]):
                seq1x[j] = word1[j][i][0]
                seq1y[j] = word1[j][i][1]  # souradnice prvniho slova za podminky ramene
                seq1z[j] = word1[j][i][2]
            elif ('Right' in joints[i]):
                seq1x[j] = word1[j][i][0] - RightShoulder[0][j]
                seq1y[j] = word1[j][i][1] - RightShoulder[1][j]  # souradnice prvniho slova s odectenim souradnic ramen
                seq1z[j] = word1[j][i][2] - RightShoulder[2][j]
            elif ('Left' in joints[i]):
                seq1x[j] = word1[j][i][0] - LeftShoulder[0][j]
                seq1y[j] = word1[j][i][1] - LeftShoulder[1][j]  # souradnice prvniho slova s odectenim souradnic ramen
                seq1z[j] = word1[j][i][2] - LeftShoulder[2][j]


        for j in range(len(word2)):  # pocet snimku druheho slova
            # skip nechtenych joints
            if any(char.isdigit() for char in joints[i]) or ('Head' in joints[i]) or ('Spine' in joints[i]) or ('Hips' in joints[i]):
                break
            if ('Shoulder' in joints[i]):
                seq2x[j] = word2[j][i][0]
                seq2y[j] = word2[j][i][1]  # souradnice druheho slova za podminky ramene
                seq2z[j] = word2[j][i][2]
            elif ('Right' in joints[i]):
                seq2x[j] = word2[j][i][0] - RightShoulder[3][j]
                seq2y[j] = word2[j][i][1] - RightShoulder[4][j]  # souradnice druheho slova s odectenim souradnic ramen
                seq2z[j] = word2[j][i][2] - RightShoulder[5][j]
            elif ('Left' in joints[i]):
                seq2x[j] = word2[j][i][0] - LeftShoulder[3][j]
                seq2y[j] = word2[j][i][1] - LeftShoulder[4][j]  # souradnice druheho slova s odectenim souradnic ramen
                seq2z[j] = word2[j][i][2] - LeftShoulder[5][j]

        # skip nechtenych joints
        if any(char.isdigit() for char in joints[i]) or ('Head' in joints[i]) or ('Spine' in joints[i]) or ('Hips' in joints[i]):
            continue
        # 1. prvek výstupu = distance, 2. = path
        dtwx = dtw.distance_fast(seq1x,seq2x,use_pruning=True)
        dtwy = dtw.distance_fast(seq1x,seq2x,use_pruning=True)
        dtwz = dtw.distance_fast(seq1x,seq2x,use_pruning=True)
        dtw_dist[joints[i]] = dtwx+dtwy+dtwz

        if ('RightShoulder' in joints[i]): # "zbaveni se" ucinku pohybu ramen na ruce - ulozeni jejich souradnic pro nasledne odecteni
            RightShoulder = np.array([seq1x,seq1y,seq1z,seq2x,seq2y,seq2z],dtype=object)
        if ('LeftShoulder' in joints[i]): # "zbaveni se" ucinku pohybu ramen na ruce - ulozeni jejich souradnic pro nasledne odecteni
            LeftShoulder = np.array([seq1x,seq1y,seq1z,seq2x,seq2y,seq2z],dtype=object)

    return dtw_dist


def compare_all(path_bvh, path_trajectory, path_jointlist, path_converted):
    """Computes dtw between majority of the words

    Args:
        path_bvh (string): A path to the folder with .bvh files
        path_trajectory (string): A path to the folder with each word trajectory in .pickle
        path_jointlist (string): A path to the joint_list.txt file, for example 'Sign_Language_BP/data/joint_list.txt'
        path_converted (string): A path to the folder with trajectory and dictionary in .pickle for each .bvh file

    Returns:
        [list]: A list of dictionaries with results of dtw for each chosen joint
    """
    import numpy as np
    import os
    import pickle as pk
    from timeit import default_timer as timer

    DTW_DICT_MATRIX = {
        'RightHand': None,
        'LeftHand': None,
        'RightArm': None,
        'LeftArm': None,
        'RightForeArm': None,
        'LeftForeArm': None,
    }

    # ziskani vsech vyskytu slov
    WORDS = []
    TOP_WORDS = []
    words_all = []
    words_tmp = count_words(1, False, path_bvh, path_converted)
    end_size_rows = 0
    end_size_cols = 0
    for i, v in enumerate(words_tmp):
        WORDS.append(v)
        end_size_cols = end_size_cols + words_tmp[v]
        if i < 12: # 12 nejcastejsich slov vuci vsem
            TOP_WORDS.append(v)
            end_size_rows = end_size_rows + words_tmp[v] #jejich cetnost = pocet radku distancni matice
        for i in range(words_tmp[v]):
            words_all.append(v) #seznam testovanych slov za sebou

    DTW_MATRIX1 = np.zeros(shape=(end_size_rows, 1000))
    DTW_MATRIX2 = np.zeros(shape=(end_size_rows, 1000))
    DTW_MATRIX3 = np.zeros(shape=(end_size_rows, 1000))
    DTW_MATRIX4 = np.zeros(shape=(end_size_rows, 1000))
    DTW_MATRIX5 = np.zeros(shape=(end_size_rows, 1000))
    DTW_MATRIX6 = np.zeros(shape=(end_size_rows, 1000))
    POSITION = np.zeros(shape=(1000), dtype=object)

    idx1 = 0
    for i in range(len(TOP_WORDS)):  # pocet iteraci = cetnost vybranych testovanych znaku
        word1 = TOP_WORDS[i]

        pk_word = open(os.path.join(path_trajectory,str(word1))+".pickle", "rb")
        words1_found = pk.load(pk_word)
        pk_word.close()

        #words1_found = find_word(word1, -1, 'Sign_Language_BP/data_bvh')

        for k in range(len(words1_found)): # pocet iteraci = cetnost prvniho znaku, ktery zrovna testuji
            idx1 += 1  # dalsi radek ve vysledne matici
            print(f"---------------------{idx1-1}: {TOP_WORDS[i]}-----------------------")
            idx2 = 0
            [word1_traj, filename1, start1, end1] = words1_found[k]
            word1_traj_np = np.array(word1_traj)

            for j in range(len(WORDS)):  # pocet iteraci = pocet vsech ruznych znaku
                word2 = WORDS[j]
                print(f"{j}: {WORDS[j]}, {round(timer(),2)} sec")

                if TOP_WORDS[i] == WORDS[j]: # testuji dva stejne znaky, nemusim nacitat znovu trajektorie
                    words2_found = words1_found
                else:
                    pk_word = open(os.path.join(path_trajectory,str(word2))+".pickle", "rb")
                    words2_found = pk.load(pk_word)
                    pk_word.close()
                    #words2_found = find_word(word2, -1, 'Sign_Language_BP/data_bvh')

                for l in range(len(words2_found)):  # pocet iteraci = cetnost druheho znaku, ktery zrovna testuji
                    [word2_traj, filename2, start2, end2] = words2_found[l]
                    word2_traj_np = np.array(word2_traj)
                    # dalsi sloupec v radku [idx1] (vysledna matice)
                    idx2 += 1

                    #times = timer()
                    dtw_result = dtw_dist(word1_traj_np, word2_traj_np, path_jointlist)
                    #timee = timer()
                    #print('4: '+str(timee-times))

                    # naplneni matice DTW vysledku
                    DTW_MATRIX1[idx1-1, idx2-1] = dtw_result['RightHand']
                    DTW_MATRIX2[idx1-1, idx2-1] = dtw_result['LeftHand']
                    DTW_MATRIX3[idx1-1, idx2-1] = dtw_result['RightArm']
                    DTW_MATRIX4[idx1-1, idx2-1] = dtw_result['LeftArm']
                    DTW_MATRIX5[idx1-1, idx2-1] = dtw_result['RightForeArm']
                    DTW_MATRIX6[idx1-1, idx2-1] = dtw_result['LeftForeArm']

                    if idx1-1 == 0:  # naplneni matice pozic jednotlivych znaku - pouze jednou
                        POSITION[idx2-1] = [filename2, start2, end2]

    DTW_DICT_MATRIX['RightHand'] = DTW_MATRIX1
    DTW_DICT_MATRIX['LeftHand'] = DTW_MATRIX2
    DTW_DICT_MATRIX['RightArm'] = DTW_MATRIX3
    DTW_DICT_MATRIX['LeftArm'] = DTW_MATRIX4
    DTW_DICT_MATRIX['RightForeArm'] = DTW_MATRIX5
    DTW_DICT_MATRIX['LeftForeArm'] = DTW_MATRIX6

    return [DTW_DICT_MATRIX, words_all, POSITION]


def import_abs_data(filepath, path_converted): 
    """Imports trajectory data recalculated in absolute coordinates saved to .pickle

    Args:
        filepath (string): A name of the file to be imported
        path_converted (string): A path to the folder with converted trajectory and dictionary to the .pickle

    Returns:
        [dictionary, list]: Dictionary of anotated words corresponding to returned list (trajectory)
    """
    import numpy as np
    import pickle as pkl
    import os

    filename = os.path.splitext(filepath)[0]

    dict_file = os.path.join(path_converted, 'dictionary_'+filename+'.pickle')
    pkl_dict = open(dict_file, "rb")
    dictionary = pkl.load(pkl_dict)
    pkl_dict.close()

    traj_file = os.path.join(path_converted, 'trajectory_'+filename+'.pickle')
    pkl_traj = open(traj_file, "rb")
    trajectory = pkl.load(pkl_traj)
    pkl_traj.close()

    return dictionary, trajectory