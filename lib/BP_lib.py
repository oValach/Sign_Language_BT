# vystřihne trajektory mezi předanými snímky, zatím neošetřeny výjimky
def get_trajectory(trajectory, start, end):
    word_trajectory = trajectory[start:end, :, :]
    return word_trajectory


# najde prvních [amount] výskytů předaného znaku word napříč všemi soubory
def find_word(word, amount):
    import os

    path = 'C:/Users/User/BP/Projekt/data_bvh'
    file_list = os.listdir(path)

    # vyhnuti se slozkam a ostatnim souborum
    file_list = [f for f in file_list if ('.bvh' in f)]

    if amount != 1: # pozaduji jeden vyskyt -> nechci vysledek appendovat do pole
        word_trajectory = []
    current_amount = 0
    test_file = 0
    for filepath in file_list:  # iterování přes jednotlivé soubory
        test_file += 1
        # nahrani prepoctenych dat z angularnich - dictionary, trajectory
        [dictionary, trajectory] = import_abs_data(filepath)

        number = 0
        for _, val in enumerate(dictionary):

            if current_amount == amount: # nalezen pozadovany pocet
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
                    word_trajectory.append([get_trajectory(trajectory, start, end),filepath,start,end])

                elif tmp_key == 'sign_id' and val[tmp_key] == word and (amount == 1): # chci vratit pouze jeden vyskyt znaku
                    current_amount += 1
                    start = val['annotation_Filip_bvh_frame'][0]
                    end = val['annotation_Filip_bvh_frame'][1]
                    word_trajectory = [get_trajectory(trajectory, start, end),filepath,start,end]

        if amount == -1 and test_file < 33:
            continue
        elif amount == -1 and test_file == 33:
            print('Cetnost slova "' + str(word)+'" je: ' + str(current_amount))
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


# spocte cetnosti vsech znaku vyskytujicich se v nahravkach
def count_words(lower_limit, graph):
    import os
    from collections import OrderedDict
    import matplotlib.pyplot as plt
    import operator
    import numpy as np

    lower_limit -= 1
    path = 'C:/Users/User/BP/Projekt/data_bvh'
    file_list = os.listdir(path)
    # vyhnuti se slozkam a ostatnim souborum
    file_list = [f for f in file_list if ('bvh' in f)]

    test_file = 0
    word_dict = {}  # slovnik znaku
    for filepath in file_list:  # iterování přes jednotlivé soubory
        test_file += 1
        #print('Prohledavani souboru cislo ' + str(test_file))

        # nahrani prepoctenych dat z angularnich - dictionary, trajectory
        [dictionary, _] = import_abs_data(filepath)

        for _, val in enumerate(dictionary):
            for tmp_key in val.keys():
                if val[tmp_key] == '':  # znak bez vyznamu
                    break
                if tmp_key == 'sign_id' and (val[tmp_key] not in word_dict.keys()):
                    word_dict[val[tmp_key]] = 1  # novy znak
                elif tmp_key == 'sign_id' and (val[tmp_key] in word_dict.keys()):
                    word_dict[val[tmp_key]] += 1  # opakujici se znak

    # serazeni slovniku cetnosti sestupne
    word_counts_sorted = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=False)
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


def dtws(type, word1, word2):
    from dtw import dtw
    from fastdtw import fastdtw
    import numpy as np
    from scipy.spatial.distance import euclidean

    file_joints = open('C:/Users/User/BP/Projekt/data/joint_list.txt', 'r')
    joints = file_joints.readlines()
    joints = [f.rstrip() for f in joints]
    if type == 'fastdtw':  # zrychlená metoda dtw
        dtw_dist = {}
        # přepočítání 3 dimenzionálních dat na 1 dimenzi porovnáváním po částech těla postupně přes x,z,y
        for i in range(len(joints)):  # joint
            seq1x = []
            seq1y = []
            seq1z = []
            seq2x = []
            seq2y = []
            seq2z = []
            for j in range(len(word1)):  # pocet snimku
                # skip nechtenych joints
                if any(char.isdigit() for char in joints[i]) or ('Head' in joints[i]) or ('Spine' in joints[i]) or ('Hips' in joints[i]):
                    break
                seq1x.append(word1[j][i][0])
                seq1y.append(word1[j][i][1])  # append souradnic prvniho slova
                seq1z.append(word1[j][i][2])

            for j in range(len(word2)):  # pocet snimku
                # skip nechtenych joints
                if any(char.isdigit() for char in joints[i]) or ('Head' in joints[i]) or ('Spine' in joints[i]) or ('Hips' in joints[i]):
                    break
                seq2x.append(word2[j][i][0])
                seq2y.append(word2[j][i][1])  # append souradnic druheho slova
                seq2z.append(word2[j][i][2])
            # skip nechtenych joints
            if any(char.isdigit() for char in joints[i]) or ('Head' in joints[i]) or ('Spine' in joints[i]) or ('Hips' in joints[i]):
                continue
            # 1. prvek výstupu = distance, 2. = path
            dtwx = fastdtw(seq1x, seq2x, dist=euclidean)[0]
            dtwy = fastdtw(seq1y, seq2y, dist=euclidean)[0]
            dtwz = fastdtw(seq1z, seq2z, dist=euclidean)[0]
            dtw_dist[joints[i]] = dtwx+dtwy+dtwz
    elif type == 'dtw':  # metoda dtw
        dtw_dist = {}
        # přepočítání 3 dimenzionálních dat na 1 dimenzi porovnáváním po částech těla
        for i in range(len(joints)):  # joint
            seq1x = []
            seq1y = []
            seq1z = []
            seq2x = []
            seq2y = []
            seq2z = []
            for j in range(len(word1)):  # pocet snimku
                # skip nechtenych joints
                if any(char.isdigit() for char in joints[i]) or ('Head' in joints[i]) or ('Spine' in joints[i]) or ('Hips' in joints[i]):
                    break
                seq1x.append(word1[j][i][0])
                seq1y.append(word1[j][i][1])
                seq1z.append(word1[j][i][2])

            for j in range(len(word2)):  # pocet snimku
                # skip nechtenych joints
                if any(char.isdigit() for char in joints[i]) or ('Head' in joints[i]) or ('Spine' in joints[i]) or ('Hips' in joints[i]):
                    break
                seq2x.append(word2[j][i][0])
                seq2y.append(word2[j][i][1])
                seq2z.append(word2[j][i][2])
            # skip nechtenych joints
            if any(char.isdigit() for char in joints[i]) or ('Head' in joints[i]) or ('Spine' in joints[i]) or ('Hips' in joints[i]):
                continue
            dtwx = dtw(seq1x, seq2x, dist=euclidean)[0]
            dtwy = dtw(seq1y, seq2y, dist=euclidean)[0]
            dtwz = dtw(seq1z, seq2z, dist=euclidean)[0]
            dtw_dist[joints[i]] = dtwx+dtwy+dtwz
    return dtw_dist


def compare_all():
    import numpy as np

    DTW_DICT_MATRIX = {
        'RightHand': None,
        'LeftHand': None,
        'RightArm': None,
        'LeftArm': None,
        'RightForeArm': None,
        'LeftForeArm': None,
    }
    ID_DICT_MATRIX = {
        'RightHand': None,
        'LeftHand': None,
        'RightArm': None,
        'LeftArm': None,
        'RightForeArm': None,
        'LeftForeArm': None,
    }
    POSITION_DICT_MATRIX = {
        'RightHand': None,
        'LeftHand': None,
        'RightArm': None,
        'LeftArm': None,
        'RightForeArm': None,
        'LeftForeArm': None,
    }

    for v,k in enumerate(DTW_DICT_MATRIX):
        # ziskani vsech vyskytu slov
        words = []
        words_tmp = count_words(lower_limit = 1, graph = False)
        end_size = 0
        for _,v in enumerate(words_tmp):
            words.append(v)
            end_size = end_size + words_tmp[v]

        DTW_MATRIX = np.zeros(shape=(1000,1000))
        ID_MATRIX = np.zeros(shape=(1000,1000), dtype=object)
        POSITION_MATRIX = np.zeros(shape=(1000,1000), dtype=object)
        
        idx1 = 0
        for i in range(len(words)):

            word1 = words[i]
            words1_found = find_word(word1,-1)
            idx1 += 1 # dalsi radek ve vysledne matici
            idx2 = 0
            for j in range(len(words)):

                word2 = words[j]
                words2_found = find_word(word2,-1)

                for k in range(len(words1_found)):
                    [word1_traj,filename1,start1,end1] = words1_found[k]
                    for l in range(len(words2_found)):
                        [word2_traj,filename2,start2,end2] = words2_found[l]
                        idx2 += 1 # dalsi sloupec v radku [idx1]

                        dtw_result = dtws('dtw', word1_traj, word2_traj)
                        DTW_MATRIX[idx1-1,idx2-1] = dtw_result['RightHand'] # naplneni matice DTW vysledku

                        ID_MATRIX[idx1-1,idx2-1] = [word1,word2] # naplneni matice znaky poslanymi do dtw

                        POSITION_MATRIX[idx1-1,idx2-1] = [[filename1,start1,end1],[filename2,start2,end2]] # naplneni matice puvodem znaku

        DTW_DICT_MATRIX[k] = DTW_MATRIX
        ID_DICT_MATRIX[k] = ID_MATRIX
        POSITION_DICT_MATRIX[k] = POSITION_MATRIX

    return [DTW_DICT_MATRIX,ID_DICT_MATRIX,POSITION_DICT_MATRIX]


def import_abs_data(filepath):
    import numpy as np
    import pickle as pkl
    import os

    filename = filepath[0:12]
    path_converted = 'C:/Users/User/BP/Projekt/data_converted'

    dict_file = os.path.join(path_converted, 'dictionary_'+filename+'.pickle')
    pkl_dict = open(dict_file, "rb")
    dictionary = pkl.load(pkl_dict)

    traj_file = os.path.join(path_converted, 'trajectory_'+filename+'.pickle')
    pkl_traj = open(traj_file, "rb")
    trajectory = pkl.load(pkl_traj)

    return dictionary, trajectory