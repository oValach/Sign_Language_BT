import os
from lib import bvh2glo_simple
from lib.BP_lib import import_abs_data
import numpy as np
from lib import SL_dict
import matplotlib.pyplot as plt
from lib import data_comp
import operator
import math

if __name__ == "__main__":
    path = 'C:/Users/User/Work/Sign_Language_BP/Projekt/data_bvh'
    file_list = os.listdir(path)
    # vyhnuti se slozkam a ostatnim souborum
    file_list = [f for f in file_list if ('bvh' in f)]

    distancesJointsAll = [[None]]*61
    fileRangesAll = []
    done_files = 0
    avg_dist_files = []
    for filepath in file_list:  # iterování přes jednotlivé soubory

        # nahrani prepoctenych dat z angularnich - dictionary, trajectory
        [dictionary, trajectory] = import_abs_data(filepath)
        file_joints = open(
            'C:/Users/User/Work/Sign_Language_BP/Projekt/data/joint_list.txt', 'r')
        joints = file_joints.readlines()
        joints = [f.rstrip() for f in joints]

        ranges = []  # pole délek jednotlivých transition [snímků/tra.]
        DISTS = []  # pole vzdáleností, které ucestovaly jednotlivé části těla během transitions
        number = 0

        frameCounter = 0
        maxprevSpeed = [0]*len(joints)
        sumprevSpeed = [0]*len(joints)
        maxtraSpeed = [0]*len(joints)
        sumtraSpeed = [0]*len(joints)
        diffsmax = [0]*len(joints)
        diffsavg = [0]*len(joints)

        for i, val in enumerate(dictionary):
            number += 1
            for tmp_key in val.keys():
                if tmp_key == 'sign_name' and val[tmp_key] == 'tra.':

                    start = val['annotation_Filip_bvh_frame'][0]
                    end = val['annotation_Filip_bvh_frame'][1]

                    # implementovaná fn na výpočet abs. vzdálenosti
                    DIST = data_comp.comp_dist(trajectory, start, end)
                    DISTS.append(DIST)
                    ranges.append(end-start)

                    nextLine = dictionary[i+1]
                    speedVals = data_comp.comp_speed(
                        trajectory, dictionary, previousLine, val)  # fce - získání rychlostí
                    if speedVals[2] == 1:
                        # kontrola maxim rychlostí pro jednotlive klouby
                        for jointIdx in range(len(joints)):
                            if speedVals[0][jointIdx] > maxprevSpeed[jointIdx]:
                                maxprevSpeed[jointIdx] = speedVals[0][jointIdx]
                            if speedVals[1][jointIdx] > maxtraSpeed[jointIdx]:
                                maxtraSpeed[jointIdx] = speedVals[1][jointIdx]

                            sumprevSpeed[jointIdx] = sumprevSpeed[jointIdx] + \
                                speedVals[0][jointIdx]
                            sumtraSpeed[jointIdx] = sumtraSpeed[jointIdx] + \
                                speedVals[1][jointIdx]
                        frameCounter += 1
                        break
            previousLine = val

        Dist_sorted = []  # přeuspořádání DISTS, index řádku = část těla, obsahuje všechny polohy dané části těla ve snímcích
        numberof = 0
        while numberof < len(DISTS[1]):
            Dist_sorted.append([])
            for i in range(len(DISTS)):
                Dist_sorted[numberof].append(DISTS[i][numberof])
            numberof += 1

        # průměrná uražená vzdálenost pro každou část těla (mometálně nepoužito)
        avg_dist_singly = []
        for i in range(len(Dist_sorted)):
            avg_dist_singly.append(np.mean(Dist_sorted[i]))
        # celková průměrná uražená vzdálenost (-||-)
        avg_dist_files.append(avg_dist_singly)

        done_files += 1
        print(done_files)

        for k in range(len(Dist_sorted)):  # akumulace hodnot do společného pole
            distancesJointsAll[k] = distancesJointsAll[k] + Dist_sorted[k]
            if distancesJointsAll[k][0] == None:
                distancesJointsAll[k].pop(0)
        fileRangesAll = fileRangesAll+ranges

        if done_files == 33:
            # pole průměrných vzdáleností pro každou část těla přes všechny soubory
            avg_dist_files_singly = []
            for i in range(len(joints)):
                values_temp = []
                for j in range(done_files):
                    values_temp.append(avg_dist_files[j][i])
                avg_dist_files_singly.append(np.mean(values_temp))

            output_avgDist_file = [0]*61
            for j in range(len(avg_dist_files_singly)):
                maxIdx = avg_dist_files_singly.index(
                    max(avg_dist_files_singly))
                maxAvgDist = avg_dist_files_singly[maxIdx]
                output_avgDist_file[j] = [joints[maxIdx], maxAvgDist]
                avg_dist_files_singly[maxIdx] = 0

            avgprevSpeed = [summary1/frameCounter for summary1 in sumprevSpeed]
            avgtraSpeed = [summary2/frameCounter for summary2 in sumtraSpeed]

            corr_coefs = []  # pole korelací částí těla s délkou transitiony přes všechny soubory
            for j in range(len(distancesJointsAll)):
                corr_matrix = np.corrcoef(distancesJointsAll[j], fileRangesAll)
                corr_coefs.append(corr_matrix[0][1]) 
            m = 0
            maxes = {}
            corr_coefsCopy = corr_coefs.copy()
            while m < 3:  # 3 maxima
                idx = corr_coefsCopy.index(max(corr_coefsCopy))
                maxes[joints[idx]] = corr_coefsCopy[idx]
                corr_coefsCopy[idx] = 0
                m += 1

            # pole rozdílů rychlostí během předchozího znaku a během transitiony
            for idx in range(len(joints)):
                diffsmax[idx] = maxtraSpeed[idx] - maxprevSpeed[idx]
                diffsavg[idx] = avgtraSpeed[idx] - avgprevSpeed[idx]

            avgDifMax = np.mean(diffsmax)
            avgDifAvg = np.mean(diffsavg)

            # uložení výsledků
            file = open("Projekt/soubory/results_duration.txt", "w")
            file.write("BIGGEST correlations:" + str(maxes)+'\n')
            file.write("ALL correlations:\n")
            for p in range(len(joints)):
                file.write(str(joints[p]) + " - " + str(corr_coefs[p])+'\n')
            file.close()

            file = open("Projekt/soubory/results_speed.txt", "w")
            file.write('Mean difference of maximal values: '+str(avgDifMax) +
                       ' in {'+str(round(min(diffsmax), 3))+';'+str(round(max(diffsmax), 3))+'}\n')
            file.write('Mean difference of average values: '+str(avgDifAvg) +
                       ' in {'+str(round(min(diffsavg), 3))+';'+str(round(max(diffsavg), 3))+'}\n')
            file.write(
                'Positive values -> bigger average speed during transitions\n')
            file.write(
                'Negative values -> smaller average speed during transitions\n\n')
            file.write(
                'Joint,   Maximal speed [pre,tra],   Average speed [pre,tra]\n\n')
            for idx in range(len(joints)):
                file.write(str(joints[idx])+'    '+str(round(maxprevSpeed[idx], 3))+','+str(round(
                    maxtraSpeed[idx], 3))+'        '+str(round(avgprevSpeed[idx], 3))+','+str(round(avgtraSpeed[idx], 3))+'\n')
            file.close()

            file = open("Projekt/soubory/avg_distances.txt", "w")
            file.write("\n")
            for k in range(len(joints)):
                file.write(joints[k]+"   ")
                for j in range(done_files):
                    file.write("  "+str(round(avg_dist_files[j][k], 3))+"  ")
                file.write("\n")
            file.write("\n")
            for l in range(len(joints)):
                file.write(str(
                    output_avgDist_file[l][0]) + " : " + str(round(output_avgDist_file[l][1], 3))+"\n")

            ''' Výstup v konzoli
                print("BIGGEST correlations:" +str(maxes)+'\n')
                print("ALL correlations:\n")
                for p in range(len(joints)):
                    print(str(joints[p]) + " - " + str(corr_coefs[p])+'\n')

                print(
                    'Joint           , Maximals [pre,tra],  Average [pre,tra]\n\n')
                for idx in range(len(joints)):
                    print(str(joints[idx])+'    '+str(round(maxprevSpeed[idx],3))+','+str(round(maxtraSpeed[idx],3)
                          )+'        '+str(round(avgprevSpeed[idx],3))+','+str(round(avgtraSpeed[idx],3))+'\n')
                '''
