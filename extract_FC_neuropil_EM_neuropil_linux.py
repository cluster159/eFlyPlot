from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import navis
import flybrains
import numpy as np
import os
import pickle
import open3d as o3d
import copy
from probreg import cpd
import random as rd
import pandas as pd
from probreg import bcpd

use_cuda = False

rd.seed(1000)

if use_cuda:
    import cupy as cp

    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
else:
    cp = np
    to_cpu = lambda x: x

source = 'JRCFIB2018Fraw'
target = 'FCWB'
tf_type = 'affine'

FlyEM_neuron_path = 'Data/FlyEM_skeleton/'
FlyEM2FCWB_path = 'Data/FlyEM2FCWB/'
FlyEM2FC_path = 'Data/FlyEM2FC/'

complete_EM_path = 'Data/FlyEM_skeleton/'
FC_neuron_path = 'Data/FlyCircuit_skeleton/'
# FC_neuron_path = 'C:/Users/clust/Desktop/x-brain-master/swc_spin/'
FC_neuropil_path = "Data/neuropil_avizo/"
FlyEM_neuropil_path = "Data/FlyEM_neuropil/"
FlyEM_neuropil_to_FCWB_path = 'Data/EM2FCWB_neuropil/'
FlyEM_neuropil_to_FC_path = 'Data/EM2FC_neuropil/'

if not os.path.isdir(FlyEM_neuropil_to_FC_path):
    os.mkdir(FlyEM_neuropil_to_FC_path)

if not os.path.isdir(FlyEM_neuropil_to_FCWB_path):
    os.mkdir(FlyEM_neuropil_to_FCWB_path)

if not os.path.isdir(FlyEM2FCWB_path):
    os.mkdir(FlyEM2FCWB_path)

if not os.path.isdir(FlyEM2FC_path):
    os.mkdir(FlyEM2FC_path)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def read_avizo_points(file, p=1):
    with open(file, 'rt')as ff:
        start = 0
        xyz = []
        for line in ff:
            '''
            format: 
            @1
            x y z
            ...

            @2
            '''
            if line[:2].find("@2") != -1:
                break
            if start == 1:
                if rd.random() > p:
                    continue
                groups = line[:-1].split(" ")
                xyz.append([float(i) for i in groups])
            if line[:2].find("@1") != -1:
                start = 1
    return np.array(xyz)


def read_obj_points(file, p=1):
    with open(file, 'rt')as ff:
        xyz = []
        for line in ff:
            if line[0].find("v") != -1:
                '''
                format: 
                v x y z
                '''
                if rd.random() > p:
                    continue
                groups = line[:-1].split(" ")[1:]
                xyz.append([float(i) for i in groups])
    return np.array(xyz)


def transform_from_source_to_target(points, source, target):
    print("PPP",source,target)
    print(points)
    points = np.array(points,dtype=float)
    # new_points = navis.xform_brain(points, source=source, target=target)
    try:
        new_points = navis.xform_brain(points, source=source, target=target)
    except:
        new_points = []
        print("ERROR")
        pass
    # print("NPP")
    print(new_points)
    return new_points


def transform_from_FlyEM_to_FCWB(points):
    try:
        new_points = navis.xform_brain(points, source=source, target=target)
    except:
        new_points = []
        print("ERROR")
        pass
    return new_points


def write_xyz(file, points):
    with open(file, 'wt')as ff:
        for point in points:
            ff.writelines(f"{point[0]} {point[1]} {point[2]}\n")
    return

def FAFB_to_FlyEM(object, source, target, tftype='affine'):
    object = cp.asarray(object, dtype=cp.float32)
    print('object',object[0])
    source = cp.asarray(source, dtype=cp.float32)
    print('source', source[0])
    target = cp.asarray(target, dtype=cp.float32)
    print('target', target[0])
    tf_param, _, _ = cpd.registration_cpd(source, target, tf_type_name=tftype, use_cuda=use_cuda)
    # tf_param = bcpd.registration_bcpd(source, target)

    result = tf_param.transform(object)
    return to_cpu(result), tf_param

def FC_to_FCWB(source, target, tftype='affine'):
    source = [[(xyz[0] * + 475)/1.7, (xyz[1] -185) / -2, (xyz[2]-125) / -2.5] for xyz in source]
    source = cp.asarray(source, dtype=cp.float32)
    target = cp.asarray(target, dtype=cp.float32)
    tf_param, _, _ = cpd.registration_cpd(source, target, tf_type_name=tftype, use_cuda=use_cuda)
    return tf_param

def FC_to_FCWB_transform(object, tf_param):
    object = [[(xyz[0] * + 475) / 1.7, (xyz[1] - 185) / -2, (xyz[2] - 125) / -2.5] for xyz in list(object)]
    return to_cpu(tf_param.transform(cp.asarray(object, dtype=cp.float32)))


def FCWB_to_FC(object, source, target, tftype='affine'):
    object = [[xyz[0]*1.7-475, xyz[1]*-2 + 185, xyz[2]*-2.5 + 125] for xyz in object]
    # object = [[xyz[0] * 1, xyz[1] * -1, xyz[2] * -1] for xyz in object]

    object = cp.asarray(object, dtype=cp.float32)
    source = [[xyz[0]*1.7-475, xyz[1]*-2 + 185, xyz[2]*-2.5 + 125] for xyz in source]
    # source = [[xyz[0] * 1, xyz[1] * -1, xyz[2] * -1] for xyz in source]

    source = cp.asarray(source, dtype=cp.float32)
    target = cp.asarray(target, dtype=cp.float32)
    # print(source)
    # print(target)
    tf_param, _, _ = cpd.registration_cpd(source, target, tf_type_name=tftype, use_cuda=use_cuda)
    result = tf_param.transform(object)
    return to_cpu(result), tf_param


def FCWB_to_FC_transform(object, tf_param):
    return to_cpu(tf_param.transform(cp.asarray(object, dtype=cp.float32)))


def write_tf(transform_maxtrix, file):
    with open(file, 'wb')as ff:
        pickle.dump(transform_maxtrix, ff)
    return


def plot_matched_neuron(source, target, other_coordinate_list=[], file_name="check_result.png",show=False):
    f = plt.figure()
    ax = plt.gca(projection='3d')
    plt.plot(source[:, 0], source[:, 1], source[:, 2], '.', label='source')
    plt.plot(target[:, 0], target[:, 1], target[:, 2], '.', label='target')
    for xyz in other_coordinate_list:
        xyz = np.array(xyz)
        plt.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], '.')
    set_axes_equal(ax)
    plt.legend()
    plt.savefig(file_name)
    if show == True:
        plt.show()
    # plt.show()
    plt.close()


def plot_before_after(source, target, result, other_coordinate_list=[], file_name="check_result.png",show=False):
    f = plt.figure()
    ax = plt.gca(projection='3d')
    plt.plot(source[:, 0], source[:, 1], source[:, 2], '.', label='source')
    plt.plot(target[:, 0], target[:, 1], target[:, 2], '.', label='target')
    plt.plot(result[:, 0], result[:, 1], result[:, 2], '.', label='result')
    for xyz in other_coordinate_list:
        xyz = np.array(xyz)
        plt.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], '.')
    # ax.set_xlim3d(-500, 500)
    # ax.set_ylim3d(300, -700)
    # ax.set_zlim3d(-500, 500)
    set_axes_equal(ax)
    plt.legend()
    plt.savefig(file_name)
    if show == True:
        plt.show()
    plt.close()


###### Combination compare example ###########################################
def get_combined_neuropil():
    FC_AL = 'al_3_instd_r.txt'
    EM_AL = [i for i in os.listdir(FlyEM_neuropil_path) if i.find("AL-") != -1 and i.find("(R)") != -1]
    EM_AL_xyz = []
    for G in EM_AL:
        EM_AL_xyz = EM_AL_xyz + list(read_obj_points(FlyEM_neuropil_path + G, p=0.01))
    EM_AL_xyz = np.array(EM_AL_xyz)
    # print(EM_AL_xyz)
    EM_AL_xyz_FCWB = transform_from_FlyEM_to_FCWB(EM_AL_xyz)
    write_xyz(FlyEM_neuropil_to_FCWB_path + "EM2FCWB_" + "AL(R)", EM_AL_xyz_FCWB)
    FC_AL_xyz = read_avizo_points(FC_neuropil_path + FC_AL, p=0.1)
    # print(FC_AL_xyz)
    EM_AL_xyz_FC, tf_param = FCWB_to_FC(list(EM_AL_xyz_FCWB), list(EM_AL_xyz_FCWB), list(FC_AL_xyz), tftype=tf_type)
    write_xyz(FlyEM_neuropil_to_FC_path + "EM2FC_" + tf_type + "_AL(R)", EM_AL_xyz_FC)
    # write_tf(tf_param, FlyEM_neuropil_to_FC_path + "EM2FC_tf_" + "AL(R)_"+ tf_type + ".pickle")
    plot_before_after(EM_AL_xyz_FCWB, FC_AL_xyz, EM_AL_xyz_FC, file_name="AL.png")
    return tf_param


def FCWB_to_FC_first_step(points):
    print("FCWB_to_FC")
    print(points)
    return [[xyz[0] * 1.7 - 475, xyz[1] * -2 + 185, xyz[2] * -2.5 + 125] for xyz in points]
    # return [[xyz[0] * 1, xyz[1] * -1, xyz[2] * -1] for xyz in points]


###### Pair matched example#######################################################
def get_pair_matched_neuropil():
    FC_MB = 'mb_4_instd_r.txt'
    FC_cal = 'cal_18_instd_r.txt'
    EM_MB = 'MB(R).obj'
    EM_MB_xyz_FCWB = transform_from_FlyEM_to_FCWB(read_obj_points(FlyEM_neuropil_path + EM_MB, p=0.01))
    write_xyz(FlyEM_neuropil_to_FCWB_path + "EM2FCWB_" + EM_MB, EM_MB_xyz_FCWB)
    FC_MB_xyz = np.concatenate(
        (read_avizo_points(FC_neuropil_path + FC_MB, p=0.5), read_avizo_points(FC_neuropil_path + FC_cal, p=0.5)))
    EM_MB_xyz_FC, tf_param = FCWB_to_FC(list(EM_MB_xyz_FCWB), list(EM_MB_xyz_FCWB), list(FC_MB_xyz), tftype=tf_type)
    write_xyz(FlyEM_neuropil_to_FC_path + "EM2FC_" + tf_type + "_" + EM_MB, EM_MB_xyz_FC)
    # write_tf(tf_param, FlyEM_neuropil_to_FC_path + "EM2FC_tf_" + EM_MB[:-4]+"_"+ tf_type + ".pickle")
    plot_before_after(EM_MB_xyz_FCWB, FC_MB_xyz, EM_MB_xyz_FC, file_name="MB.png")
    return tf_param


def get_neuropil_neuron_transform(neuropil_list=['AL_R', 'MB_R', 'PCB_R','LH_R']):
    FlyEM_neuropil_list = []
    FlyCircuit_neuropil_list = []
    data = pd.read_excel("FC_EM_neuropil_matching.xlsx")
    for neuropil in neuropil_list:
        mask = data['neuropil'] == neuropil
        if neuropil not in data['neuropil'].values.tolist():
            print(f"{neuropil} not exist!!")
            continue
        FlyEM_neuropil_list += data[mask]['FlyEM'].values.tolist()[0].split(" ")
        FlyCircuit_neuropil_list += data[mask]['FlyCircuit'].values.tolist()[0].split(" ")
    # print([read_obj_points(FlyEM_neuropil_path + EM_neuropil, p=0.01) for EM_neuropil in FlyEM_neuropil_list])
    FlyEM_xyz_collection = []
    # EM_file_list = ['300972942.swc', '1078693835.swc', '581678043.swc', '542634818.swc']
    # FC_file_list = ['G0239-F-000001.swc', 'Cha-F-000009.swc', 'Gad1-F-500365.swc', 'VGlut-F-400008.swc']
    # EM_file_list = ['300972942.swc']
    # FC_file_list = ['G0239-F-000001.swc']
    EM_file_list = []
    FC_file_list = []

    for EM_neuropil in FlyEM_neuropil_list:
        FlyEM_xyz_collection += list(read_obj_points(FlyEM_neuropil_path + EM_neuropil, p=0.01))
    for EM_neuron in EM_file_list:
        FlyEM_xyz_collection += list(read_swc_xyz(FlyEM_neuron_path + EM_neuron, p=0.1))
    FlyEM_xyz_FCWB = transform_from_FlyEM_to_FCWB(np.array(FlyEM_xyz_collection))
    print("EM data loaded!")
    FlyCircuit_xyz = np.concatenate(
        [read_avizo_points(FC_neuropil_path + FC_neuropil, p=0.5) for FC_neuropil in FlyCircuit_neuropil_list])
    if len(FC_file_list)>0:
        FC_neuron_xyz = np.concatenate([read_swc_xyz(FC_neuron_path + FC_neuron, p=1) for FC_neuron in FC_file_list])
        FlyCircuit_xyz = np.concatenate([FlyCircuit_xyz, FC_neuron_xyz])
    print("FC data loaded!")
    FlyEM_xyz_FC, tf_param = FCWB_to_FC(list(FlyEM_xyz_FCWB), list(FlyEM_xyz_FCWB), list(FlyCircuit_xyz),
                                        tftype=tf_type)
    plot_before_after(FlyEM_xyz_FCWB, FlyCircuit_xyz, FlyEM_xyz_FC, file_name='tmp.png')
    return tf_param


def get_neuropil_transform(neuropil_list=['AL_R', 'MB_R', 'PCB','LH_R','EB'],source='FCWB',target='FlyCircuit'):
    FlyEM_neuropil_list = []
    FlyCircuit_neuropil_list = []
    data = pd.read_excel("FC_EM_neuropil_matching.xlsx")
    for neuropil in neuropil_list:
        mask = data['neuropil'] == neuropil
        if neuropil not in data['neuropil'].values.tolist():
            print(f"{neuropil} not exist!!")
            continue
        FlyEM_neuropil_list += data[mask]['FlyEM'].values.tolist()[0].split(" ")
        FlyCircuit_neuropil_list += data[mask]['FlyCircuit'].values.tolist()[0].split(" ")
    # print([read_obj_points(FlyEM_neuropil_path + EM_neuropil, p=0.01) for EM_neuropil in FlyEM_neuropil_list])
    FlyEM_xyz_collection = []
    for EM_neuropil in FlyEM_neuropil_list:
        FlyEM_xyz_collection += list(read_obj_points(FlyEM_neuropil_path + EM_neuropil, p=0.01))
    FlyEM_xyz_FCWB = transform_from_FlyEM_to_FCWB(np.array(FlyEM_xyz_collection))
    print("EM data loaded!")
    FlyCircuit_xyz = np.concatenate(
        [read_avizo_points(FC_neuropil_path + FC_neuropil, p=0.5) for FC_neuropil in FlyCircuit_neuropil_list])
    print("FC data loaded!")
    if source=='FCWB' and target=='FlyCircuit':
        FlyEM_xyz_FC, tf_param = FCWB_to_FC(list(FlyEM_xyz_FCWB), list(FlyEM_xyz_FCWB), list(FlyCircuit_xyz),
                                            tftype=tf_type)
        plot_before_after(FlyEM_xyz_FCWB, FlyCircuit_xyz, FlyEM_xyz_FC, file_name='tmp.png')
    elif source=='FlyCircuit' and target=='FCWB':
        print("11111")
        tf_param = FC_to_FCWB(source=list(FlyCircuit_xyz),target=list(FlyEM_xyz_FCWB))
        print("22222")
        FC_xyz_FCWB = FC_to_FCWB_transform(object=list(FlyCircuit_xyz),tf_param=tf_param)
        print("33333")
        plot_before_after(FC_xyz_FCWB,FlyEM_xyz_FCWB,FC_xyz_FCWB, file_name='FC_to_FCWB.png')
        print("4444")

    return tf_param

def get_neuropil_transform_FAFB_to_FlyEM(neuropil_list=['MB_R']):
    FlyEM_neuropil_list = []
    FAFB_neuropil_list = []
    FAFB_neuropil_path = 'Data/FAFB_neuropil/'
    data = pd.read_excel("FC_EM_neuropil_matching.xlsx")
    for neuropil in neuropil_list:
        mask = data['neuropil'] == neuropil
        if neuropil not in data['neuropil'].values.tolist():
            print(f"{neuropil} not exist!!")
            continue
        FlyEM_neuropil_list += data[mask]['FlyEM'].values.tolist()[0].split(" ")
        FAFB_neuropil_list += data[mask]['FAFB'].values.tolist()[0].split(" ")
    # print([read_obj_points(FlyEM_neuropil_path + EM_neuropil, p=0.01) for EM_neuropil in FlyEM_neuropil_list])
    FlyEM_xyz_collection = []
    for EM_neuropil in FlyEM_neuropil_list:
        FlyEM_xyz_collection += list(read_obj_points(FlyEM_neuropil_path + EM_neuropil, p=0.05))
    print("EM data loaded!")
    f = plt.figure()
    ax = plt.gca(projection='3d')
    plot_EM = np.array(FlyEM_xyz_collection)
    plt.plot(plot_EM[:, 0], plot_EM[:, 1], plot_EM[:, 2], '.')
    # plt.axis('equal')
    plt.show()

    FAFB_xyz = np.concatenate(
        [read_obj_points(FAFB_neuropil_path + FAFB_neuropil, p=1) for FAFB_neuropil in FAFB_neuropil_list])
    print("FAFB data loaded!")
    with open("neuropil.txt",'wt')as ff:
        for x,y,z in FAFB_xyz:
            ff.writelines(f"{x} {y} {z}\n")
    FAFB_xyz = np.array(transform_from_source_to_target(np.array(FAFB_xyz),source="FAFB",target='JRCFIB2018Fraw'))
    f = plt.figure()
    ax = plt.gca(projection='3d')
    plt.plot(FAFB_xyz[:,0],FAFB_xyz[:,1],FAFB_xyz[:,2],'.')
    # plt.axis('equal')
    plt.show()
    FAFB_xyz_FlyEM, tf_param = FAFB_to_FlyEM(source=list(FAFB_xyz), target=list(FlyEM_xyz_collection), object=list(FAFB_xyz),
                                        tftype=tf_type)
    print('result',FAFB_xyz_FlyEM[0])
    print(len(FAFB_xyz_FlyEM))

    plot_before_after(target=np.array(FlyEM_xyz_collection), result=np.array(FAFB_xyz_FlyEM), source=np.array(FAFB_xyz), file_name='tmp.png',show=True)
    return tf_param



def read_swc_xyz(file_name, p=1):
    xyz = []
    with open(file_name, 'rt')as ff:
        for line in ff:
            if line.find("#") != -1:
                continue
            if rd.random() > p:
                continue
            groups = line[:-1].split(" ")
            itx = 0
            for ptr in groups:
                if len(ptr) == 0:
                    continue
                if itx == 2:
                    x = float(ptr)
                elif itx == 3:
                    y = float(ptr)
                elif itx == 4:
                    z = float(ptr)
                    xyz.append([x, y, z])
                    break
                itx += 1
    return np.array(xyz)


def read_swc(file_name, p=1):
    all_parameters = []
    with open(file_name, 'rt')as ff:
        for line in ff:
            parameter = []
            if line.find("#") != -1:
                continue
            if rd.random() > p:
                continue
            groups = line[:-1].split(" ")
            itx = 0
            for ptr in groups:
                if len(ptr) == 0:
                    continue
                if ptr.find(".") != -1:
                    parameter.append(float(ptr))
                else:
                    parameter.append(int(ptr))
                itx += 1
            all_parameters.append(parameter)
    return all_parameters


def write_all_swc(all_parameters, xyz, file_name):
    with open(file_name, 'wt')as ff:
        for i in range(len(all_parameters)):
            ff.writelines(
                f"{all_parameters[i][0]} {all_parameters[i][1]} {xyz[i][0]} {xyz[i][1]} {xyz[i][2]} {all_parameters[i][5]} {all_parameters[i][6]}\n")
    return




if __name__ == '__main__':
    # tf_param = get_neuropil_transform_FAFB_to_FlyEM()

    # EM_file_list = ['300972942.swc', '1078693835.swc', '581678043.swc']
    # FC_file_list = ['G0239-F-000001.swc', 'Cha-F-000009.swc', 'Gad1-F-500365.swc']
    # neuropil_list = ['AL_R', 'MB_R', 'PCB_R', 'EB', 'FB']
    tf_param = get_neuropil_transform(source='FlyCircuit',target='FCWB')

    # for EM_neuron, FC_neuron in zip(EM_file_list, FC_file_list):
    #     EM_neuron_xyz = FCWB_to_FC_transform(
    #         FCWB_to_FC_first_step(transform_from_FlyEM_to_FCWB(read_swc_xyz(FlyEM_neuron_path + EM_neuron, p=1))),
    #         tf_param)
    #     FC_neuron_xyz = read_swc_xyz(FC_neuron_path + FC_neuron, p=0.9)
    #     plot_matched_neuron(EM_neuron_xyz, FC_neuron_xyz, file_name=f'{EM_neuron}.png',show=True)

    # EM_file_list = os.listdir(complete_EM_path)
    # for EM_neuron in EM_file_list:
    #     if os.path.isfile(FlyEM2FC_path + EM_neuron):
    #         continue
    #     try:
    #         all_parameters = read_swc(complete_EM_path + EM_neuron, p=1)
    #         EM_neuron_xyz = FCWB_to_FC_transform(
    #             FCWB_to_FC_first_step(transform_from_FlyEM_to_FCWB(read_swc_xyz(complete_EM_path + EM_neuron, p=1))),
    #             tf_param)
    #         write_all_swc(all_parameters, EM_neuron_xyz, FlyEM2FC_path + EM_neuron)
    #     except:
    #         print(EM_neuron)
    #         pass
    #
