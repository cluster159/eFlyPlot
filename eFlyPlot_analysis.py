import networkx as nx
import plotly.graph_objects as go
import pandas
from pandas import DataFrame
import navis
import flybrains
import numpy as np
import os
import nrrd
import random as rd
import nibabel as nib  # pip install nibabel
# from extract_FC_neuropil_EM_neuropil_linux import *
import math
from numba import njit
from pandas import DataFrame as Df
import pandas as pd
import base64
import extract_FC_neuropil_EM_neuropil_linux as warp_tool
import networkx.algorithms as al
from matplotlib import pyplot as plt
import re
import check_coordinates_in_neuropil as NN

'''
Function plan
1. compare connection number to same subtype in same neuropil
2. kde map of points
3. compare spatial distribution similarity in a given neuropil
4. analyze neuronal path length etc.
5. connection map (networkx etc.)
6. sankey chart for info flow (acyclic)
7. clustering
8. read simulation results for further visualization

'''
'''
 {
        'data': {'id': 'one', 'label': 'Modified Color'},
        'position': {'x': 75, 'y': 75},
        'classes': 'red' # Single class
    },
'''
import copy
from collections import defaultdict

class calculate_distribution:
    def __init__(self):
        self.x_slice = 8
        self.y_slice = 4
        self.z_slice = 2
        self.standard_neuropil = 'CA(R).obj'
        self.neuropil_path = 'C:/Users/clust/PycharmProjects/flyem/EM_neuropil/'
        self.neuropil_list = ['CA(R).obj']
        # self.coordinates = []
        self.xmax = 0
        self.xmin = 0
        self.ymax = 0
        self.ymin = 0
        self.zmax = 0
        self.zmin = 0
        self.default_distance = 0
        self.target_points = []
        self.x_unit = 1
        self.y_unit = 1
        self.z_unit = 1
        self.neuropil_coordinate_reference = []
        self.seqeunce_matrix = []

    def get_neuropil_transform(self, points):
        obj_data = open(self.neuropil_path + self.standard_neuropil, "rt")
        vertices, faces = obj_data_to_mesh3d(obj_data)
        x, y, z = vertices[:, :3].T
        pca = PCA(n_components=3)
        vertices = np.array([x, y, z]).transpose()
        pca.fit(vertices)
        pooled_x = []
        pooled_y = []
        pooled_z = []
        for neuropil in self.neuropil_list:
            obj_data = open(self.neuropil_path + neuropil, "rt")
            vertices, faces = obj_data_to_mesh3d(obj_data)
            x, y, z = vertices[:, :3].T
            pooled_x += list(x)
            pooled_y += list(y)
            pooled_z += list(z)
        vertices = np.array([x, y, z]).transpose()
        vertices_ca = pca.transform(vertices)
        d = self.default_distance
        self.neuropil_coordinate_reference = vertices_ca
        x, y, z = self.neuropil_coordinate_reference[:, :3].T
        self.xmax, self.xmin, self.ymax, self.ymin, self.zmax, self.zmin = max(x) + d, min(x) - d, max(y) + d, min(y) - d, max(z) + d, min(
            z) - d
        self.x_unit, self.y_unit, self.z_unit = (self.xmax - self.xmin) / self.x_slice, (self.ymax - self.ymin) / self.y_slice, (self.zmax - self.zmin) / self.z_slice
        self.target_points = pca.transform(points)

    def draw_distribution_3dbar(self, file_name, color=(rd.random(), rd.random(), rd.random(), 0.5)):
        xpos = np.array([i for i in range(self.x_slice)])
        ypos = np.array([j for j in range(self.y_slice)])
        xposM, yposM = np.meshgrid(xpos, ypos, copy=False)
        xp, yp, zp = copy.deepcopy(self.target_points)[:,:3].T
        synapse_distribution = count_synapse_distribution(np.array([xp, yp, zp]).transpose(),
                                                          [self.x_unit, self.y_unit, self.z_unit], [self.xmin, self.ymin, self.zmin],
                                                          (self.x_slice, self.y_slice, self.z_slice))
        for k in range(self.z_slice):
            zpos = []
            for j in range(self.y_slice):
                for i in range(self.x_slice):
                    zpos.append(synapse_distribution[i][j][k])
            zpos = np.array(zpos)
            zpos = zpos.ravel()
            dx = 0.5
            dy = 0.5
            dz = zpos
            fig = plt.figure()
            xposM = xposM.ravel()
            yposM = yposM.ravel()
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.bar3d(xposM, yposM, dz * 0, dx, dy, dz, color=[color for _ in range(self.x_slice * self.y_slice)])
            plt.plot((self.neuropil_coordinate_reference[:, 0] - self.xmin) / (self.xmax - self.xmin) * self.x_slice,
                     (self.neuropil_coordinate_reference[:, 1] - self.ymin) / (self.ymax - self.ymin) * self.y_slice,
                     [0 for _ in range(len(self.neuropil_coordinate_reference))], '.', color=(0.5, 0.5, 0.5, 0.1))

            plt.savefig(f"{file_name}_3dbar_layer{k}.png")
            plt.close()

    def draw_2d_contourplot(self, file_name, cmap='hot', scatter=True, scatter_color='white', title="",neglect_ratio=15):
        xp, yp, zp = copy.deepcopy(self.target_points)[:, :3].T
        if os.path.isfile("calyx_alpha_shape.picke"):
            with open("calyx_alpha_shape.picke", 'rb')as ff:
                alpha_shape = pickle.load(ff)
        else:
            alpha_shape = alphashape.alphashape(np.transpose(
                np.array([self.neuropil_coordinate_reference[:, 0], self.neuropil_coordinate_reference[:, 1]])))
            with open("calyx_alpha_shape.picke", 'wb')as ff:
                pickle.dump(alpha_shape, ff)
        coordinate_group = defaultdict(list)
        record = 0
        for x, y, z in zip(xp,yp,zp):
            record += 1
            if record < neglect_ratio:
                continue
            z_index = int((z - self.zmin) / self.z_unit)
            coordinate_group[z_index].append([x,y])
            record = 0
        for z_id in range(self.z_slice):
            coordinates = np.array(coordinate_group[z_id])
            if len(coordinates) < 3:
                continue
            coordinates = np.transpose(coordinates)
            plt.figure()
            g = sn.jointplot(coordinates[0],coordinates[1],shade=True,cbar=False,cmap=cmap,kind='kde',xlim=(self.xmin*1.2, self.xmax*1.2),ylim=(self.ymin*1.4, self.ymax*1.4))
            g.ax_joint.add_patch(PolygonPatch(alpha_shape, alpha=.5, fc='gray', ec='gray'))
            if scatter == True:
                g.plot_joint(plt.scatter, c=scatter_color, s=1)
            if title:
                plt.title(title)
            plt.savefig(f"{file_name}_layer_{z_id}.png")

    def draw_2d_contourplot_pooled(self, pooled_points_collection, file_name,group_label,palette,cmap=[], scatter=True, scatter_color=[], title="",kde_palette = {0:'Blues', 1:'Greens',2 :'Reds'},neglect_ratio=15):
        plt.figure()
        if os.path.isfile("calyx_alpha_shape.picke"):
            with open("calyx_alpha_shape.picke", 'rb')as ff:
                alpha_shape = pickle.load(ff)
        else:
            alpha_shape = alphashape.alphashape(np.transpose(
                np.array([self.neuropil_coordinate_reference[:, 0], self.neuropil_coordinate_reference[:, 1]])))
            with open("calyx_alpha_shape.picke", 'wb')as ff:
                pickle.dump(alpha_shape, ff)
        group_layer_data = defaultdict(list)
        for group_id, pooled_points in enumerate(pooled_points_collection):
            pooled_points = np.array(pooled_points)
            xp, yp, zp = pooled_points[:, :3].T
            # coordinate_group = defaultdict(list)
            for x, y, z in zip(xp, yp, zp):
                z_index = int((z - self.zmin) / self.z_unit)
                print(x,z_index)
                # coordinate_group[z_index].append([x, y])
                group_layer_data[z_index].append([x,y,group_label[group_id]])

        for z_id in range(self.z_slice):
            data = Df(data=group_layer_data[z_id], columns=['x', 'y', 'group'])
            scatter_data = Df(data=group_layer_data[z_id][::neglect_ratio], columns=['x', 'y', 'group'])
            fig = plt.figure(figsize=(12, 7))
            widths = [4, 1]
            heights = [1, 4]
            ### 1. gridspec preparation
            spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights,wspace=0.05, hspace=0.05)  # setting spaces
            ### 2. setting axes
            axs = {}
            for i in range(len(heights) * len(widths)):
                axs[i] = fig.add_subplot(spec[i // len(widths), i % len(widths)])
            ### 3. bill_length_mm vs bill_depth_mm
            # 3.1. kdeplot
            sn.kdeplot("x", "y", data=data, hue="group", alpha=0.9, ax=axs[2], zorder=1,palette=palette,levels=4)
            axs[2].add_patch(PolygonPatch(alpha_shape, alpha=.1, fc='gray', ec='gray'))
            # 3.2. scatterplot
            sn.scatterplot("x", "y", data=scatter_data, hue="group", ax=axs[2], zorder=2,palette=palette,s=5)
            # 3.3. histogram (bill_length_mm)
            sn.kdeplot(x="x", data=data, hue="group", ax=axs[0], legend=False, zorder=1,palette=palette)
            axs[0].set_xlim(axs[2].get_xlim())
            axs[0].set_xlabel('')
            axs[0].set_xticklabels([])
            axs[0].set_yticklabels([])
            axs[0].spines["left"].set_visible(False)
            axs[0].spines["top"].set_visible(False)
            axs[0].spines["right"].set_visible(False)
            axs[0].spines["right"].set_visible(False)
            axs[0].set_yticks([])
            # axs[0].get_yaxis().get_major_formatter().set_scientific(True)
            # axs[0].get_xaxis().get_major_formatter().set_scientific(True)
            # axs[0].ticklabel_format(useOffset=True, axis='y')
            # axs[0].ticklabel_format(useOffset=True, axis='x')

            # 3.3. histogram (bill_depth_mm)
            sn.kdeplot(y="y", data=data, hue="group", ax=axs[3], legend=False, zorder=1,palette=palette)
            axs[3].set_ylim(axs[2].get_ylim())
            axs[3].set_ylabel('')
            axs[3].set_yticklabels([])
            axs[3].set_xticklabels([])
            axs[3].spines["bottom"].set_visible(False)
            axs[3].spines["top"].set_visible(False)
            axs[3].spines["right"].set_visible(False)
            axs[3].set_xticks([])

            # axs[3].get_xaxis().get_major_formatter().set_scientific(True)
            # axs[3].get_yaxis().get_major_formatter().set_scientific(True)
            # axs[3].ticklabel_format(useOffset=True, axis='y')
            # axs[3].ticklabel_format(useOffset=True, axis='x')

            # 5.1. upper-right axes
            axs[1].axis("off")
            plt.savefig(f"{file_name}_layer_{z_id}.png",dpi=500)

    def draw_distribution_2d_heatmap(self,file_name,cmap='hot'):
        xp, yp, zp = copy.deepcopy(self.target_points)[:, :3].T
        synapse_distribution = count_synapse_distribution(np.array([xp, yp, zp]).transpose(),
                                                          [self.x_unit, self.y_unit, self.z_unit],
                                                          [self.xmin, self.ymin, self.zmin],
                                                          (self.x_slice, self.y_slice, self.z_slice))
        print(synapse_distribution)
        # synapse_distribution =
        max_count = np.max(synapse_distribution)
        rearrange_list = []
        sequence = []
        for k in range(self.z_slice):
            table = []
            for j in range(self.y_slice):
                row_list = []
                for i in range(self.x_slice):
                    row_list.append(synapse_distribution[i][j][k])
                    sequence.append(synapse_distribution[i][j][k])
                table.append(row_list)
            rearrange_list.append(table)
        Df(data=np.array([sequence]).transpose()).to_excel(f'{file_name}_spatial_sequence.xlsx')
        for k in range(self.z_slice):
            fig = plt.figure()
            ax=sn.heatmap(np.array(rearrange_list[k]),cmap=cmap,vmin=0,vmax=max_count)
            ax.invert_yaxis()
            plt.plot((self.neuropil_coordinate_reference[:, 0] - self.xmin) / (self.xmax - self.xmin) * self.x_slice,
                     (self.neuropil_coordinate_reference[:, 1] - self.ymin) / (self.ymax - self.ymin) * self.y_slice,
                      '.', color=(0.5, 0.5, 0.5, 0.4),markersize=5)

            plt.savefig(f"{file_name}_2d_heatmap_layer{k}.png")
            plt.close()
        return sequence



class connection_analysis_tool:
    @njit
    def construct_pearson_correlation_table(matrix):
        correlation_matrix = np.zeros((len(matrix), len(matrix)))
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                correlation = np.corrcoef(matrix[i], matrix[j])
                # print(correlation)
                correlation_matrix[i][j] = correlation[0][1]
                if j < i:
                    continue
        return correlation_matrix

    def export_cluster(row_linkage, label_list, file_name, threshold=0.8):
        den = scipy.cluster.hierarchy.dendrogram(row_linkage, labels=label_list, color_threshold=threshold)
        plt.savefig(f"{file_name}_dendrogram.png")
        plt.close()
        clusters = get_cluster_classes(den)
        cluster = []
        count = 0
        tmp_ids = []
        for j in clusters:
            for neuronId in clusters[j]:
                cluster.append([neuronId, count])
                tmp_ids.append(neuronId)
            count += 1
        for neuronId in label_list:
            if neuronId not in tmp_ids:
                cluster.append([neuronId, -1])
        Df(data=cluster, columns=['neuronId', 'cluster']).to_excel(f"{file_name}.xlsx")
        return

    def calculate_cosine_correlation(self,v1, v2):
        length_v1 = np.linalg.norm(v1)
        length_v2 = np.linalg.norm(v2)
        dominant = length_v1 * length_v2
        if dominant == 0:
            print("one vector is zero")
            print(v1, v2)
            return 0
        else:
            return np.dot(v1, v2) / dominant

    def calculate_Jaccard_index(self,v1, v2):
        print("Notice!!! all the elements in the vector should be larger than zero!!!!!!!!!!!!!!!")
        if np.dot(v1, v2) == 0:
            print("one vector is zero")
            print(v1, v2)
            return 0
        else:
            sum_max = 0
            sum_min = 0
            for x, y in zip(v1, v2):
                if x < 0 or y < 0:
                    print("ERROR")
                    return
                sum_max += max((x, y))
                sum_min += min((x, y))
            return sum_min / sum_max
    def construct_correlation_plot(self):
        correlation_matrix_KCab = construct_pearson_correlation_table(
            KC_MBON_Glomerulus[0:len(Subtype_to_KCid["KCab"])])
        yticklabels = [i for i in range(0, len(Subtype_to_KCid["KCab"]))]
        d = distance.pdist(correlation_matrix_KCab)
        row_linkage = hierarchy.linkage(d, method=method)
        g = sn.clustermap(correlation_matrix_KCab, row_linkage=row_linkage, col_linkage=row_linkage,
                          yticklabels=yticklabels)
        KCab_list = [network.KCid_list[int(i.get_text())] for i in g.ax_heatmap.yaxis.get_majorticklabels()]
        plt.title("KCab_correlation")
        plt.tight_layout()
        plt.savefig(f"KCab_correlation_via_MBON_G_{method}.png")
        plt.close()
        export_cluster(row_linkage, KCab_list, f"KCab_correlation_via_MBON_G_{method}")

    # def routing_from_source_to_target(self):
    #     sequence_collection_dict = {}
    #     for source_id in self.source_list:
    #         waiting_list = [source_id]
    #         while len(waiting_list)!=0:
    #             new_waiting_list = []
    #             for ptr_id in waiting_list:
    #                 if ptr_id in source_list:
    #                     sequence_collection_dict[ptr_id] = [ptr_id]
    #                 else:
    #                     for parent in self.upstream_dict[ptr_id]:
    #                         if parent
    #                 for down_id in self.downstream_dict[ptr_id]:
    #
    #                 if down_id in self.source_list:
    #                     continue
    #                 if down_id not in sequence_collection_dict:
    #                     sequence_collection_dict[down_id] = []
    #                     for sequence in sequence_collection_dict[self.upstream_dict[down_id]]:
    #                         sequence_collection_dict[down_id].append(sequence + [down_id])
    #                 if down_id in sequence_collection_dict:
    #                     sequence_collection_dict[down_id].append(sequence + [down_id])
    #
    #                 if down_id not in self.target_list:
    #                     waiting_list.append(down_id)
    #
    #     node_candidates_list = []
    #     for node in sequence_collection_dict:
    #         if sequence_colle

    def detect_cycles(self, source_list, target_list, connection_list, id_type_dict):
        upstream_dict = defaultdict(list)
        downstream_dict = defaultdict(list)
        for connection in connection_list:
            upid, downid, weight = connection
            upstream_dict[downid].append([upid,weight])
            downstream_dict[upid].append([downid,target])
        print("Constructed network!")
        self.source_list = source_list
        self.target_list = target_list
        self.id_type_dict = id_type_dict
        self.upstream_dict = upstream_dict
        self.downstream_dict = downstream_dict


        cycles = al.simple_cycles(g)
        # assuming that the exterior cycle will contain the most nodes



class brain_template_warping_tool:
    '''
    1. different file format transformation
    2. warping between different brain templates

    '''
    def __init__(self):
        self.tf_param_FCWB2FC = warp_tool.get_neuropil_transform()
        self.tf_param_FC2FCWB = warp_tool.get_neuropil_transform(neuropil_list=['AL_R', 'MB_R', 'PCB','LH_R','EB'],source='FlyCircuit',target='FCWB')
        self.label_to_template_dict={"FlyEM":'JRCFIB2018Fraw',
                                     "FlyEM_um":'JRCFIB2018Fum',
                                     "vfb":'JRC2018F',
                                     "FlyCircuit":"FCWB",
                                     "JRC2018U":"JRC2018U",
                                     "FAFB":"FAFB"
                                     }

    def nrrd2nii(self,file):
        # load nrrd
        _nrrd = nrrd.read(file)
        data = _nrrd[0]
        header = _nrrd[1]
        print(data.shape, header)
        # save nifti
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, f'{file[:-5]}.nii.gz')
        return

    def read_nrrd_data(self,file):
        # load nrrd
        _nrrd = nrrd.read(file)
        data = _nrrd[0]
        header = _nrrd[1]
        print(data.shape, header)
        return data, header

    def write_nii(self,file, data):
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, f'{file[:-5]}.nii.gz')
        return

    def read_swc_xyz(self,file_name, p=1):
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
        return xyz

    def from_nrrd_to_coordinate_reduced(self,data, header):
        coordinate = []
        x_scale = header['space directions'][0][0]
        y_scale = header['space directions'][1][1]
        z_scale = header['space directions'][2][2]
        for x in range(len(data)):
            for y in range(len(data[0])):
                for z in range(len(data[0][0])):
                    if data[x][y][z] > 0:
                        xyz = [float(int(x * x_scale)), float(int(y * y_scale)), float(int(z * z_scale))]
                        if xyz not in coordinate:
                            coordinate.append(xyz)
        return np.array(coordinate)

    def from_nrrd_to_coordinate(self,data, header):
        coordinate = []
        x_scale = header['space directions'][0][0]
        y_scale = header['space directions'][1][1]
        z_scale = header['space directions'][2][2]
        for x in range(len(data)):
            for y in range(len(data[0])):
                for z in range(len(data[0][0])):
                    if data[x][y][z] > 0:
                        coordinate.append([x * x_scale, y * y_scale, z * z_scale])
        return np.array(coordinate)

    def write_xyz_am(self,file, coordinate):
        with open(f'{file}.am', 'wt')as ff:
            ff.writelines('# AmiraMesh 3D ASCII 2.0\ndefine Markers ' + str(len(
                coordinate)) + '\nParameters {    ContentType "LandmarkSet",\n    NumSets 1\n}\nMarkers { float[3] Coordinates } @1\n# Data section follows\n@1\n')
            for point in coordinate:
                ff.writelines(f"{point[0]} {point[1]} {point[2]}\n")

    def get_header(self,img, dimension, translation):
        for i in range(3):
            img.header['dim'][i + 1] = dimension[i]
            img.affine[i][3] = translation[i]

        img.header['qoffset_x'] = translation[0]
        img.header['qoffset_y'] = translation[1]
        img.header['qoffset_z'] = translation[2]
        return img

    # @njit()
    def get_max_min_of_coordinates(self,xyz):
        xyz_min = np.zeros((3))
        xyz_max = np.zeros((3))
        for i in np.arange(0, 3):
            xyz_min[i] = np.min(xyz[:, i])
            xyz_max[i] = np.max(xyz[:, i])
        return xyz_min, xyz_max

    # @njit()
    def get_dimenstion(self,xyz_min, xyz_max, resoultion=1):
        dim = np.zeros(3, dtype=int)
        for i in np.arange(0, 3):
            dim[i] = int(math.ceil((xyz_max[i] - xyz_min[i]) / resoultion))
        return dim

    @njit()
    def get_nii_array(self,EM_neuron_xyz, xyz_min, xyz_max, dim, resolution=1):
        ##order: x,z,y ------------wait for check
        data = np.zeros((dim[0], dim[1], dim[2]))
        for i in np.arange(0, len(EM_neuron_xyz)):
            x = math.floor((EM_neuron_xyz[i][0] - xyz_min[0]) / resolution)
            y = math.floor((EM_neuron_xyz[i][1] - xyz_min[1]) / resolution)
            z = math.floor((EM_neuron_xyz[i][2] - xyz_min[2]) / resolution)
            data[x][y][z] = 2000
        return data

    def get_affine(self,xyz_min):
        affine = np.eye(4)
        for i in np.arange(0, 3):
            affine[i][3] = xyz_min[i]
        return affine

    def from_swc_to_nii(self,EM_neuron_xyz, file, savefile=True, resoultion=1):
        '''
        1. get min_xyz, max_xyz and then get bounding box and dimension
        2. transform to array-like data by translation ****** the strength will be set to 2000 (swc data does not contain original intensity info)
        3. save file
        :return:
        '''
        xyz_min, xyz_max = get_max_min_of_coordinates(EM_neuron_xyz)
        dim = get_dimenstion(xyz_min, xyz_max, resoultion)
        data = get_nii_array(EM_neuron_xyz, xyz_min, xyz_max, dim, resolution=1)
        affine = get_affine(xyz_min)
        array_img = nib.Nifti1Image(data, affine)
        if savefile == True:
            nib.save(array_img, f'{file}.nii.gz')
        return array_img

    def get_example_img(self):
        path = "FC1.3_PN/"
        file = "G0239-F-000001.nii.gz"
        n1_img = nib.load(path + file)
        return n1_img

    def read_xyz_file(self,file_name):
        coordinate = []
        with open(file_name,"rt")as ff:
            for line in ff:
                if line.find("\n")!=-1:
                    line = line[:-1]
                x,y,z = [float(i) for i in line.split(" ")]
                coordinate.append([x,y,z])
        return coordinate

    def read_swc_transform(self,file_name) -> (list, list):
        neuron_coordinate = []
        pooled_data = []
        check_index = 10
        ptr_index = 0
        re_float = re.compile(
            r"([\d]+)[\s]+([\+|\-]?[\d]+)[\s]+([\+|\-]?[\d]+[\.][\d]+)[\s]+([\+|\-]?[\d]+[\.][\d]+)[\s]+([\+|\-]?[\d]+[\.][\d]+)[\s]+([\+|\-]?[\d]+[\.][\d]+)[\s]+([\+|\-]?[\d]+)"
        )
        re_int= re.compile(
            r"([\d]+)[\s]+([\+|\-]?[\d]+)[\s]+([\+|\-]?[\d]+)[\s]+([\+|\-]?[\d]+)[\s]+([\+|\-]?[\d]+)[\s]+([\+|\-]?[\d]+[\.][\d]+)[\s]+([\+|\-]?[\d]+)"
        )
        # with open(file_name,'rt')as ff:
        #     for line in ff:
        #         for iter in re_float.finditer(line):
        #             pooled_data.append([iter.group(i) for i in range(1, 8)])
        #     print(pooled_data)
        with open(file_name,'rt')as ff:
            for line in ff:
                if re.search('[a-zA-Z]', line) or '#' in line:
                    continue
                if line[-1] == '\n':
                    line=line[:-1]
                group = line.split(" ")
                pooled_data.append(group)
        neuron_coordinate = [[float(i[2]),float(i[3]),float(i[4])] for i in pooled_data]
        return neuron_coordinate, pooled_data

    def get_transform(self,source,target,file_list,path='Data/xyz/',overwrite=True):
        if not os.path.isdir(path):
            os.mkdir(path)
        for file in file_list:
            print(f"{path}{file}")
            if overwrite==False:
                tmp_list = os.listdir(path)
                if 'swc' in file:
                    if f"{file[:-4]}_from_{source}_to_{target}.swc" in tmp_list:
                        continue
                    with open(f"{path}{file[:-4]}_from_{source}_to_{target}.swc",'wt')as ff:
                        ff.writelines("tmp")
                elif 'txt' in file:
                    if f"{file[:-4]}_from_{source}_to_{target}.txt" in tmp_list:
                        continue
                    with open(f"{path}{file[:-4]}_from_{source}_to_{target}.txt",'wt')as ff:
                        ff.writelines("tmp")
            print(self.label_to_template_dict[source],self.label_to_template_dict[target])
            print(source,self.label_to_template_dict[source])
            print(target,self.label_to_template_dict[target])
            # print(file[-4:])
            if '.txt' in file[-4:]:
                print('txt')
                coordinate = np.array(self.read_xyz_file(f"{path}{file}"))
            elif '.swc' in file[-4:]:
                print('swc')
                coordinate, otherinfo = self.read_swc_transform(f"{path}{file}")
                coordinate = np.array(coordinate)
                print("coordinate")
                print(coordinate)
                print("otherinfo")
                print(otherinfo)
            # print(coordinate[0],len(coordinate))
            if 'FlyCircuit' not in target and 'FlyCircuit' not in source:
                coordinate = list(warp_tool.transform_from_source_to_target(
                    coordinate, source=self.label_to_template_dict[source],
                    target=self.label_to_template_dict[target]))
            elif 'FlyCircuit' in source:
                coordinate = warp_tool.transform_from_source_to_target(warp_tool.FC_to_FCWB_transform(
                    coordinate,self.tf_param_FC2FCWB),source=self.label_to_template_dict[source],
                    target=self.label_to_template_dict[target])
            elif 'FlyCircuit' in target:
                coordinate = warp_tool.FCWB_to_FC_transform(
                    warp_tool.FCWB_to_FC_first_step(warp_tool.transform_from_source_to_target(
                    coordinate, source=self.label_to_template_dict[source],
                    target=self.label_to_template_dict[target])),self.tf_param_FCWB2FC)
            if 'swc' in file:
                with open(f"{path}{file[:-4]}_from_{source}_to_{target}.swc",'wt')as ff:
                    for point,para in zip(coordinate,otherinfo):
                        ff.writelines(f"{para[0]} {para[1]} {point[0]} {point[1]} {point[2]} {para[5]} {para[6]}\n")
            else:
                with open(f"{path}{file[:-4]}_from_{source}_to_{target}.txt",'wt')as ff:
                    for point in coordinate:
                        ff.writelines(f"{point[0]} {point[1]} {point[2]}\n")
    def transform_xyz(self,source, target, coordinate) -> list:
        print(self.label_to_template_dict[source], self.label_to_template_dict[target])
        print(source, self.label_to_template_dict[source])
        print(target, self.label_to_template_dict[target])
        coordinate = np.array(coordinate)
        print("coordinate")
        print(coordinate[0])
        a = warp_tool.transform_from_source_to_target(
                coordinate, source=self.label_to_template_dict[source],
                target=self.label_to_template_dict[target])
        print("a1",a)
        a = warp_tool.FCWB_to_FC_first_step(a)
        print("a2",a)
        a = warp_tool.FCWB_to_FC_transform(a,self.tf_param_FCWB2FC)
        print("a3",a)
        coordinate = warp_tool.FCWB_to_FC_transform(
            warp_tool.FCWB_to_FC_first_step(warp_tool.transform_from_source_to_target(
                coordinate, source=self.label_to_template_dict[source],
                target=self.label_to_template_dict[target])), self.tf_param_FCWB2FC)
        return list(coordinate)

class generate_synaptic_dendrogram():
    def __init__(self,brain_template='FlyCircuit'):
        self.dendrogram_path = 'Output/Analysis_result/dendrogram/'
        self.finished_neuron_path = "Data/em_skeletons_bridge_so_v1_1/"
        self.finished_neuron_cc_path = "Data/skeleton_via_ChaoChung/"
        self.brain_template = brain_template

    def calculate_dis(self,v1, v2):
        return ((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2) ** 0.5

    def draw_neuron_swc_synapse(self,neuron_coordinate, parent_list, path_length_list, Synapse_dict, Instance_record_list,
                                color_list):
        max_path = max(path_length_list)
        f = plt.figure(figsize=(16,12))
        ax = f.gca(projection='3d')
        for i in range(1, len(neuron_coordinate)):
            plt.plot([neuron_coordinate[i][0], neuron_coordinate[parent_list[i]][0]],
                     [neuron_coordinate[i][1], neuron_coordinate[parent_list[i]][1]],
                     [neuron_coordinate[i][2], neuron_coordinate[parent_list[i]][2]],
                     color=(path_length_list[i] / max_path, 0, 0))
        # print(len(neuron_coordinate))
        record_list = []
        for Neuron_id in Synapse_dict:
            synaptic_coordinate = np.array(Synapse_dict[Neuron_id])
            if Instance_dict[Neuron_id] not in record_list:
                record_list.append(Instance_dict[Neuron_id])
                plt.plot(synaptic_coordinate[:, 0], synaptic_coordinate[:, 1], synaptic_coordinate[:, 2], ".",
                         color=color_list[Instance_record_list.index(Instance_dict[Neuron_id])],
                         label=Instance_dict[Neuron_id])
            else:
                plt.plot(synaptic_coordinate[:, 0], synaptic_coordinate[:, 1], synaptic_coordinate[:, 2], ".",
                         color=color_list[Instance_record_list.index(Instance_dict[Neuron_id])])
            plt.legend()

        plt.show()

    def draw_neuron_swc(self,neuron_coordinate, parent_list, path_length_list):
        max_path = max(path_length_list)
        f = plt.figure()
        ax = f.gca(projection='3d')
        for i in range(1, len(neuron_coordinate)):
            plt.plot([neuron_coordinate[i][0], neuron_coordinate[parent_list[i]][0]],
                     [neuron_coordinate[i][1], neuron_coordinate[parent_list[i]][1]],
                     [neuron_coordinate[i][2], neuron_coordinate[parent_list[i]][2]],
                     color=(path_length_list[i] / max_path, 0, 0))
        # print(len(neuron_coordinate))
        plt.show()

    def read_swc(self,file_name) -> (list, list, list, list):
        neuron_coordinate = []
        daughter_list = []
        parent_list = []
        path_length_list = []
        so_list = []
        with open(self.finished_neuron_cc_path + file_name, "rt")as ff:
            for line in ff:
                if line.find("\n") != -1:
                    line = line[:-1]
                group = line.split(" ")
                neuron_coordinate.append([int(float(group[2])), int(float(group[3])), int(float(group[4]))])
                parent = int(group[6]) - 1
                parent_list.append(parent)
                daughter_list.append([])
                if parent >= 0:
                    daughter_list[parent].append(int(group[0]) - 1)
                if len(group) == 8:
                    so_list.append(int(group[7]))
                if parent < 0:
                    path_length_list.append(0)
                else:
                    path_length_list.append(
                        self.calculate_dis(neuron_coordinate[-1], neuron_coordinate[parent]) + path_length_list[parent])
        return neuron_coordinate, daughter_list, parent_list, path_length_list, so_list

    def _get_branch_id(self,daughter_list, parent_list, path_length_list):
        branch_id_list = []
        branch_path_length = []
        branch_id = 0
        start_path_length = 0
        branch_parent_list = []
        tmp_parent = -1
        start_index = -1
        branch_path_length_index = []
        for node_id in range(len(parent_list)):
            ##bipolar neuron, 第0個點本身也沒有任何長度可言##需再作check
            if node_id == 0:
                start_path_length = 0
                start_index = 0
                tmp_parent = -1
            ##上個點也是分岔點或結束點##
            elif len(daughter_list[parent_list[node_id]]) > 1:
                start_index = parent_list[node_id]
                start_path_length = path_length_list[parent_list[node_id]]
                tmp_parent = branch_id_list[parent_list[node_id]]
            if len(daughter_list[node_id]) != 1:
                branch_id_list.append(branch_id)
                branch_path_length.append([start_path_length, path_length_list[node_id]])
                branch_parent_list.append(tmp_parent)
                branch_path_length_index.append([start_index, node_id])
                branch_id += 1
            else:
                branch_id_list.append(branch_id)
        # for i in range(len(daughter_list)):
        #     print(daughter_list[i],branch_id_list[i])
        # print(branch_path_length_index)
        # print(len(branch_id_list),len(daughter_list))
        # print(len(branch_path_length),len(branch_parent_list))
        return branch_id_list, branch_path_length, branch_parent_list

    '''
    可能的加速作法
    1. 切格子，只比對附近的格子
    2. 計算的部分找numba

    '''

    # @njit()
    def get_synaptic_candidate(self,neuron_coordinates, synapse_coordinates, index_list, dis_threshold):
        for sid in range(len(synapse_coordinates)):
            min_dis = dis_threshold
            index = -1
            for nid in range(len(neuron_coordinates)):
                dis = np.linalg.norm(synapse_coordinates[sid] - neuron_coordinates[nid])
                if dis < min_dis:
                    min_dis = dis
                    index = nid
                print(nid)
                # break
            if index != -1:
                index_list[sid] = index
        return index_list
    def get_threshold(self):
        if self.brain_template == 'FlyEM':
            return 375
        elif self.brain_template == 'FlyCircuit':
            return 3
    def divide_neuron_into_blocks(self,Synapse_dict, neuron_coordinate, file_name):
        '''
        要做的事情:
        1. 遺漏多少點
        2. 收集遺漏的點座標
        3. 確認dendrogram的座標是否取的是對的→製作examples
        '''
        synapse_distance_threshold = self.get_threshold()  ### block_size, unit length is 8nm so the threshold is 8*200 nm
        block_size = synapse_distance_threshold
        xmin = min(neuron_coordinate[:][0])
        ymin = min(neuron_coordinate[:][1])
        zmin = min(neuron_coordinate[:][2])
        neuron_block = {}
        print("construct neuron block")
        for x, y, z in neuron_coordinate:
            x_index = int((x - xmin + block_size) / block_size)  ### For padding, we add blocksize to shift the index.
            y_index = int((y - ymin + block_size) / block_size)
            z_index = int((z - zmin + block_size) / block_size)
            if (x_index, y_index, z_index) not in neuron_block:
                neuron_block[(x_index, y_index, z_index)] = []
            neuron_block[(x_index, y_index, z_index)].append([x, y, z])
        print(neuron_block)
        # for block_index in neuron_block:
        #     neuron_block[block_index] = np.array(neuron_block[block_index])
        fail_dict = {}
        Final_result_dict = {}
        for Neuron_id in Synapse_dict:
            fail_dict[Neuron_id] = 0
            synapse_block = {}
            syn_to_neuron_coordinate = []
            syn_to_neuron_index = []
            ## get corresponding coordinate in the skeleton
            print(f"construct {Neuron_id} synapse block")
            synapse_block[(-10000, -10000, -10000)] = []  # To record the failed coordinate
            for s_index in range(len(Synapse_dict[Neuron_id])):
                x, y, z = Synapse_dict[Neuron_id][s_index]
                x_index = int((x - xmin + block_size) / block_size)
                y_index = int((y - ymin + block_size) / block_size)
                z_index = int((z - zmin + block_size) / block_size)
                if x_index < 0 or y_index < 0 or z_index < 0:
                    print("cannot find correspoding synapse site on the skeleton.")
                    fail_dict[Neuron_id] += 1
                    synapse_block[(-10000, -10000, -10000)].append([x, y, z])
                    continue
                if (x_index, y_index, z_index) not in synapse_block:
                    synapse_block[(x_index, y_index, z_index)] = []
                synapse_block[(x_index, y_index, z_index)].append([x, y, z])
            print(synapse_block)
            print("finding partner")
            for synapse_block_index in synapse_block:
                s_x_index, s_y_index, s_z_index = synapse_block_index
                candidate_block_index_list = [(s_x_index + i, s_y_index + j, s_z_index + k) for i in range(-1, 2, 1) for
                                              j in range(-1, 2, 1) for k in range(-1, 2, 1)]
                pooled_coordinates = []
                for candidate_block_id in candidate_block_index_list:
                    if candidate_block_id in neuron_block:
                        pooled_coordinates += neuron_block[candidate_block_id]
                np_pooled_neuron_coordinates = np.array(pooled_coordinates, dtype=float)
                target_synapse_coordinates = np.array(synapse_block[synapse_block_index], dtype=float)
                corresponding_index_list = self.get_synaptic_candidate(np_pooled_neuron_coordinates,
                                                                  target_synapse_coordinates,
                                                                  np.ones(len(target_synapse_coordinates),
                                                                          dtype=int) * -1, synapse_distance_threshold)
                for i in corresponding_index_list:
                    if i < 0:
                        fail_dict[Neuron_id] += -1
                        syn_to_neuron_coordinate.append([math.nan, math.nan, math.nan])
                        syn_to_neuron_index.append(-1)
                    else:
                        syn_to_neuron_coordinate.append([pooled_coordinates[i]])
                        syn_to_neuron_index.append(neuron_coordinate.index(pooled_coordinates[i]))
            # data_merge = np.array([[target_synapse_coordinates[i][0],target_synapse_coordinates[i][1],target_synapse_coordinates[i][2],
            #                  syn_to_neuron_coordinate[i][0],syn_to_neuron_coordinate[i][1],syn_to_neuron_coordinate[i][2]]
            #                 for i in range(len(target_synapse_coordinates))])
            # data = Df(data=data_merge,columns=['syn_x','syn_y','syn_z','syn_to_neuron_x','syn_to_neuron_y','syn_to_neuron_z'])
            # data.to_excel(f"{file_name}_{Neuron_id}.xlsx")
            print(syn_to_neuron_index)
            Final_result_dict[Neuron_id] = syn_to_neuron_index
            print(f"finish {Neuron_id}")
        return Final_result_dict

    def draw_dendrogram_like_synaptic_arrangement(self,Synapse_dict, neuron_coordinate, Instance_dict, branch_id_list,
                                                  branch_path_length, branch_parent_list, path_length_list,color_dict, file_name):
        Synaptic_position_index_list_collection = self.divide_neuron_into_blocks(Synapse_dict, neuron_coordinate, file_name)
        print("Start to draw dendrogram")
        f = plt.figure(figsize=(12,6))
        ax = f.add_subplot(111)
        count = 0
        for branch_id in range(len(branch_path_length)):
            # print(branch_path_length[branch_id])
            ax.plot(branch_path_length[branch_id], [branch_id, branch_id], 'k')
            if branch_parent_list[branch_id] != -1:
                count += 1
                ax.plot([branch_path_length[branch_id][0], branch_path_length[branch_id][0]],
                         [branch_id, branch_parent_list[branch_id]], "k")
        # print(count)
        # print(Synapse_dict)
        record_list = []
        neuron_list = list(Synaptic_position_index_list_collection.keys())
        sorted_neuron_list = sorted(range(len(neuron_list)),
                                    key=lambda k: Synaptic_position_index_list_collection[neuron_list[k]], reverse=True)
        neuron_list = [neuron_list[sorted_neuron_list[i]] for i in range(len(neuron_list))]
        if len(neuron_list)>15:
            neuron_list = neuron_list[:15]
        for Neuron_id in Synaptic_position_index_list_collection:
            synaptic_position_index_list = Synaptic_position_index_list_collection[Neuron_id]
            synaptic_y_pos = [branch_id_list[i] for i in synaptic_position_index_list]
            synaptic_x_pos = [path_length_list[i] for i in synaptic_position_index_list]
            if Instance_dict[Neuron_id] not in record_list:
                record_list.append(Instance_dict[Neuron_id])
                # plt.plot(synaptic_x_pos, synaptic_y_pos, ".",markersize=8,
                #          color=color_dict[Neuron_id],
                #          label=Instance_dict[Neuron_id])
                if Neuron_id in neuron_list:
                    ax.scatter(synaptic_x_pos, synaptic_y_pos, s=40,
                         c=[color_dict[Neuron_id][0] for _ in range(len(synaptic_x_pos))],
                         label=Instance_dict[Neuron_id],
                            cmap=color_dict[Neuron_id][1],vmax=1.0,vmin=0.0)
                else:
                    ax.scatter(synaptic_x_pos, synaptic_y_pos, s=40,
                                   c=[color_dict[Neuron_id][0] for _ in range(len(synaptic_x_pos))],
                                   cmap=color_dict[Neuron_id][1], vmax=1.0, vmin=0.0)
            else:
                # plt.plot(synaptic_x_pos, synaptic_y_pos, ".",markersize=8,
                #          color=color_dict[Neuron_id],label=str(Neuron_id))
                if Neuron_id in neuron_list:
                    ax.scatter(synaptic_x_pos, synaptic_y_pos,s=40,
                           c=[color_dict[Neuron_id][0] for _ in range(len(synaptic_x_pos))],
                            label=str(Neuron_id),
                            cmap=color_dict[Neuron_id][1],vmax=1.0,vmin=0.0)
                else:
                    ax.scatter(synaptic_x_pos, synaptic_y_pos, s=40,
                               c=[color_dict[Neuron_id][0] for _ in range(len(synaptic_x_pos))],
                               cmap=color_dict[Neuron_id][1], vmax=1.0, vmin=0.0)

            # plt.legend()
        # h, l = ax.get_legend_handles_labels()
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',fontsize=14)
        # plt.legend(h[:10], l[:10],bbox_to_anchor=(1.05, 1.0), loc='upper left',fontsize=14)
        plt.title(file_name,fontsize=12)
        # plt.xlim([1000,10000])
        plt.tight_layout()
        plt.savefig(f"{self.dendrogram_path}{file_name}.png")
        # plt.show()
        plt.close()

    def draw_dendrogram_like_synaptic_arrangement_old(self,Synapse_dict, neuron_coordinate, Instance_dict, branch_id_list,
                                                      branch_path_length, branch_parent_list, path_length_list,
                                                      Instance_record_list, color_list):
        f = plt.figure()
        count = 0
        for branch_id in range(len(branch_path_length)):
            # print(branch_path_length[branch_id])
            plt.plot(branch_path_length[branch_id], [branch_id, branch_id], 'k')
            if branch_parent_list[branch_id] != -1:
                count += 1
                plt.plot([branch_path_length[branch_id][0], branch_path_length[branch_id][0]],
                         [branch_id, branch_parent_list[branch_id]], "k")
        # print(count)
        # print(Synapse_dict)
        record_list = []
        for Neuron_id in Synapse_dict:
            ## get corresponding coordinate in the skeleton
            synaptic_position_index_list = []
            for s_index in range(len(Synapse_dict[Neuron_id])):
                min_dis = 1000000000000
                min_point_index = -1
                for n_index in range(len(neuron_coordinate)):
                    dis = self.calculate_dis(Synapse_dict[Neuron_id][s_index], neuron_coordinate[n_index])
                    if dis < min_dis:
                        min_dis = dis
                        min_point_index = n_index
                if min_dis > 1000 or min_point_index == -1:
                    print("Don't find correspoding synapse site.")
                else:
                    synaptic_position_index_list.append(min_point_index)
            synaptic_y_pos = [branch_id_list[i] for i in synaptic_position_index_list]
            synaptic_x_pos = [path_length_list[i] for i in synaptic_position_index_list]
            if Instance_dict[Neuron_id] not in record_list:
                record_list.append(Instance_dict[Neuron_id])
                plt.plot(synaptic_x_pos, synaptic_y_pos, ".",
                         color=color_list[Instance_record_list.index(Instance_dict[Neuron_id])],
                         label=Instance_dict[Neuron_id])
            else:
                plt.plot(synaptic_x_pos, synaptic_y_pos, ".",
                         color=color_list[Instance_record_list.index(Instance_dict[Neuron_id])])
            plt.legend()
        # plt.xlim([1000,10000])
        plt.show()

    def _get_Synapse_dict_by_id(self,target_id, file_name, path_connection,neuropil=''):
        Synapse_dict, Instance_dict = defaultdict(list), defaultdict(str)
        connection_data = pd.read_excel(path_connection + file_name)
        neuropil_list = []
        answer = [[] for _ in range(len(connection_data['up.bodyId']))]
        if neuropil:
            neuropil_list = neuropil.split(" ")
            xyz = np.array([connection_data["up_syn_coordinate_x"].values.tolist(),connection_data["up_syn_coordinate_y"].values.tolist(),connection_data["up_syn_coordinate_z"].values.tolist()]).T
            xyz = xyz.tolist()
            answer = NN.query_coordinate_identity(xyz)
            print(answer)
        for up_id,up_type,down_id,down_type,up_syn_x,up_syn_y,up_syn_z,down_syn_x,down_syn_y,down_syn_z,node_neuropil in zip(
                connection_data['up.bodyId'],connection_data['up.type'],connection_data['down.bodyId'],connection_data['down.type'],
                connection_data["up_syn_coordinate_x"],connection_data["up_syn_coordinate_y"],connection_data["up_syn_coordinate_z"],
                connection_data["down_syn_coordinate_x"],connection_data["down_syn_coordinate_y"],connection_data["down_syn_coordinate_z"],
                answer
        ):
            if int(up_id) == int(target_id):
                if str(down_type).find("nan")!=-1:
                    neuron_type = str(down_id)
                else:
                    neuron_type = str(down_type)+"_"+str(down_id)
                Instance_dict[down_id] = neuron_type
                if neuropil_list:
                    for neuropil in neuropil_list:
                        if neuropil in node_neuropil:
                            Synapse_dict[down_id].append([up_syn_x, up_syn_y, up_syn_z])
                            break
                else:
                    Synapse_dict[down_id].append([up_syn_x,up_syn_y,up_syn_z])

            elif int(down_id) == int(target_id):
                if str(up_type).find("nan")!=-1:
                    neuron_type = str(up_id)
                else:
                    neuron_type = str(up_type)+"_"+str(up_id)
                Instance_dict[up_id] = neuron_type
                if neuropil_list:
                    for neuropil in neuropil_list:
                        if neuropil in node_neuropil:
                            Synapse_dict[down_id].append([up_syn_x, up_syn_y, up_syn_z])
                            break
                else:
                    Synapse_dict[up_id].append([down_syn_x,down_syn_y,down_syn_z])
        return Synapse_dict, Instance_dict

def generate_network_file(connection_data,node_type='bodyId',type_cluster=True):
    connection_data_list = copy.deepcopy(connection_data).values.tolist()
    node_list = []
    id_edge_dict = {}
    type_edge_dict = defaultdict(list)
    id_to_type_dict = defaultdict(str)
    type_to_id_dict = defaultdict(list)
    f = open('network_sample.txt','wt')
    for connection in connection_data_list:
        if connection[1] not in node_list:
            node_list.append(connection[1])
            if not connection[3]:
                connection[3] = 'unknown'
            id_to_type_dict[connection[1]] = connection[3]
            type_to_id_dict[connection[3]].append(connection[1])
        if connection[4] not in node_list:
            node_list.append(connection[4])
            if not connection[6]:
                connection[6] = 'unknown'
            id_to_type_dict[connection[4]] = connection[6]
            type_to_id_dict[connection[6]].append(connection[4])
        id_edge_dict[(connection[1],connection[4])] = connection[7]
        type_edge_dict[(connection[3],connection[4])].append(connection[7])
        f.writelines(f"{connection[1]} {connection[4]}\n")
    f.close()

def generate_cytoscope(connection_data,node_type='bodyId',type_cluster=True):
    connection_data_list = copy.deepcopy(connection_data).values.tolist()
    node_list = []
    id_edge_dict = {}
    type_edge_dict = defaultdict(list)
    id_to_type_dict = defaultdict(str)
    type_to_id_dict = defaultdict(list)
    for connection in connection_data_list:
        if connection[0] not in node_list:
            node_list.append(connection[0])
            if not connection[2]:
                connection[2] = 'unknown'
            id_to_type_dict[connection[0]] = connection[2]
            type_to_id_dict[connection[2]].append(connection[0])
        if connection[3] not in node_list:
            node_list.append(connection[3])
            if not connection[5]:
                connection[5] = 'unknown'
            id_to_type_dict[connection[3]] = connection[5]
            type_to_id_dict[connection[5]].append(connection[3])
        id_edge_dict[(connection[0],connection[3])] = connection[6]
        type_edge_dict[(connection[2],connection[3])].append(connection[6])


    for node in node_list:
        data = {'id': f'{node}', 'label': f'{id_to_type_dict[node]}'}
        element = {}
        element['data'] = data
        if type_cluster:
            element['position']


def get_network_graph(connection_data,node_type='bodyId',style='spring'):
    Graph = nx.from_pandas_edgelist(connection_data, source=f'up.{node_type}', target=f'down.{node_type}', edge_attr=True, create_using=nx.DiGraph())
    if style == 'spring':
        pos = nx.spring_layout(Graph)
    figure = networkGraph(Graph,pos)
    return figure

def networkGraph(G,pos):

    # edges trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(color='black', width=1),
        hoverinfo='none',
        showlegend=False,
        mode='lines')

    # nodes trace
    node_x = []
    node_y = []
    text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y, text=text,
        mode='markers+text',
        showlegend=False,
        hoverinfo='none',
        marker=dict(
            color='pink',
            size=50,
            line=dict(color='black', width=1)))

    # layout
    layout = dict(plot_bgcolor='white',
                  paper_bgcolor='white',
                  margin=dict(t=10, b=10, l=10, r=10, pad=0),
                  xaxis=dict(linecolor='black',
                             showgrid=False,
                             showticklabels=False,
                             mirror=True),
                  yaxis=dict(linecolor='black',
                             showgrid=False,
                             showticklabels=False,
                             mirror=True))

    # figure
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

    return fig

if __name__=='__main__':
    warp = brain_template_warping_tool()
    path = 'Data/skeleton_via_ChaoChung/'
    file_list = [i for i in os.listdir(path) if i.find("FlyCircuit")==-1]
    print(file_list)
    # file_list = ['k7a8-E-941469110_seg001_Fix.swc']
    warp.get_transform(source="FlyEM_um",target='FlyEM',file_list=file_list,path=path,overwrite=True)
    # a=np.array([[32215, 30045, 15601], [31710, 29190, 16070]])
    # print(a)
    # a = warp_tool.FCWB_to_FC_transform(
    #     warp_tool.FCWB_to_FC_first_step(warp_tool.transform_from_source_to_target(
    #         coordinate, source='JRCFIB2018Fraw',
    #         target='FCWB')), warp.tf_param_FCWB2FC)
    # print(a)