import math
import numpy as np
import os
import random as rd
import plotly.graph_objects as go
import pickle
import plotly
import dash_daq as daq
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from pandas import DataFrame as Df
import re
from neuprint import Client
import pandas as pd
import time as tt
import eFlyplot_image_processing_v2 as eFlyplot_image_processing
import copy
import eFlyPlot_layout
from collections import Counter
import test_cyto
import eFlyPlot_analysis as eFlyA

##Available datasets: ['hemibrain:v0.9', 'hemibrain:v1.0.1', 'hemibrain:v1.1', 'hemibrain:v1.2.1']
'''
目前問題:
version和label要怎麼從gui加入
(若每次都要打~或是都要改好麻煩)
若用拉的~直接就有version等相關的值
目前還欠缺connection的callback
'''

class CoordinateCollection:
    def __init__(self, template='FlyEM'):
        self.xyz_path = "Data/xyz/"
        self.xyz_collection = {}
        self.FlyEM_swc_path = "Data/FlyEM_skeletons/"
        self.swc_parent_collection = {}
        self.FC_swc_path = "Data/FC_skeletons/"
        self.template = template
        if template == 'FlyEM':
            self.neuropil_path = "FlyEM_neuropil/"
        elif template == 'FlyCircuit':
            self.neuropil_path = ""
        else:
            self.neuropil_path = input("Please specify your path of neuropil template.\n")
        neuropil = []
        neuropil_values = []
        for neuropil in os.listdir(neuropil_path):
            neuropil_options.append({'label': neuropil[:-4], 'value': neuropil[:-4]})
            neuropil_values.append(neuropil[:-4])

        self.neuron_swc_path = f'dash_{self.template}_neuron/'
        self.neuropil_mesh_path = 'dash_EM_neuropil/'
        self.xyz_path = 'dash_EM_xyz/'
        self.neuropil_options = neuropil
        self.neuropil_values = neuropil_values

        if os.path.isfile(neuropil_mesh_path + 'neuro_pil_mesh.pickle'):
            with open(neuropil_mesh_path + 'neuro_pil_mesh.pickle', 'rb')as ff:
                all_neuropil_mesh = pickle.load(ff)
        else:
            print(f"First time to use {self.template}. Initializing for neuropil loading.")
            all_neuropil_mesh = obtain_neuropil_mesh(neuropil_path)
            with open(neuropil_mesh_path + 'neuro_pil_mesh.pickle', 'wb')as ff:
                pickle.dump(all_neuropil_mesh, ff)

        self.all_neuropil_mesh = all_neuropil_mesh
        return all_neuropil_mesh, neuropil_options, neuropil_values

    def read_xyz(self,file_name,brain_template="FlyEM"):
        Node_xyz = []
        with open(f"{file_name}", "rt") as ff:
            for line in ff:
                if line.find("#") != -1:
                    continue
                line = line[:-1]
                group = line.split(" ")
                Node_xyz.append([float(group[0]), float(group[1]), float(group[2])])
        self.xyz_collection[file_name] = Node_xyz
        return Node_xyz

    def read_swc(self,file_name,brain_template="FlyEM"):
        ## 須改正則運算
        Node_xyz = []
        node_parent = []
        with open(f"{file_name}", "rt") as ff:
            for line in ff:
                if line.find("#") != -1:
                    continue
                line = line[:-1]
                group = line.split(" ")
                Node_xyz.append([float(group[2]), float(group[3]), float(group[4])])
                node_parent.append(int(group[-1]))
        self.xyz_collection[file_name] = Node_xyz
        self.swc_parent_collection[file_name] = node_parent
        return Node_xyz, node_parent

    # def coordinate_warping(self, points,src,target):

class neuprint_query_tool:
    '''
        1. set basic neuprint interface
        2. incooperate neuprint survey tools and then return results for brain_space
        '''
    def __init__(self, version='v1.2.1',template='FlyEM'):
        self.Server = 'neuprint.janelia.org'
        self.Token = self.get_token()
        self.version = version
        self.template='FlyEM'
        self.connecting_neuprint_server()
        self.initialize_path()
        self.get_query_phrase()
        self.info_name_list = []
        self.conn_name_list = []
        self.syn_name_list = []


    def initialize_path(self):
        #### DATA ###############
        self.data_path = 'Data/'
        self.xyz_path = f"{self.data_path}{self.template}_xyz/"
        self.skeleton_path = f"{self.data_path}{self.template}_skeleton/"
        self.neuropil_path = f"{self.data_path}{self.template}_neuropil/"

        #### tmp_folder ############
        self.tmp_path = 'Tmp/'

        #### precomputed data ################
        self.precomputed_data_path = "Precomputed_data/"
        self.precomputed_skeleton_path = f'{self.precomputed_data_path}Dash_{self.template}_skeleton/'
        self.precomputed_xyz_path = f'{self.precomputed_data_path}Dash_{self.template}_xyz/'
        self.precomputed_neuropil_path = f'{self.precomputed_data_path}Dash_{self.template}__neuropil/'

        #### output ##########################
        self.output_path = 'output/'
        self.connection_data_path = f'{self.output_path}Connection_data/'
        self.synapse_data_path = f'{self.output_path}Synapse_data/'
        self.other_basic_info_path = f'{self.output_path}Other_basic_info/'
        self.analysis_result_path = f'{self.output_path}Analysis_result/'
        self.other_path = f"{self.output_path}Others/"
        ## data
        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.isdir(self.xyz_path):
            os.mkdir(self.xyz_path)
        if not os.path.isdir(self.skeleton_path):
            os.mkdir(self.skeleton_path)
        if not os.path.isdir(self.neuropil_path):
            os.mkdir(self.neuropil_path)
        ## tmp
        if not os.path.isdir(self.tmp_path):
            os.mkdir(self.tmp_path)
        if os.path.isfile(self.tmp_path + "search_button_click_record.pickle"):
            os.remove(self.tmp_path + "search_button_click_record.pickle")
        ## precomputed data
        if not os.path.isdir(self.precomputed_data_path):
            os.mkdir(self.precomputed_data_path)
        if not os.path.isdir(self.precomputed_skeleton_path):
            os.mkdir(self.precomputed_skeleton_path)
        if not os.path.isdir(self.precomputed_xyz_path):
            os.mkdir(self.precomputed_xyz_path)
        if not os.path.isdir(self.precomputed_neuropil_path):
            os.mkdir(self.precomputed_neuropil_path)
        ## output
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        if not os.path.isdir(self.connection_data_path):
            os.mkdir(self.connection_data_path)
        if not os.path.isdir(self.synapse_data_path):
            os.mkdir(self.synapse_data_path)
        if not os.path.isdir(self.other_basic_info_path):
            os.mkdir(self.other_basic_info_path)
        if not os.path.isdir(self.analysis_result_path):
            os.mkdir(self.analysis_result_path)
        if not os.path.isdir(self.other_path):
            os.mkdir(self.other_path)

    def get_token(self):
        token_file_name = "token_for_check_connection.txt"
        if not os.path.isfile(token_file_name):
            Token = input(
                "Please type your token\n If you don't know how to get token, please read the instruction or directly visit neuprint using your google account to login and then get token from your account information.\n")
            with open(token_file_name, "wt")as ff:
                ff.writelines(Token)
        else:
            with open(token_file_name, "rt")as ff:
                for line in ff:
                    if line.find("\n") != -1:
                        line = line[:-1]
                    Token = line
                    break
        return Token

    def connecting_neuprint_server(self):
        while 1:
            print(f'connecting {self.version}')
            try:
                c = Client(self.Server, dataset=f'hemibrain:{self.version}', token=self.Token)
                self.c = c
                return
            except:
                print("Please check your dataset version and your passing token.\n")
                self.version = input()

    ######## 先以現有的改完，之後再擴充功能
    def get_query_phrase(self, query_type='synapse'):
        '''
        info
        1. get roi
        2. get synapse
        3. get connection
        4. get type or other basic info

        subject
        1. itself
        2. intersection
        3.
        '''
        self.q_syn = "MATCH (up:`Neuron`)-[:Contains]->(:`SynapseSet`)-[:Contains]->(up_syn:`Synapse`)-[:SynapsesTo]->(down_syn:`Synapse`)<-[:Contains]-(:`SynapseSet`)<-[:Contains]-(down:`Neuron`)"
        self.r_syn = " RETURN DISTINCT up.bodyId, up.instance, up.type, down.bodyId, down.instance, down.type, up_syn.location, down_syn.location"
        self.q_con = "MATCH (up:Neuron)-[w:ConnectsTo]->(down:Neuron)"
        self.r_con = " RETURN DISTINCT up.bodyId, up.instance, up.type, down.bodyId, down.instance, down.type, w.weight, w.roiInfo"
        self.q_type = "MATCH (n:Neuron)"
        self.r_type = " RETURN DISTINCT n.bodyId, n.instance, n.type"
        self.r_type_simple = " RETURN DISTINCT n.instance, n.type"
        self.qn = "MATCH (n:Neuron)"
        self.rn = " RETURN DISTINCT n.bodyId, n.instance, n.type, n.roiInfo"

    ################ Generate query phrases ##########################################
    def lookup(self,line):
        if re.search('[a-zA-Z]', line): ## query condition is neuron type not bodyId
            condition = "type="
            if line.find("*") != -1: ## query condition contain fuzzy string
                condition = condition + "~"
            condition = condition + "\""
            if line[0].find("*") != -1:
                condition = condition + "."
            if line[-1].find("*") != -1:
                condition = condition + line[:-1] + ".*\""
            else:
                condition = condition + line + "\""
            return condition
        else:
            condition = "bodyId=" + line
            return condition

    def is_type(self,line):
        if line.find("*") != -1:
            return 2
        if re.search('[a-zA-Z]', line):
            return 1
        else:
            return 0

    def generate_group_upstream(self,neuron, candidate_list):
        condition = self.lookup(neuron)
        line = "["
        for candidate in candidate_list[:-1]:
            line = line + str(candidate) + ", "
        line = line + str(candidate_list[-1]) + "]"
        return " WHERE down." + condition + " AND up.bodyId IN " + str(line)

    def generate_group_downstream(self,neuron, candidate_list):
        condition = self.lookup(neuron)
        line = "["
        for candidate in candidate_list[:-1]:
            line = line + str(candidate) + ", "
        line = line + str(candidate_list[-1]) + "]"
        return " WHERE up." + condition + " AND down.bodyId IN " + str(line)

    def generate_condition_upstream(self,neuron):
        condition = self.lookup(neuron)
        return " WHERE down." + condition

    def generate_condition_downstream(self,neuron):
        condition = self.lookup(neuron)
        return " WHERE up." + condition

    def generate_condition_pair(self,group):
        pre, post = group
        condition_pre = self.lookup(pre)
        condition_post = self.lookup(post)
        return " WHERE up." + condition_pre + " AND down." + condition_post

    def forsave(self,line):
        if line[0].find("*") != -1:
            if line[-1].find("*") != -1:
                return line[1:-1]
            else:
                return line[1:]
        else:
            if line[-1].find("*") != -1:
                return line[:-1]
            else:
                return line

    def generate_condition_single(self,neuron):
        condition = self.lookup(neuron)
        return "WHERE n." + condition
    def generate_condition_neuron(self,neuron,symbol='n.'):
        condition = self.lookup(neuron)
        return f"{symbol}{condition}"
    def generate_condition_roi(self,roi,symbol='n.'):
        return f"{symbol}`{roi}`"
    def generate_condition_neuron_info(self,query,roi,result_info={},symbol='n.',file_name = "neuronInfo_"):
        condition = ' WHERE '
        #################### generate condition #######################
        if query:
            group = query.split(" ")
            for i in group:
                file_name = file_name + i + "_"
                condition = condition + self.generate_condition_neuron(i, symbol) + ' AND '
        if roi:
            group = roi.split(" ")
            for i in group:
                file_name = file_name + i + "_"
                condition = condition + self.generate_condition_roi(i, symbol) + ' AND '
        #################### generate return clause #######################
        r_type = " RETURN DISTINCT "
        if 'id' in result_info:
            r_type = r_type + f"{symbol}bodyId, "
            file_name = file_name + "id_"
        if 'type' in result_info:
            r_type = r_type + f"{symbol}type, {symbol}instance, "
            file_name = file_name + "type_"
        if 'roi' in result_info:
            r_type = r_type + f"{symbol}roiInfo, "
            file_name = file_name + "in_roi_"

        return condition, r_type[:-2], file_name

    def complicated_upstream_downstream_old(self,line_up, line_down, connection_type):
        if connection_type.find("connection"):
            q = self.q_con
            r = self.r_con
        elif connection_type.find("synapse"):
            q = self.q_syn
            r = self.r_syn
        division_number = 400
        print("Here we output the intersection of all conditions")
        All_up_dict = {}
        All_down_dict = {}
        if len(line_up) == 0 and len(line_down) == 0:
            return
        try:
            file_name = ""
            first = 0
            candidate_list = []
            if (line_up.find("\n")) != -1:
                line = line_up[:-1]
            else:
                line = line_up
            group = line.split(" ")
            if len(group[0]) != 0:
                print("upstream of ", group)
                if len(group) >= 1:
                    file_name = "upstream_of_"
                    for neuron in group:
                        if len(neuron) == 0:
                            break
                        neuron_name = neuron
                        if neuron[0].find("*") != -1:
                            neuron_name = neuron_name[1:]
                        if neuron[-1].find("*") != -1:
                            neuron_name = neuron_name[:-1]
                        file_name = file_name + neuron_name + "_"
                        # print(neuron_name)
                        # up_list.append(file_name)
                        if first == 0:
                            first = 1
                            condition = self.generate_condition_upstream(neuron)
                            q_tmp = q + condition + r
                            # print(q_tmp)
                            results = self.c.fetch_custom(q_tmp)
                            candidate_list = []
                            tmp_candidate_list = results['up.bodyId'].values.tolist()
                            # print("tmp",tmp_candidate_list)
                            updated_All_up_dict = {}
                            for candidate in tmp_candidate_list:
                                if candidate not in candidate_list:
                                    candidate_list.append(candidate)
                                if candidate not in All_up_dict:
                                    All_up_dict[candidate] = []
                                pooled_result = results[results['up.bodyId'] == candidate].values.tolist()
                                updated_All_up_dict[candidate] = All_up_dict[candidate] + pooled_result
                            All_up_dict = updated_All_up_dict
                            # print("candidate",candidate_list)

                            if len(candidate_list) == 0:
                                print("No candidate")
                                break
                        else:
                            tt.sleep(3)
                            list_length = len(candidate_list)
                            list_fraction = int(list_length / division_number)
                            tmp_record_list = candidate_list
                            time = 0
                            for time in range(list_fraction):
                                candidate_list = tmp_record_list[(time) * division_number:(time + 1) * division_number]
                                condition = self.generate_group_upstream(neuron, candidate_list)
                                q_tmp = q + condition + r
                                # print(q_tmp)
                                if time == 0:
                                    results = self.c.fetch_custom(q_tmp)
                                else:
                                    results = pd.merge(results, self.c.fetch_custom(q_tmp), how='outer')
                            if list_length - list_fraction * division_number > 0:
                                candidate_list = tmp_record_list[list_fraction * division_number:]
                                condition = self.generate_group_upstream(neuron, candidate_list)
                                q_tmp = q + condition + r
                                # print(q_tmp)
                                if list_fraction == 0:
                                    results = self.c.fetch_custom(q_tmp)
                                else:
                                    results = pd.merge(results, self.c.fetch_custom(q_tmp), how='outer')
                            candidate_list = []
                            tmp_candidate_list = results['up.bodyId'].values.tolist()
                            updated_All_up_dict = {}
                            for candidate in tmp_candidate_list:
                                if candidate not in candidate_list:
                                    candidate_list.append(candidate)
                                if candidate not in All_up_dict:
                                    All_up_dict[candidate] = []
                                pooled_result = results[results['up.bodyId'] == candidate].values.tolist()
                                updated_All_up_dict[candidate] = All_up_dict[candidate] + pooled_result
                            All_up_dict = updated_All_up_dict
                            if len(candidate_list) == 0:
                                print("No candidate")
                                break

            if (line_down.find("\n")) != -1:
                line = line_down[:-1]
            else:
                line = line_down
            group = line.split(" ")
            if len(group[0]) != 0:
                if len(group) >= 1:
                    file_name = file_name + "downstream_of_"
                    print("downstream of ", group)
                    for neuron in group:
                        if len(neuron) == 0:
                            break
                        neuron_name = neuron
                        if neuron[0].find("*") != -1:
                            neuron_name = neuron_name[1:]
                        if neuron[-1].find("*") != -1:
                            neuron_name = neuron_name[:-1]
                        file_name = file_name + neuron_name + "_"
                        # down_list.append(neuron_name)
                        if first == 0:
                            first = 1
                            condition = self.generate_condition_downstream(neuron)
                            q_tmp = q + condition + r
                            # print(q_tmp)
                            results = self.c.fetch_custom(q_tmp)
                            candidate_list = []
                            tmp_candidate_list = results['down.bodyId'].values.tolist()
                            updated_All_down_dict = {}
                            for candidate in tmp_candidate_list:
                                if candidate not in candidate_list:
                                    candidate_list.append(candidate)
                                if candidate not in All_down_dict:
                                    All_down_dict[candidate] = []
                                pooled_result = results[results['down.bodyId'] == candidate].values.tolist()
                                updated_All_down_dict[candidate] = All_down_dict[candidate] + pooled_result
                            All_down_dict = updated_All_down_dict
                            if len(candidate_list) == 0:
                                print("No candidate")
                                break
                        else:
                            tt.sleep(1)
                            list_length = len(candidate_list)
                            list_fraction = int(list_length / division_number)
                            tmp_record_list = candidate_list
                            time = 0
                            for time in range(list_fraction):
                                # print(time)
                                candidate_list = tmp_record_list[(time) * division_number:(time + 1) * division_number]
                                condition = self.generate_group_downstream(neuron, candidate_list)
                                q_tmp = q + condition + r
                                # print(q_tmp)
                                if time == 0:
                                    results = self.c.fetch_custom(q_tmp)
                                else:
                                    results = pd.merge(results, self.c.fetch_custom(q_tmp), how='outer')
                            if list_length - list_fraction * division_number > 0:
                                candidate_list = tmp_record_list[list_fraction * division_number:]
                                condition = self.generate_group_downstream(neuron, candidate_list)
                                q_tmp = q + condition + r

                                if list_fraction == 0:
                                    results = self.c.fetch_custom(q_tmp)
                                # print(q_tmp)
                                else:
                                    results = pd.merge(results, self.c.fetch_custom(q_tmp), how='outer')
                            candidate_list = []
                            tmp_candidate_list = results['down.bodyId'].values.tolist()
                            updated_All_down_dict = {}
                            for candidate in tmp_candidate_list:
                                if candidate not in candidate_list:
                                    candidate_list.append(candidate)
                                if candidate not in All_down_dict:
                                    All_down_dict[candidate] = []
                                pooled_result = results[results['down.bodyId'] == candidate].values.tolist()
                                updated_All_down_dict[candidate] = All_down_dict[candidate] + pooled_result
                            All_down_dict = updated_All_down_dict
                            if len(candidate_list) == 0:
                                print("No candidate")
                                break
                            if len(candidate_list) == 0:
                                print("No candidate")
                                break
            file_name = f"{file_name}intersection_{connection_type}_{self.version}.xlsx"
            intersection_data = []
            for candidate in candidate_list:
                if len(All_up_dict) != 0:
                    for data in All_up_dict[candidate]:
                        intersection_data.append(data)
                if len(All_down_dict) != 0:
                    for data in All_down_dict[candidate]:
                        intersection_data.append(data)

            intersection_data_df = Df(data=intersection_data)
            # intersection_data_df = pd.DataFrame.transpose(intersection_data_df)
            if connection_type.find("connection") != -1:
                intersection_data_df.columns = ["up.bodyId", "up.instance", "up.type", "down.bodyId", "down.instance",
                                                "down.type", "w.weight", "w.roiInfo"]
                intersection_data_df.to_excel(f'{self.connection_data_path}{file_name}')
                with open(f'{self.connection_data_path}{file_name[:-4]}.txt', "wt")as ff:
                    for candidate in candidate_list:
                        ff.writelines(str(candidate) + "\n")

            elif connection_type.find("synapse") != -1:
                intersection_data_df.columns = ["up.bodyId", "up.instance", "up.type", "down.bodyId", "down.instance",
                                                "down.type", "up_syn.location", "down_syn.location"]
                intersection_data_df.to_excel(f'{self.synapse_data_path}{file_name}')
                with open(f'{self.synapse_data_path}{file_name[:-4]}.txt', "wt")as ff:
                    for candidate in candidate_list:
                        ff.writelines(str(candidate) + "\n")
                return [results['up_syn_coordinate_x'].values.tolist(), results['up_syn_coordinate_y'].values.tolist(),
                results['up_syn_coordinate_z'].values.tolist()], [results['down_syn_coordinate_x'].values.tolist(),
                                                                  results['down_syn_coordinate_y'].values.tolist(),
                                                                  results['down_syn_coordinate_z'].values.tolist()]
        except:
            print("Encounter some problems. Maybe you need to do more accurate survey to reduce data amount.")
        return

    def get_neuron_in_roi(self,line):
        qn = self.qn
        rn = self.rn
        if len(line) == 0:
            return
        if (line.find("\n")) != -1:
            line = line[:-1]
        file_name = ""
        group = line.split(" ")
        condition = ' WHERE '
        for i in group:
            file_name = file_name + i + "_"
            condition = condition + self.generate_condition_roi(i) + ' AND '
            # q = qn + condition + rn
        condition = condition[:-5]
        q = f"""{qn}{condition}{rn}"""
        print(f"Searching neurons in roi: {condition}")
        try:
            results = self.c.fetch_custom(q)
        except:
            print(
                "This query fails. The reason may result from bad query request or the neuprint restriction. You can try it again.")
            pass
        results.to_excel(f"{self.other_basic_info_path}neurons_in_{str(forsave(file_name[:-1]))}_info_{self.version}.xlsx")
        print(f"neurons_in_{str(forsave(file_name[:-1]))}_info_{self.version} Finished\n")
        return
    def get_result_intersection(self,result_list_1,first_loc_index=0, result_list_2=[],second_loc_index=0):
        if not result_list_1:
            return []
        neuron_result_dict = {}
        result_number_1 = len(result_list_1)
        check = 1
        for result_id, result in enumerate(result_list_1):
            for neuron_data in result:
                if check == 1:
                    print("ccc",type(neuron_data))
                    print(neuron_data)
                    check=0
                if neuron_data[first_loc_index] not in neuron_result_dict:
                    if result_id != 0:
                        continue
                    neuron_result_dict[neuron_data[first_loc_index]] = np.zeros(result_number_1)
                    neuron_result_dict[neuron_data[first_loc_index]][0] = 1
                else:
                    neuron_result_dict[neuron_data[first_loc_index]][result_id] = 1
        candidate_neuron_list = [neuron for neuron in neuron_result_dict if sum(neuron_result_dict[neuron])==result_number_1]
        if not result_list_2:
            intersection_data = []
            for result in result_list_1:
                for neuron_data in result:
                    if neuron_data[first_loc_index] in candidate_neuron_list:
                        intersection_data.append(neuron_data)
            return intersection_data
        result_number_2 = len(result_list_2)
        intersection_data = []
        neuron_result_dict_2 = {}
        for result_id, result in enumerate(result_list_2):
            for neuron_data in result:
                if neuron_data[second_loc_index] not in candidate_neuron_list:
                        continue
                elif neuron_data[first_loc_index] not in neuron_result_dict_2:
                    if result_id != 0:
                        continue
                    neuron_result_dict_2[neuron_data[second_loc_index]] = np.zeros(result_number_2)
                    neuron_result_dict_2[neuron_data[second_loc_index]][0] = 1
                else:
                    neuron_result_dict_2[neuron_data[second_loc_index]][result_id] = 1
        final_candidate_neuron_list = [neuron for neuron in neuron_result_dict_2 if sum(neuron_result_dict_2[neuron])==result_number_2 and neuron in candidate_neuron_list]
        intersection_data = []
        for result in result_list_1:
            for neuron_data in result:
                if neuron_data[first_loc_index] in final_candidate_neuron_list:
                    intersection_data.append(neuron_data)
        for result in result_list_2:
            for neuron_data in result:
                if neuron_data[second_loc_index] in final_candidate_neuron_list:
                    intersection_data.append(neuron_data)
        return intersection_data

    def filtered_by_weight(self,results,weight_threshold):
        results = copy.deepcopy(results).values.tolist()
        Synapse_counting_dict = Counter()
        for connection in results:
            Synapse_counting_dict[(connection[0],connection[3])] += 1
        data = []
        for connection in results:
            if Synapse_counting_dict[(connection[0],connection[3])]>=weight_threshold:
                data.append(connection)
        return data


    def get_synapse_all(self,upn,upn_w,downn,downn_w,query,roi):
        columns = ['up.bodyId', 'up.instance', 'up.type', 'down.bodyId', 'down.instance', 'down.type', 'up_syn_coordinate_x',
                   'up_syn_coordinate_y', 'up_syn_coordinate_z', 'down_syn_coordinate_x', 'down_syn_coordinate_y', 'down_syn_coordinate_z']

        raw_columns = ['up.bodyId', 'up.instance', 'up.type', 'down.bodyId', 'down.instance', 'down.type', 'up_syn.location','down_syn.location']
        if not downn_w:
            downn_w = 0
        if not upn_w:
            upn_w = 0
        '''
        up.bodyId, up.instance, up.type, down.bodyId, down.instance, down.type, up_syn.location, down_syn.location
        幾種狀況:
        1. 若只有query，則分別取得query neurons的所有共同上游，以及共同下游
        2. 若有上游，則取得所有符合上游連結的query neuron，反之下游一樣
        3. 若同時有上游下游，則取得同時有連到這些上下游的神經元
        4. 若同時有上游下游+query，則同時要滿足神經只屬於query，且也要滿足同時連到這些上游及這些下游
        5. 只有上游+query，則同樣滿足這兩個的連結才取得


        '''
        if not upn and not downn and not query:
            return
        # elif not upn and not downn:
        else:
            ##只有query
            ## 先看query到下游
                #self.get_neuron_info_all(query,roi,upn,upn_w,downn,down_w, result_info)
            print("CHEKC0")
            query_neuron_info = self.get_neuron_info_all(query,roi,upn,upn_w,downn,downn_w, ['id','type','roi'])['n.type'].values.tolist()
            query_group = []
            if len(query_neuron_info) < 400:
                query_group.append(query)
            else:
                query_group = []
                for neuron_type in query_neuron_info:
                    if neuron_type not in query_group:
                        query_group.append(neuron_type)
            print("CHEKC1")
            # downn_neuron_group = []
            # if query or downn:
            #     downn_neuron_info = self.get_neuron_info_all(downn, "", query, downn_w, "", "", ['id', 'type', 'roi'])[
            #         'n.type'].values.tolist()
            #
            #     if downn:
            #         neurons = downn.split(" ")
            #         downn_neuron_types = self.get_neuron_types_with_bodyId(downn)
            #         downn_neuron_group = [neuron_type for neuron_type in downn_neuron_types if
            #                               neuron_type in downn_neuron_info]
            #     else:
            #         for neuron_type in downn_neuron_info:
            #             if neuron_type not in downn_neuron_group:
            #                 downn_neuron_group.append(neuron_type)
            #
            # upn_neuron_group = []
            # if query or upn:
            #     upn_neuron_info = self.get_neuron_info_all(upn, "", "", "", query, upn_w, ['id', 'type', 'roi'])[
            #         'n.type'].values.tolist()
            #     upn_neuron_group = []
            #     if upn:
            #         upn_neuron_types = self.get_neuron_types_with_bodyId(upn)
            #         upn_neuron_group = [neuron_type for neuron_type in upn_neuron_types if
            #                             neuron_type in upn_neuron_info]
            #     else:
            #         for neuron_type in upn_neuron_info:
            #             if neuron_type not in upn_neuron_group:
            #                 upn_neuron_group.append(neuron_type)
            #
                ####################################################
                # 假設query只會有一個條件，不然找交集太怪
                # 1. 找到可能的query neuron type
                # 2. 不去拆up或down
                # 3. 若down和up有多個~則各自找完再合併找交集
                # 4. 同個query subtype先找完和上游or下游的交集，最後data再合併輸出
                ####################################################
            # query,roi,result_info={},symbol='n.',file_name = "neuronInfo_
            Total_downn_result_list = []
            if downn:
                print("CHEKC3")
                condition, r_type, file_name = self.generate_condition_neuron_info(query, roi, {}, 'up.', 'Synapse_')
                downn_neuron_group = downn.split(" ")
                for query_id, sub_query in enumerate(query_group):
                    result_list = []
                    condition, r_type, tmp_file_name = self.generate_condition_neuron_info(sub_query, roi, {}, 'up.',
                                                                                           'Synapse_')
                    if query_id == 0:
                        file_name = file_name + "upstream_of_"
                    for subtype in downn_neuron_group:
                        if query_id == 0:
                            file_name = file_name + subtype + "_"
                        tmp_condition = condition + self.generate_condition_neuron(subtype, 'down.')
                        q = f"""{self.q_syn}{tmp_condition}{self.r_syn}"""
                        print("HERE2")
                        print(q)
                        results = self.c.fetch_custom(q)
                        if downn_w > 0:
                            results = self.filtered_by_weight(results, downn_w)
                        else:
                            results = results.values.tolist()
                        result_list.append(results)
                    if query_id == 0:
                        file_name = file_name + f"w_{downn_w}_"
                    intersection_data = self.get_result_intersection(result_list, first_loc_index=0)
                    Total_downn_result_list = Total_downn_result_list + intersection_data
                downn_results = Df(data=Total_downn_result_list, columns=raw_columns)
                final_downn_result = self.output_syn_xyz(copy.deepcopy(downn_results),f"{self.synapse_data_path}{file_name[:-1].replace('*', 's')}_{self.version}.xlsx")
            ###############################################################################################################

            Total_upn_result_list = []
            if upn:
                print("CHEKC4")
                condition, r_type, up_file_name = self.generate_condition_neuron_info(query, roi, {}, 'down.',
                                                                                      'Synapse_')
                if not downn:
                    file_name = copy.deepcopy(up_file_name)
                upn_neuron_group = upn.split(" ")
                for query_id, sub_query in enumerate(query_group):
                    result_list = []
                    condition, r_type, tmp_file_name = self.generate_condition_neuron_info(sub_query, roi, {}, 'down.',
                                                                                           'Synapse_')
                    if query_id == 0:
                        file_name = file_name + "downstream_of_"
                        up_file_name = up_file_name + "downstream_of_"
                    for subtype in upn_neuron_group:
                        if query_id == 0:
                            file_name = file_name + subtype + "_"
                            up_file_name = up_file_name + subtype + "_"
                        tmp_condition = condition + self.generate_condition_neuron(subtype, 'up.')
                        q = f"""{self.q_syn}{tmp_condition}{self.r_syn}"""
                        print("HERE3")
                        print(q)
                        results = self.c.fetch_custom(q)
                        print(results)
                        if upn_w > 0:
                            results = self.filtered_by_weight(results, upn_w)
                        else:
                            results = results.values.tolist()
                        result_list.append(results)
                    if query_id == 0:
                        file_name = file_name + f"w_{upn_w}_"
                        up_file_name = up_file_name + f"w_{upn_w}_"
                    intersection_data = self.get_result_intersection(result_list, first_loc_index=3)
                    Total_upn_result_list = Total_upn_result_list + intersection_data
                upn_results = Df(data=Total_upn_result_list, columns=raw_columns)
                final_upn_result = self.output_syn_xyz(copy.deepcopy(upn_results),
                               f"{self.synapse_data_path}{up_file_name[:-1].replace('*', 's')}_{self.version}.xlsx")
            final_results = Df(columns=columns)
            if downn and upn:
                final_results = self.get_result_intersection([Total_downn_result_list],0,[Total_upn_result_list],3)
                final_results = Df(data=final_results,columns=raw_columns)
                final_results = self.output_syn_xyz(copy.deepcopy(final_results),f"{self.synapse_data_path}{file_name[:-1].replace('*', 's')}_{self.version}.xlsx")
            elif downn:
                final_results = final_downn_result
            else:
                final_results = final_upn_result
            # self.syn_name_list.append(f"{file_name[:-1].replace('*', 's')}_{self.version}.xlsx")
        return final_results

    def generate_file_name_syn(self,upn,upn_w,downn,downn_w,query,roi):
        if not upn and not downn and not query:
            return
        else:
            if downn:
                condition, r_type, file_name = self.generate_condition_neuron_info(query, roi, {}, 'up.', 'Synapse_')
                downn_neuron_group = downn.split(" ")
                file_name = file_name + "upstream_of_"
                for subtype in downn_neuron_group:
                    file_name = file_name + subtype + "_"
                file_name = file_name + f"w_{downn_w}_"
            if upn:
                if not downn:
                    condition, r_type, file_name = self.generate_condition_neuron_info(query, roi, {}, 'down.',
                                                                                       'Synapse_')
                upn_neuron_group = upn.split(" ")
                file_name = file_name + "downstream_of_"
                for subtype in upn_neuron_group:
                    file_name = file_name + subtype + "_"
                file_name = file_name + f"w_{upn_w}_"
        return f"{self.synapse_data_path}{file_name[:-1].replace('*', 's')}_{self.version}.xlsx"

    def get_synapse_pooled(self,search_dict,upload_dict,fig):
        ##Connection_KCs_downstream_of_DM2s_w_0_v1.2.1.xlsx  (upstream_id,up_w, downstream_id,down_w,syn_query_neuron,syn_roi,syn_random)
        #### search #########
        file_list = []
        for search_para in search_dict:
            upn, upn_w, downn, downn_w, query, roi = copy.deepcopy(search_para).split(" ")
            upn_w = int(upn_w)
            downn_w = int(downn_w)
            if upn == "noinfo":
                upn = ""
            if downn == "noinfo":
                downn = ""
            if query == "noinfo":
                query = ""
            if roi == "noinfo":
                roi = ""
            file_name = self.generate_file_name_syn(upn,upn_w,downn,downn_w,query,roi)
            tmp_file_name = copy.deepcopy(file_name).split("/")[-1]
            if tmp_file_name not in self.syn_name_list:
                self.syn_name_list.append(tmp_file_name)
            if os.path.isfile(file_name):
                data = pd.read_excel(file_name)
            else:
                data = self.get_synapse_all(upn,upn_w,downn,downn_w,query,roi)

            x,y,z = np.array(data[ 'down_syn_coordinate_x'].values.tolist()),\
                    np.array(data[ 'down_syn_coordinate_y'].values.tolist()),np.array(data[ 'down_syn_coordinate_z'].values.tolist())
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                       mode='markers',
                                       name=file_name[file_name.find("a/Synapse")+2:file_name.find(".xlsx")],
                                       marker=dict(size=2, color=search_dict[search_para])))
        #### upload ##############
        for file_name in upload_dict:
            data = pd.read_excel(f"{self.synapse_data_path}{file_name}")
            x,y,z = np.array(data[ 'down_syn_coordinate_x'].values.tolist()),\
                    np.array(data[ 'down_syn_coordinate_y'].values.tolist()),np.array(data[ 'down_syn_coordinate_z'].values.tolist())
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                       mode='markers',
                                       name=file_name[file_name.find("a/Synapse")+2:file_name.find(".xlsx")],
                                       marker=dict(size=2, color=upload_dict[file_name])))
        return fig

    # connection = (upstream_id, up_w, downstream_id, down_w, syn_query_neuron, syn_roi,syn_rd)

    def get_result_union(self,result_list):
        neuron_list = []
        union_data = []
        for result in result_list:
            for neuron_data in result:
                if neuron_data[0] not in neuron_list:
                    union_data.append(neuron_data)
                    neuron_list.append(neuron_data[0])
        return union_data

    def divide_query_for_search(self):
        return

    def get_neuron_connection_all(self,query,roi,upn,upn_w,downn,down_w):
        columns = ['up.bodyId', 'up.instance', 'up.type', 'down.bodyId', 'down.instance', 'down.type', 'w.weight', 'w.roiInfo']
        '''
        幾種狀況:
        1. 若只有query，則分別取得query neurons的所有共同上游，以及共同下游
        2. 若有上游，則取得所有符合上游連結的query neuron，反之下游一樣
        3. 若同時有上游下游，則取得同時有連到這些上下游的神經元
        4. 若同時有上游下游+query，則同時要滿足神經只屬於query，且也要滿足同時連到這些上游及這些下游
        5. 只有上游+query，則同樣滿足這兩個的連結才取得


        '''
        if not upn and not downn and not roi and not query:
            return
        elif not upn and not downn:
            if not query:
                print("Currently no only roi!!!!. The data will too large to attain.")
                return
            if not down_w:
                down_w = 0
            condition, r_type, file_name = self.generate_condition_neuron_info(query, roi, {}, 'up.','Connection_')
            result_list = []
            group = downn.split(" ")
            file_name = file_name + "upstream_of_"
            for i in group:
                file_name = file_name + i + "_"
                tmp_condition = condition + self.generate_condition_neuron(i,'down.') + ' AND ' + f' w.weight > {down_w}'
                q = f"""{self.q_con}{tmp_condition}{self.r_con}"""
                print("HERE2")
                print(q)
                results = self.c.fetch_custom(q)
                result_list.append(copy.deepcopy(results).values.tolist())
            file_name = file_name + f"w_{down_w}_"
            intersection_data = self.get_result_intersection(result_list,first_loc_index=0)
            results = Df(data=intersection_data, columns=columns)
            results.to_excel(f"{self.connection_data_path}{file_name[:-1].replace('*', 's')}_{self.version}.xlsx")
            ###############################################################################################################
            if not upn_w:
                upn_w = 0
            condition, r_type, file_name = self.generate_condition_neuron_info(query, roi, {}, 'down.',
                                                                                   'Connection_')
            result_list = []
            group = upn.split(" ")
            file_name = file_name + "downstream_of_"
            for i in group:
                file_name = file_name + i + "_"
                tmp_condition = condition + self.generate_condition_neuron(i, 'up.') + ' AND ' + f' w.weight > {upn_w}'
                q = f"""{self.q_con}{tmp_condition}{self.r_con}"""
                print("HERE3")
                print(q)
                results = self.c.fetch_custom(q)
                result_list.append(copy.deepcopy(results).values.tolist())
            file_name = file_name + f"w_{upn_w}_"
            intersection_data = self.get_result_intersection(result_list,first_loc_index=3)
            results = Df(data=intersection_data, columns=columns)
            results.to_excel(f"{self.connection_data_path}{file_name[:-1].replace('*', 's')}_{self.version}.xlsx")
            self.conn_name_list.append(f"{file_name[:-1].replace('*', 's')}_{self.version}.xlsx")

        else:
            downn_result_list = []
            if downn:
                if not down_w:
                    down_w = 0
                condition, r_type, file_name = self.generate_condition_neuron_info(query, roi, {}, 'up.','Connection_')
                group = downn.split(" ")
                file_name = file_name + "upstream_of_"
                for i in group:
                    file_name = file_name + i + "_"
                    tmp_condition = condition + self.generate_condition_neuron(i,'down.') + ' AND ' + f' w.weight > {down_w}'
                    q = f"""{self.q_con}{tmp_condition}{self.r_con}"""
                    print("HERE2")
                    print(q)
                    results = self.c.fetch_custom(q)
                    # if not results:
                    #     return
                    downn_result_list.append(copy.deepcopy(results).values.tolist())
                file_name = file_name + f"w_{down_w}_"
            upn_result_list = []
            if upn:
                if not upn_w:
                    upn_w = 0
                condition, r_type, tmp_file_name = self.generate_condition_neuron_info(query, roi, {}, 'down.','Connection_')
                group = upn.split(" ")
                if not downn:
                    file_name = tmp_file_name
                file_name = file_name + "downstream_of_"
                for i in group:
                    file_name = file_name + i + "_"
                    tmp_condition = condition + self.generate_condition_neuron(i,'up.') + ' AND ' + f' w.weight > {upn_w}'
                    q = f"""{self.q_con}{tmp_condition}{self.r_con}"""
                    print("HERE3")
                    print(q)
                    results = self.c.fetch_custom(q)
                    # if not results:
                    #     return
                    upn_result_list.append(copy.deepcopy(results).values.tolist())
                file_name = file_name + f"w_{upn_w}_"
            if downn_result_list:
                intersection_data = self.get_result_intersection(result_list_1=downn_result_list,result_list_2=upn_result_list,first_loc_index=0,second_loc_index=3)
            else:
                intersection_data = self.get_result_intersection(upn_result_list,first_loc_index=3)
            results = Df(data=intersection_data, columns=columns)
            results.to_excel(f"{self.connection_data_path}{file_name[:-1].replace('*','s')}_{self.version}.xlsx")
            self.conn_name_list.append(f"{file_name[:-1].replace('*', 's')}_{self.version}.xlsx")

        return results

    def get_neuron_info_all(self,query,roi,upn,upn_w,downn,down_w, result_info):
        if not upn and not downn and not roi and not query:
            return
        elif not upn and not downn:
            condition, r_type, file_name = self.generate_condition_neuron_info(query,roi,result_info,'n.','neuronInfo_')
            q = f"""{self.q_type}{condition[:-5]}{r_type}"""
            print(q)
            print("HERE")
            try:
                results = self.c.fetch_custom(q)
            except:
                print(
                    "This query fails. The reason may result from bad query request or the neuprint restriction. You can try it again.")
                return
                pass
            results.to_excel(f"{self.other_basic_info_path}{file_name[:-1].replace('*','s')}_{self.version}.xlsx")
        else:
            columns = []
            if 'id' in result_info:
                columns.append('n.bodyId')
            if 'type' in result_info:
                columns.append('n.type')
                columns.append('n.instance')
            if 'roi' in result_info:
                columns.append("n.roiInfo ")
            result_list = []

            if downn:
                if not down_w:
                    down_w = 0
                condition, r_type, file_name = self.generate_condition_neuron_info(query, roi, result_info, 'up.','neuronInfo_')
                group = downn.split(" ")
                file_name = file_name + "upstream_of_"
                for i in group:
                    file_name = file_name + i + "_"
                    tmp_condition = condition + self.generate_condition_neuron(i,'down.') + ' AND ' + f' w.weight > {down_w}'
                    q = f"""{self.q_con}{tmp_condition}{r_type}"""
                    print("HERE2")
                    print(q)
                    results = self.c.fetch_custom(q)
                    # if not results:
                    #     return
                    result_list.append(copy.deepcopy(results).values.tolist())
                file_name = file_name + f"w_{down_w}_"

            if upn:
                if not upn_w:
                    upn_w = 0
                condition, r_type, tmp_file_name = self.generate_condition_neuron_info(query, roi, result_info, 'down.','neuronInfo_')
                group = upn.split(" ")
                if not downn:
                    file_name = tmp_file_name
                file_name = file_name + "downstream_of_"
                for i in group:
                    file_name = file_name + i + "_"
                    tmp_condition = condition + self.generate_condition_neuron(i,'up.') + ' AND ' + f' w.weight > {upn_w}'
                    q = f"""{self.q_con}{tmp_condition}{r_type}"""
                    print("HERE3")
                    print(q)
                    print("WHAT_HAPPEND",i)
                    results = self.c.fetch_custom(q)
                    # if not results:
                    #     return
                    result_list.append(copy.deepcopy(results).values.tolist())
                print(result_list)
                file_name = file_name + f"w_{upn_w}_"
            intersection_data = self.get_result_intersection(result_list,3)
            print('finish in',intersection_data)
            results = Df(data=intersection_data, columns=columns)
            results.to_excel(f"{self.other_basic_info_path}{file_name[:-1].replace('*','s')}_{self.version}.xlsx")
        self.info_name_list.append(f"{file_name[:-1].replace('*','s')}_{self.version}.xlsx")
        return results

    def get_neuron_info(self,line):
        if len(line) == 0:
            return
        if (line.find("\n")) != -1:
            line = line[:-1]
        group = line.split(" ")
        for i in group:
            condition = self.generate_condition_single(i)
            # q = qn + condition + rn
            q = f"""{self.qn}{condition}{self.rn}"""
            try:
                results = self.c.fetch_custom(q)
            except:
                print(
                    "This query fails. The reason may result from bad query request or the neuprint restriction. You can try it again.")
                break
            results.to_excel(f"{self.other_basic_info_path}{str(forsave(i))}_info_{version}.xlsx")
            print(f"{str(forsave(i))}_info_{version}Finished\n")
        return

    def get_neuron_info_simple(self,line):
        qn = "MATCH (n:Neuron)"
        rn = " RETURN DISTINCT n.instance, n.type"

        if len(line) == 0:
            return
        if (line.find("\n")) != -1:
            line = line[:-1]
        group = line.split(" ")
        for i in group:
            condition = self.generate_condition_single(i)
            # q = qn + condition + rn
            q = f"""{qn}{condition}{rn}"""
            try:
                results = self.c.fetch_custom(q)
            except:
                print(
                    "This query fails. The reason may result from bad query request or the neuprint restriction. You can try it again.")
                break
            results.to_excel(f"{self.other_basic_info_path}{str(forsave(i))}_simple_info_{self.version}.xlsx")
            print(f"{str(forsave(i))}_simple_info_{self.version} Finished\n")
        return

    def get_neuron_types(self,name):
        condition = self.generate_condition_single(name)
        q_type = "MATCH (n:Neuron)"
        r_type = " RETURN DISTINCT n.type"
        q = f"""{q_type}{condition}{r_type}"""
        result = self.c.fetch_custom(q)
        return result['n.type']

    def get_neuron_types_with_bodyId(self,name):
        condition = self.generate_condition_single(name)
        q_type = "MATCH (n:Neuron)"
        r_type = " RETURN DISTINCT n.bodyId, n.type"
        q = f"""{q_type}{condition}{r_type}"""
        result = self.c.fetch_custom(q)
        return result


    def get_synapse(self, upstream_id, downstream_id):
        connection_type = "synapse"
        q = self.q_syn
        r = self.r_syn
        if len(upstream_id) != 0 and len(downstream_id) != 0:
            up_syn, down_syn = self.pair_neuron(q, r, connection_type, upstream_id, downstream_id)
        elif len(downstream_id) != 0:
            up_syn, down_syn = self.single_upstream(q, r, connection_type, downstream_id)
        elif len(upstream_id) != 0:
            up_syn, down_syn = self.single_downstream(q, r, connection_type, upstream_id)
        else:
            up_syn, down_syn = [[], [], []], [[], [], []]
        return up_syn, down_syn

    #################################################
    def single_upstream(self,qu, ru, query_type, downstream_id):
        condition = self.generate_condition_upstream(downstream_id)
        # q = qu + condition + ru
        q = f"""{qu}{condition}{ru}"""

        try:
            results = self.c.fetch_custom(q)
        except:
            print(
                "This query fails. The reason may result from bad query request or the neuprint restriction. You can try it again.")
            return [[], [], []], [[], [], []]
        for syn_loc in ['up', 'down']:
            tmp_list = results[syn_loc + '_syn.location'].tolist()
            X_collect = []
            Y_collect = []
            Z_collect = []
            for j in range(len(tmp_list)):
                X_collect.append(tmp_list[j]['coordinates'][0])
                Y_collect.append(tmp_list[j]['coordinates'][1])
                Z_collect.append(tmp_list[j]['coordinates'][2])
            results[syn_loc + '_syn_coordinate_x'] = X_collect
            results[syn_loc + '_syn_coordinate_y'] = Y_collect
            results[syn_loc + '_syn_coordinate_z'] = Z_collect
        results = results.drop(columns=['up_syn.location', 'down_syn.location'])
        results.to_excel(
            f"{self.synapse_data_path}upstream of {str(forsave(upstream_id))}_{query_type}_{self.version}.xlsx")
        print("upstream of " + str(forsave(downstream_id)) + " Finished\n")
        return [results['up_syn_coordinate_x'].values.tolist(), results['up_syn_coordinate_y'].values.tolist(),
                results['up_syn_coordinate_z'].values.tolist()], [results['down_syn_coordinate_x'].values.tolist(),
                                                                  results['down_syn_coordinate_y'].values.tolist(),
                                                                  results['down_syn_coordinate_z'].values.tolist()]

    def output_syn_xyz(self,results,file_name):
        for syn_loc in ['up', 'down']:
            tmp_list = results[syn_loc + '_syn.location'].tolist()
            X_collect = []
            Y_collect = []
            Z_collect = []
            for j in range(len(tmp_list)):
                X_collect.append(tmp_list[j]['coordinates'][0])
                Y_collect.append(tmp_list[j]['coordinates'][1])
                Z_collect.append(tmp_list[j]['coordinates'][2])
            results[syn_loc + '_syn_coordinate_x'] = X_collect
            results[syn_loc + '_syn_coordinate_y'] = Y_collect
            results[syn_loc + '_syn_coordinate_z'] = Z_collect
        results = results.drop(columns=['up_syn.location', 'down_syn.location'])
        results.to_excel(f"{file_name}")
        return results


    def single_downstream(self, qd, rd, query_type, upstream_id):
        condition = self.generate_condition_downstream(upstream_id)
        # q = qd + condition + rd
        q = f"""{qd}{condition}{rd}"""
        try:
            results = self.c.fetch_custom(q)
        except:
            print(
                "This query fails. The reason may result from bad query request or the neuprint restriction. You can try it again.")
            return [[], [], []], [[], [], []]
        for syn_loc in ['up', 'down']:
            tmp_list = results[syn_loc + '_syn.location'].tolist()
            X_collect = []
            Y_collect = []
            Z_collect = []
            for j in range(len(tmp_list)):
                X_collect.append(tmp_list[j]['coordinates'][0])
                Y_collect.append(tmp_list[j]['coordinates'][1])
                Z_collect.append(tmp_list[j]['coordinates'][2])
            results[syn_loc + '_syn_coordinate_x'] = X_collect
            results[syn_loc + '_syn_coordinate_y'] = Y_collect
            results[syn_loc + '_syn_coordinate_z'] = Z_collect
        results = results.drop(columns=['up_syn.location', 'down_syn.location'])
        results.to_excel(f"{self.synapse_data_path}downstream of {str(forsave(upstream_id))}_{query_type}_{self.version}.xlsx")
        print("downstream of " + str(forsave(upstream_id)) + " Finished\n")
        return [results['up_syn_coordinate_x'].values.tolist(), results['up_syn_coordinate_y'].values.tolist(),
                results['up_syn_coordinate_z'].values.tolist()], [results['down_syn_coordinate_x'].values.tolist(),
                                                                  results['down_syn_coordinate_y'].values.tolist(),
                                                                  results['down_syn_coordinate_z'].values.tolist()]

    def pair_neuron(self, qp, rp, query_type, upstream_id, downstream_id):
        count = 0
        group = [upstream_id, downstream_id]
        if group[0].find("*") != -1:
            pretypes = self.get_neuron_types(group[0])
        else:
            pretypes = [group[0]]
        if group[1].find("*") != -1:
            postypes = self.get_neuron_types(group[1])
        else:
            postypes = [group[1]]
        pre_length = 0
        # pre_list=[]
        for pre in pretypes:
            # if pre_length<20:
            #     pre_length=pre_length+1
            #     pre_list.append(pre)
            #     continue
            for post in postypes:
                # pre = pre_list
                condition = self.generate_condition_pair([pre, post])
                # q = qp + condition + rp
                q = f"""{qp}{condition}{rp}"""
                # print(rp)
                if count == 0:
                    results = self.c.fetch_custom(q)
                    count = 1
                    for syn_loc in ['up', 'down']:
                        tmp_list = results[syn_loc + '_syn.location'].tolist()
                        X_collect = []
                        Y_collect = []
                        Z_collect = []
                        for i in range(len(tmp_list)):
                            X_collect.append(tmp_list[i]['coordinates'][0])
                            Y_collect.append(tmp_list[i]['coordinates'][1])
                            Z_collect.append(tmp_list[i]['coordinates'][2])
                        results[syn_loc + '_syn_coordinate_x'] = X_collect
                        results[syn_loc + '_syn_coordinate_y'] = Y_collect
                        results[syn_loc + '_syn_coordinate_z'] = Z_collect
                    results = results.drop(columns=['up_syn.location', 'down_syn.location'])
                else:
                    print("start_get_" + pre + " " + post)
                    tmp_result = self.c.fetch_custom(q)
                    print("over")
                    for syn_loc in ['up', 'down']:
                        tmp_list = tmp_result[syn_loc + '_syn.location'].tolist()
                        X_collect = []
                        Y_collect = []
                        Z_collect = []
                        for i in range(len(tmp_list)):
                            X_collect.append(tmp_list[i]['coordinates'][0])
                            Y_collect.append(tmp_list[i]['coordinates'][1])
                            Z_collect.append(tmp_list[i]['coordinates'][2])
                        tmp_result[syn_loc + '_syn_coordinate_x'] = X_collect
                        tmp_result[syn_loc + '_syn_coordinate_y'] = Y_collect
                        tmp_result[syn_loc + '_syn_coordinate_z'] = Z_collect
                    tmp_result = tmp_result.drop(columns=['up_syn.location', 'down_syn.location'])
                    results = pd.merge(results, tmp_result, how='outer')
            pre_length += 1
            if pre_length > 10:
                results.to_excel("tmp_output.xlsx")
                pre_length = 0
        pre_name = forsave(group[0])
        post_name = forsave(group[1])
        results.to_excel(f"{self.synapse_data_path}{pre_name}_to_{post_name}_{query_type}_{self.version}.xlsx")
        print(pre_name + "_to_" + post_name + " Finished\n")
        return [results['up_syn_coordinate_x'].values.tolist(), results['up_syn_coordinate_y'].values.tolist(),
                results['up_syn_coordinate_z'].values.tolist()], [results['down_syn_coordinate_x'].values.tolist(),
                                                                  results['down_syn_coordinate_y'].values.tolist(),
                                                                  results['down_syn_coordinate_z'].values.tolist()]

    ##########extract synpase






class Template_brain(CoordinateCollection):
    '''
    main program to deal with the data in the template brain
    1. initialize all data path
    2. load brain coordinates, neuron, neuropil
    3. interact with image processing tool for warping
    4. interact with data analysis
    5.
    '''
    def __init__(self, template='FlyEM', offline=False):
        self.neuprint_tool = neuprint_query_tool(template=template)
        self.brain_template_warping_tool = eFlyA.brain_template_warping_tool()
        print(self.neuprint_tool.version)
        self.xyz_collection = {}
        self.swc_parent_collection = {}
        self.template = template
        self.get_neuropil_path()
        self.initialize_path()
        self.load_neuropil()
        self.neuropil_history = []
        self.dendro = eFlyA.generate_synaptic_dendrogram()

        if offline == False:
            self.neuprint_tool.connecting_neuprint_server()
        if os.path.isfile(f"{self.tmp_path}conn_tmp.pickle"):
            os.remove(f"{self.tmp_path}conn_tmp.pickle")

        return

    ###注意all_neuropil_mesh的問題###############
    def initialize_path(self):
        #### precomputed data ################
        self.precomputed_data_path = "Precomputed_data/"
        self.precomputed_skeleton_path = f'{self.precomputed_data_path}Dash_{self.template}_skeleton/'
        self.precomputed_xyz_path = f'{self.precomputed_data_path}Dash_{self.template}_xyz/'
        self.precomputed_neuropil_path = f'{self.precomputed_data_path}Dash_{self.template}_neuropil/'

        if not os.path.isdir(self.precomputed_data_path):
            os.mkdir(self.precomputed_data_path)
        if not os.path.isdir(self.precomputed_skeleton_path):
            os.mkdir(self.precomputed_skeleton_path)
        if not os.path.isdir(self.precomputed_xyz_path):
            os.mkdir(self.precomputed_xyz_path)
        if not os.path.isdir(self.precomputed_neuropil_path):
            os.mkdir(self.precomputed_neuropil_path)

        #### DATA ###############
        self.data_path = 'Data/'
        self.xyz_path = f"{self.data_path}{self.template}_xyz/"
        self.skeleton_path = f"{self.data_path}{self.template}_skeleton/"
        self.neuropil_path = f"{self.data_path}{self.template}_neuropil/"
        self.all_neuropil_mesh = self.obtain_neuropil_mesh() ##################################################
        self.warping_xyz_path = f"{self.data_path}xyz/"

        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.isdir(self.xyz_path):
            os.mkdir(self.xyz_path)
        if not os.path.isdir(self.skeleton_path):
            os.mkdir(self.skeleton_path)
        if not os.path.isdir(self.neuropil_path):
            os.mkdir(self.neuropil_path)


        #### tmp_folder ############
        self.tmp_path = 'Tmp/'

        if not os.path.isdir(self.tmp_path):
            os.mkdir(self.tmp_path)
        if os.path.isfile(self.tmp_path + "search_button_click_record.pickle"):
            os.remove(self.tmp_path + "search_button_click_record.pickle")


        #### output ##########################
        self.output_path = 'output/'
        self.connection_data_path = f'{self.output_path}Connection_data/'
        self.synapse_data_path = f'{self.output_path}Synapse_data/'
        self.other_basic_info_path = f'{self.output_path}Other_basic_info/'
        self.analysis_result_path = f'{self.output_path}Analysis_result/'
        self.other_path = f"{self.output_path}Others/"

        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        if not os.path.isdir(self.connection_data_path):
            os.mkdir(self.connection_data_path)
        if not os.path.isdir(self.synapse_data_path):
            os.mkdir(self.synapse_data_path)
        if not os.path.isdir(self.other_basic_info_path):
            os.mkdir(self.other_basic_info_path)
        if not os.path.isdir(self.analysis_result_path):
            os.mkdir(self.analysis_result_path)
        if not os.path.isdir(self.other_path):
            os.mkdir(self.other_path)

    def load_neuropil(self):
        self.neuropil = []
        self.neuropil_values = []
        for neuropil in os.listdir(self.neuropil_path):
            self.neuropil.append({'label': neuropil[:-4], 'value': neuropil[:-4]})
            self.neuropil_values.append(neuropil[:-4])
        return self.neuropil, self.neuropil_values

    def get_neuropil_path(self):
        if self.template == 'FlyEM':
            self.neuropil_path = "EM_neuropil/"
        elif self.template == 'FlyCircuit':
            self.neuropil_path = ""
        else:
            self.neuropil_path = input("Please specify your path of neuropil template.\nFlyEM or FlyCircuit")
        return self.neuropil_path

    def obtain_neuropil_mesh(self): ##是否要旋轉，需再測試，另opacity也需要再確認
        if os.path.isfile(self.precomputed_neuropil_path + 'neuro_pil_mesh.pickle'):
            with open(self.precomputed_neuropil_path + 'neuro_pil_mesh.pickle', 'rb')as ff:
                all_neuropil_mesh = pickle.load(ff)
            return all_neuropil_mesh
        print(f"First time to use {self.template}. Initializing for neuropil loading.")
        neuropil_list = os.listdir(self.neuropil_path)
        all_neuropil_mesh = {}
        for name in neuropil_list:
            obj_data = open(self.neuropil_path + name, "rt")
            vertices, faces = self.obj_data_to_mesh3d(obj_data)
            x, y, z = vertices[:, :3].T
            # rotation_data_neuropil = eFlyplot_image_processing.rotation([[x[i], y[i], z[i]] for i in range(len(x))], 3.14, 'xz')
            # x, y, z = rotation_data_neuropil[:, 0], rotation_data_neuropil[:, 1], rotation_data_neuropil[:, 2]
            I, J, K = faces.T
            mesh = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                # vertexcolor=vertices[:, 3:],  # the color codes must be triplets of floats  in [0,1]!!
                i=I,
                j=J,
                k=K,
                name=name,
                showscale=False,
                opacity=0.4,
                color='rgb(' + str(rd.randint(0, 255)) + ',' + str(rd.randint(0, 255)) + ',' + str(
                    rd.randint(0, 255)) + ')'
            )
            all_neuropil_mesh[name[:-4]] = mesh
        with open(self.precomputed_neuropil_path + 'neuro_pil_mesh.pickle', 'wb')as ff:
            pickle.dump(all_neuropil_mesh, ff)
        return all_neuropil_mesh

    def obj_data_to_mesh3d(self,odata):  ##可能要再重刻
        vertices = []
        faces = []
        lines = odata
        for line in lines:
            slist = line.split()
            if slist:
                if slist[0] == 'v':
                    vertex = np.array(slist[1:], dtype=float)
                    vertices.append(vertex)
                elif slist[0] == 'f':
                    face = []
                    for k in range(1, len(slist)):
                        face.append([int(s) for s in slist[k].replace('//', '/').split('/')])
                    if len(face) > 3:  # triangulate the n-polyonal face, n>3
                        faces.extend(
                            [[face[0][0] - 1, face[k][0] - 1, face[k + 1][0] - 1] for k in range(1, len(face) - 1)])
                    else:
                        faces.append([face[j][0] - 1 for j in range(len(face))])
                else:
                    pass
        return np.array(vertices), np.array(faces)

    def get_neuropil(self, names, select_color):
        '''
        visualize neuropil.
        if the neuropil has been visualized before, we will read the history to visualize the neuropil
        :param names: the neuropil selected
        :param select_color:  the color of selected neuropil
        :return:
        '''
        select_color = select_color['rgb']
        r, g, b, a = select_color['r'], select_color['g'], select_color['b'], select_color['a']
        if r == 255 and g == 255 and b == 255:
            select_color = f'rgb({rd.randint(0, 255)},{rd.randint(0, 255)},{rd.randint(0, 255)})'
        else:
            select_color = f'rgb({r},{g},{b},{a})'
        neuropil_history = []
        neuropil_color = []
        if os.path.isfile(self.tmp_path + "neuropil_history.pickle") and len(neuropil_history)==0:
            with open(self.tmp_path + "neuropil_history.pickle", "rb")as ff:
                neuropil_history = pickle.load(ff)
            with open(self.tmp_path + "neuropil_color.pickle", "rb")as ff:
                neuropil_color = pickle.load(ff)
        mesh_list = []
        for name in names:
            if name not in neuropil_history:
                mesh = all_neuropil_mesh[name]
                tmp_color = select_color
                mesh.color = tmp_color
                mesh_list.append(mesh)
                neuropil_color.append(tmp_color)
                neuropil_history.append(name)
            else:
                mesh = self.all_neuropil_mesh[name]
                mesh.color = neuropil_color[neuropil_history.index(name)]
                mesh_list.append(mesh)
        with open(self.tmp_path + "neuropil_history.pickle", "wb")as ff:
            pickle.dump(neuropil_history, ff)
        with open(self.tmp_path + "neuropil_color.pickle", "wb")as ff:
            pickle.dump(neuropil_color, ff)
        self.neuropil_history = neuropil_history
        self.neuropil_color = neuropil_color
        self.mesh_list = mesh_list
        return mesh_list

    def get_xyz(self, xyz_files, fig):
        #### draw neuron
        xyz_file_list = os.listdir(self.precomputed_xyz_path)
        for xyz_file in xyz_files:
            select_color = xyz_files[xyz_file]
            if xyz_file[:-4] + ".pickle" not in xyz_file_list:
                Node_xyz = self.read_xyz(self.xyz_path + xyz_file)
                # print(Node_xyz[0], Node_xyz[-1])
                # rotation_data_neuron = rotation(Node_xyz, 3.14, 'xz')
                with open(self.precomputed_xyz_path + xyz_file[:-4] + ".pickle", "wb")as ff:
                    pickle.dump(Node_xyz, ff)
            else:
                with open(self.precomputed_xyz_path + xyz_file[:-4] + ".pickle", "rb")as ff:
                    Node_xyz = pickle.load(ff)
            Node_xyz = np.array(Node_xyz)
            # x, y, z = rotation_data_neuron[:, 0], rotation_data_neuron[:, 1], rotation_data_neuron[:, 2]
            x, y, z = Node_xyz[:, 0], Node_xyz[:, 1], Node_xyz[:, 2]
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                       mode='markers',
                                       name=xyz_file[:-4],
                                       marker=dict(size=5, color=select_color)))
        else:
            pass
        return fig

    def get_synapse(self, upstream_id, downstream_id, fig, select_color):
        points_xyz_file_skeleton_list = os.listdir(self.synapse_data_path)
        original_up_id = upstream_id
        original_down_id = downstream_id
        if len(upstream_id) != 0 and len(downstream_id) != 0:
            if upstream_id[0] == '*':
                upstream_id = "s_" + upstream_id[1:]
            if upstream_id[-1] == '*':
                upstream_id = upstream_id[:-1] + "_s"
            if downstream_id[0] == '*':
                downstream_id = "s_" + downstream_id[1:]
            if downstream_id[-1] == '*':
                downstream_id = downstream_id[:-1] + "_s"
            points_xyz_file = f"{upstream_id}_to_{downstream_id}_synapse.txt"
        elif len(downstream_id) != 0:
            if downstream_id[0] == '*':
                downstream_id = "s_" + downstream_id[1:]
            if downstream_id[-1] == '*':
                downstream_id = downstream_id[:-1] + "_s"
            points_xyz_file = f"upstream_neurons_to_{downstream_id}_synapse.txt"
        elif len(upstream_id) != 0:
            if upstream_id[0] == '*':
                upstream_id = "s_" + upstream_id[1:]
            if upstream_id[-1] == '*':
                upstream_id = upstream_id[:-1] + "_s"
            points_xyz_file = f"downstream_neurons_from_{upstream_id}_synapse.txt"
        else: ##no input but press the button
            return fig

        # First, check the existence of pickle
        if points_xyz_file[:-4] + ".pickle" not in points_xyz_file_skeleton_list:
            # Second, check the existence of .xlsx
            if not os.path.isfile("output/" + points_xyz_file[:-4] + ".xlsx"):
                up_syn, down_syn = self.neuprint_tool.get_synapse(original_up_id, original_down_id)
            else:
                data = pd.read_excel("output/" + points_xyz_file[:-4] + ".xlsx")
                up_syn, down_syn = [data['up_syn_coordinate_x'].values.tolist(),
                                    data['up_syn_coordinate_y'].values.tolist(),
                                    data['up_syn_coordinate_z'].values.tolist()], [
                                       data['down_syn_coordinate_x'].values.tolist(),
                                       data['down_syn_coordinate_y'].values.tolist(),
                                       data['down_syn_coordinate_z'].values.tolist()]
            if len(down_syn) == 0:
                return fig
            elif len(down_syn[0]) == 0:
                return fig
            Node_xyz = np.array(down_syn).transpose().tolist()
            rotation_data_neuron = rotation(Node_xyz, 0, 'xz')
            with open(neuron_swc_path + points_xyz_file[:-4] + ".pickle", "wb")as ff:
                pickle.dump(rotation_data_neuron, ff)
        else:
            with open(neuron_swc_path + points_xyz_file[:-4] + ".pickle", "rb")as ff:
                rotation_data_neuron = pickle.load(ff)
        ###這邊先直接預設看下游的
        x, y, z = rotation_data_neuron[:, 0], rotation_data_neuron[:, 1], rotation_data_neuron[:, 2]
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                   mode='markers',
                                   name=points_xyz_file[:-4],
                                   marker=dict(size=5, color=select_color)))
        return fig

    def get_neuron(self, neuron_names_dict, fig):
        #### draw neuron
        neuron_list = []
        color_list = []
        for neuron_name in neuron_names_dict:
            select_color = neuron_names_dict[neuron_name]
            if os.path.isfile(f"{self.skeleton_path}{neuron_name}"):
                neuron_list.append(neuron_name)
                color_list.append(select_color)
        neuron_skeleton_list = os.listdir(self.precomputed_skeleton_path)
        for neuron_file, select_color in zip(neuron_list, color_list):
            if neuron_file[:-4] + ".pickle" not in neuron_skeleton_list:
                Node_xyz, node_parent = self.read_swc(f"{self.skeleton_path}{neuron_file}")
                # Node_xyz = rotation(Node_xyz, 3.14, 'xz')
                Node_xyz = np.array(Node_xyz)
                with open(self.precomputed_skeleton_path + neuron_file[:-4] + ".pickle", "wb")as ff:
                    pickle.dump(Node_xyz, ff)
            else:
                with open(self.precomputed_skeleton_path + neuron_file[:-4] + ".pickle", "rb")as ff:
                    Node_xyz = pickle.load(ff)
            x, y, z = Node_xyz[:, 0], Node_xyz[:, 1], Node_xyz[:, 2]
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                       mode='markers',
                                       name=neuron_file,
                                       marker=dict(size=2, color=select_color)))
        else:
            pass
        return fig

class eFlyPlot_environment:
    '''
    eFlyplot:
    1. construct all component
    2. draw picture, plot
    3. interact with brain space
    4. interact with neuprint survey system

    '''
    def __init__(self,brain_template="FlyEM"):
        print("Initializing eFlyPlot......\n")
        self.brain_space = Template_brain(template=brain_template)
        # self.initialize_path()
        # self.tmp_path = "tmp_eFlyPlot/"
        # self.neuron_swc_path = 'dash_eFlyplot_neuron/'
        # self.neuropil_mesh_path = 'dash_eflyplot_neuropil/'
        # self.xyz_path = 'dash_eflyplot_xyz/'
    def create_environment(self):
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        app.layout = html.Div([
            dcc.Tabs([
                eFlyPlot_layout.get_tab_for_eFlyPlot_info(),
                eFlyPlot_layout.get_tab_for_basic_info(),
                eFlyPlot_layout.get_tab_for_neuron_connection(),
                test_cyto.get_tab_cytoscape_layout(),
                eFlyPlot_layout.get_tab_for_neuron_synapses(neuropil_options=list(self.brain_space.all_neuropil_mesh.keys())),
                eFlyPlot_layout.get_tab_for_synapse_analysis()
            ])
        ])
        return app

    def get_layout_of_3dObject(self):
        layout = go.Layout(
            title="eFlyPlot",
            scene=plotly.graph_objs.layout.Scene(bgcolor='rgba(0,0,0,0)',  # Sets background color to white
                                                 xaxis=plotly.graph_objs.layout.scene.XAxis(showgrid=False,
                                                                                            zeroline=False,backgroundcolor="rgba(0,0,0,0)"),
                                                 yaxis=plotly.graph_objs.layout.scene.YAxis(showgrid=False,
                                                                                            zeroline=False,backgroundcolor="rgba(0,0,0,0)"),
                                                 zaxis=plotly.graph_objs.layout.scene.ZAxis(showgrid=False,
                                                                                            zeroline=False,backgroundcolor="rgba(0,0,0,0)"),
                                                 ),
            # plot_bgcolor='white',
            # paper_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            width=1500,
            height=1000,
            scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.25, y=1.25, z=1.25))

        )
        return layout

    def draw_new_figure(self):
        return go.Figure(data=self.brain_space.mesh_list, layout=self.get_layout_of_3dObject())

    def add_neuropil(self,neuropil, fig, select_color):
        self.brain_space.all_neuropil_mesh[neuropil]['color'] = self.get_select_color(select_color)
        fig.add_trace(self.brain_space.all_neuropil_mesh[neuropil])
        return fig


    def get_select_color(self, select_color):
        original_select_color = copy.deepcopy(select_color)
        select_color = select_color[4:-1].split(",")
        r, g, b = int(select_color[0]), int(select_color[1]), int(select_color[2])
        # select_color = f'rgb({r},{g},{b},{a})'
        if r == 255 and g == 255 and b == 255: ##not select any color
            select_color = f'rgb({rd.randint(0, 255)},{rd.randint(0, 255)},{rd.randint(0, 255)})'
        else:
            select_color = original_select_color
        return select_color




# def check_new_input(names,neuron_ids,other_file,button_press,select_color,upstream_id,downstream_id,history_data):
#
#     'neuropil': [], 'neuron_input': [], 'synapse': [], 'xyz': [], 'neuropil_color': [],
#     'neuron_color': [], 'synapse_color': [], 'xyz_color': []
#
#     return