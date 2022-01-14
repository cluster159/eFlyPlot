from neuprint import Client
# import networkx as nx
import math
import random
from pandas import DataFrame as Df
# from matplotlib import pyplot as plt
import os
import json
import numpy as np
import re
import pandas as pd
import time as tt
# from neuprint.tests import NEUPRINT_SERVER, DATASET
Server="emdata1.int.janelia.org:11000"
print("******************The program is excutable, now.******************************")
print("The current version of data is v1.2")
print("The data is provided by Janelia. To get further information, please visit: https://neuprint.janelia.org/")
print("This program for FlyEM data acuisition from neuprint is developed by Ching-Che Charng (Jerry).\n Current version is beta version.\nIf you encounter any problem, please contact: cluster159@gmail.com\n")

# Token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImNoYXJuZ2NoaW5nY2hlQGxvbGFiLW50aHUub3JnIiwibGV2ZWwiOiJub2F1dGgiLCJpbWFnZS11cmwiOiJodHRwczovL2xoNC5nb29nbGV1c2VyY29udGVudC5jb20vLTNQdU9Xb0RVQUxNL0FBQUFBQUFBQUFJL0FBQUFBQUFBQUFBL0FDSGkzcmRFZ2NhSDJlVzNSbVJfc2ZuM1F2RUNVdWRBQlEvcGhvdG8uanBnP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzYwODgwNDE4fQ.haDl_9vFGuJRxP8UAJIKaHocYtPYwhIcrrotq4VJSpQ"

token_file_name="token_for_check_connection.txt"
if not os.path.isfile(token_file_name):
    check_token=0
    while check_token==0:
        Token = input(
            "The token is not correct. Please type it again."
        )
        if len(Token)>10:
            try:
                c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2', token=Token)
                with open(token_file_name, "wt")as ff:
                    ff.writelines(Token)
                check_token=1
            except:
                pass
else:
    with open(token_file_name,"rt")as ff:
        for line in ff:
            if line.find("\n")!=-1:
                line=line[:-1]
            Token=line
            break
c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2', token=Token)
c.fetch_version()



def lookup(line):
    if re.search('[a-zA-Z]', line) != None:
        condition = "type="
        if line.find("*") != -1:
            condition=condition+"~"
        condition=condition+"\""
        if line[0].find("*")!=-1:
            condition=condition+"."
        if line[-1].find("*")!=-1:
            condition=condition+line[:-1]+".*\""
        else:
            condition=condition+line+"\""
        return condition
    else:
        condition="bodyId="+line
        return condition
def is_type(line):
    if line.find("*")!=-1:
        return 2
    if re.search('[a-zA-Z]', line) != None:
        return 1
    else:
        return 0


def generate_condition_pair(group):
    pre, post=group
    condition_pre=lookup(pre)
    condition_post=lookup(post)
    return " WHERE up."+ condition_pre + " AND down." +condition_post

def generate_condition_upstream(neuron):
    condition=lookup(neuron)
    return " WHERE down." +condition

def generate_condition_downstream(neuron):
    condition=lookup(neuron)
    return " WHERE up." +condition


def forsave(line):
    if line[0].find("*") != -1:
        if line[-1].find("*")!=-1:
            return line[1:-1]
        else:
            return line[1:]
    else:
        if line[-1].find("*")!=-1:
            return line[:-1]
        else:
            return line
def generate_condition_single(neuron):
    condition = lookup(neuron)
    return "WHERE n."+condition

def generate_condition_roi(roi):
    return f"n.`{roi}`"



def get_neuron_in_roi(line):
    qn = "MATCH (n:Neuron)"
    rn = " RETURN DISTINCT n.bodyId, n.instance, n.type, n.roiInfo"
    if len(line) == 0:
        return
    if (line.find("\n")) != -1:
        line = line[:-1]
    file_name = ""
    group = line.split(" ")
    condition = ' WHERE '
    for i in group:
        file_name = file_name + i +"_"
        condition = condition + generate_condition_roi(i) + ' AND '
        # q = qn + condition + rn
    condition = condition[:-5]
    q = f"""{qn}{condition}{rn}"""
    print("check roi query", q)
    try:
        results = c.fetch_custom(q)
    except:
        print(
            "This query fails. The reason may result from bad query request or the neuprint restriction. You can try it again.")
        pass
    results.to_excel("output/neurons_in_" + str(forsave(file_name[:-1])) + "_info" + ".xlsx")
    print(str(forsave(file_name[:-1])) + " Finished\n")
    return

def get_neuron_info(line):
    qn = "MATCH (n:Neuron)"
    rn = " RETURN DISTINCT n.bodyId, n.instance, n.type, n.roiInfo"
    if len(line) == 0:
        return
    if (line.find("\n")) != -1:
        line = line[:-1]
    group = line.split(" ")
    for i in group:
        condition = generate_condition_single(i)
        # q = qn + condition + rn
        q = f"""{qn}{condition}{rn}"""
        try:
            results = c.fetch_custom(q)
        except:
            print(
                "This query fails. The reason may result from bad query request or the neuprint restriction. You can try it again.")
            break
        results.to_excel("output/" + str(forsave(i)) + "_info" + ".xlsx")
        print(str(forsave(i)) + " Finished\n")
    return


def get_neuron_info_simple(line):
    qn = "MATCH (n:Neuron)"
    rn = " RETURN DISTINCT n.instance, n.type"

    if len(line)==0:
        return
    if (line.find("\n")) != -1:
        line = line[:-1]
    group = line.split(" ")
    for i in group:
        condition = generate_condition_single(i)
        # q = qn + condition + rn
        q = f"""{qn}{condition}{rn}"""
        try:
            results = c.fetch_custom(q)
        except:
            print(
                "This query fails. The reason may result from bad query request or the neuprint restriction. You can try it again.")
            break
        results.to_excel("output/" + str(forsave(i)) + "_simple_info" + ".xlsx")
        print(str(forsave(i)) + " Finished\n")
    return


def get_neuron_types(name):
    condition = generate_condition_single(name)
    q_type = "MATCH (n:Neuron)"
    r_type = " RETURN DISTINCT n.type"
    # q = q_type + condition + r_type
    q = f"""{q_type}{condition}{r_type}"""

    result = c.fetch_custom(q)
    return result['n.type']


def single_upstream(qu,ru,query_type,downstream_id):
    condition = generate_condition_upstream(downstream_id)
    # q = qu + condition + ru
    q = f"""{qu}{condition}{ru}"""

    try:
        results = c.fetch_custom(q)
    except:
        print(
            "This query fails. The reason may result from bad query request or the neuprint restriction. You can try it again.")
        return [[],[],[]],[[],[],[]]
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
    results.to_excel("output/" + "upstream of " + str(forsave(downstream_id)) + "_" + query_type + ".xlsx")
    print("upstream of "+str(forsave(downstream_id)) + " Finished\n")
    return [results['up_syn_coordinate_x'].values.tolist(),results['up_syn_coordinate_y'].values.tolist(),results['up_syn_coordinate_z'].values.tolist()],[results['down_syn_coordinate_x'].values.tolist(),results['down_syn_coordinate_y'].values.tolist(),results['down_syn_coordinate_z'].values.tolist()]

def single_downstream(qd,rd,query_type,upstream_id):
    condition = generate_condition_downstream(upstream_id)
    # q = qd + condition + rd
    q = f"""{qd}{condition}{rd}"""
    try:
        results = c.fetch_custom(q)
    except:
        print("This query fails. The reason may result from bad query request or the neuprint restriction. You can try it again.")
        return [[],[],[]],[[],[],[]]
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
    results.to_excel("output/" + "downstream of " + str(forsave(upstream_id)) + "_" + query_type + ".xlsx")
    print("downstream of " + str(forsave(upstream_id)) + " Finished\n")
    return [results['up_syn_coordinate_x'].values.tolist(),results['up_syn_coordinate_y'].values.tolist(),results['up_syn_coordinate_z'].values.tolist()],[results['down_syn_coordinate_x'].values.tolist(),results['down_syn_coordinate_y'].values.tolist(),results['down_syn_coordinate_z'].values.tolist()]

def pair_neuron(qp,rp,query_type,upstream_id,downstream_id):
    count = 0
    group = [upstream_id,downstream_id]
    if group[0].find("*") != -1:
        pretypes = get_neuron_types(group[0])
    else:
        pretypes = [group[0]]
    if group[1].find("*") != -1:
        postypes = get_neuron_types(group[1])
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
            condition = generate_condition_pair([pre, post])
            # q = qp + condition + rp
            q = f"""{qp}{condition}{rp}"""
            # print(rp)
            if count == 0:
                results = c.fetch_custom(q)
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
                tmp_result = c.fetch_custom(q)
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
    results.to_excel("output/" + pre_name + "_to_" + post_name + "_" + query_type + ".xlsx")
    print(pre_name + "_to_" + post_name + " Finished\n")
    return [results['up_syn_coordinate_x'].values.tolist(),results['up_syn_coordinate_y'].values.tolist(),results['up_syn_coordinate_z'].values.tolist()],[results['down_syn_coordinate_x'].values.tolist(),results['down_syn_coordinate_y'].values.tolist(),results['down_syn_coordinate_z'].values.tolist()]


def generate_group_upstream(neuron,candidate_list):
    condition=lookup(neuron)
    line="["
    for candidate in candidate_list[:-1]:
        line = line + str(candidate) + ", "
    line=line+str(candidate_list[-1])+"]"
    return " WHERE down." + condition + " AND up.bodyId IN "+str(line)

def generate_group_downstream(neuron,candidate_list):
    condition=lookup(neuron)
    line="["
    for candidate in candidate_list[:-1]:
        line = line + str(candidate) + ", "
    line=line+str(candidate_list[-1])+"]"
    return " WHERE up." + condition + " AND down.bodyId IN "+str(line)


def complicated_upstream_downstream(q,r,connection_type):
    division_number=400
    print("Here we output the intersection of all conditions")
    line_up = input("Which neuron's (id or name) upstream, exit will leave the cycle? If no upstream condition, directly press enter.\n")
    line_down = input("Which neuron's (id or name) downstream, exit will leave the cycle? If no downstream condition, directly press enter.\n")
    All_up_dict = {}
    All_down_dict = {}
    try:
        while not (line_up.find("exit") != -1 or line_down.find("exit") != -1):
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
                            condition = generate_condition_upstream(neuron)
                            q_tmp = q + condition + r
                            # print(q_tmp)
                            results = c.fetch_custom(q_tmp)
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
                                condition = generate_group_upstream(neuron, candidate_list)
                                q_tmp = q + condition + r
                                # print(q_tmp)
                                if time == 0:
                                    results = c.fetch_custom(q_tmp)
                                else:
                                    results = pd.merge(results, c.fetch_custom(q_tmp), how='outer')
                            if list_length - list_fraction * division_number > 0:
                                candidate_list = tmp_record_list[list_fraction * division_number:]
                                condition = generate_group_upstream(neuron, candidate_list)
                                q_tmp = q + condition + r
                                # print(q_tmp)
                                if list_fraction == 0:
                                    results = c.fetch_custom(q_tmp)
                                else:
                                    results = pd.merge(results, c.fetch_custom(q_tmp), how='outer')
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
                            condition = generate_condition_downstream(neuron)
                            q_tmp = q + condition + r
                            # print(q_tmp)
                            results = c.fetch_custom(q_tmp)
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
                                condition = generate_group_downstream(neuron, candidate_list)
                                q_tmp = q + condition + r
                                # print(q_tmp)
                                if time == 0:
                                    results = c.fetch_custom(q_tmp)
                                else:
                                    results = pd.merge(results, c.fetch_custom(q_tmp), how='outer')
                            if list_length - list_fraction * division_number > 0:
                                candidate_list = tmp_record_list[list_fraction * division_number:]
                                condition = generate_group_downstream(neuron, candidate_list)
                                q_tmp = q + condition + r

                                if list_fraction == 0:
                                    results = c.fetch_custom(q_tmp)
                                # print(q_tmp)
                                else:
                                    results = pd.merge(results, c.fetch_custom(q_tmp), how='outer')
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
            file_name = file_name + "intersection.xlsx"
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
            else:
                intersection_data_df.columns = ["up.bodyId", "up.instance", "up.type", "down.bodyId", "down.instance",
                                                "down.type", "up_syn.location", "down_syn.location"]
            intersection_data_df.to_excel("output/"+file_name)
            with open("output/"+file_name[:-4] + ".txt", "wt")as ff:
                for candidate in candidate_list:
                    ff.writelines(str(candidate) + "\n")
            line_up = input(
                "Which neuron's (id or name) upstream, exit will leave the cycle? If no upstream condition, directly press enter.\n")
            line_down = input(
                "Which neuron's (id or name) downstream, exit will leave the cycle? If no downstream condition, directly press enter.\n")
            All_up_dict = {}
            All_down_dict = {}
    except:
        print("Encounter some problems. Maybe you need to do more accurate survey to reduce data amount.")
        if os.path.isfile("output/"+file_name):
            with open("output/"+file_name,"wt")as ff:
                ff.writelines("errors")
        if os.path.isfile("output/"+file_name[:-4] + ".txt"):
            with open("output/"+file_name[:-4] + ".txt", "wt")as ff:
                ff.writelines("errors")

    return


##########extract synpase
q_syn="MATCH (up:`Neuron`)-[:Contains]->(:`SynapseSet`)-[:Contains]->(up_syn:`Synapse`)-[:SynapsesTo]->(down_syn:`Synapse`)<-[:Contains]-(:`SynapseSet`)<-[:Contains]-(down:`Neuron`)"
r_syn=" RETURN DISTINCT up.bodyId, up.instance, up.type, down.bodyId, down.instance, down.type, up_syn.location, down_syn.location"
q_con="MATCH (up:Neuron)-[w:ConnectsTo]->(down:Neuron)"
r_con=" RETURN DISTINCT up.bodyId, up.instance, up.type, down.bodyId, down.instance, down.type, w.weight, w.roiInfo"
q_type="MATCH (n:Neuron)"
r_type=" RETURN DISTINCT n.bodyId, n.instance, n.type"
r_type_simple=" RETURN DISTINCT n.instance, n.type"

##########
##所有上下游
def get_synapse(upstream_id,downstream_id):
    connection_type = "synapse"
    q = q_syn
    r = r_syn
    if len(upstream_id)!=0 and len(downstream_id)!=0:
        up_syn, down_syn = pair_neuron(q,r,connection_type,upstream_id,downstream_id)
    elif len(downstream_id)!=0:
        up_syn, down_syn = single_upstream(q,r,connection_type,downstream_id)
    elif len(upstream_id)!=0:
        up_syn, down_syn = single_downstream(q, r,connection_type,upstream_id)
    else:
        up_syn, down_syn = [[], [], []], [[], [], []]
    return up_syn,down_syn