import argparse
import os
import pickle
import datetime
import numpy as np
import xml.etree.ElementTree as ET
from ProNE.proNE import ProNE

NETWORK_PATH = "PROMISE/network/"
DATA_PATH = "./PROMISE/source code/"
BUG_PATH = "PROMISE/AST_encoding/datasets_1018.pickle"
TARGET_PATH = "PROMISE/AST_encoding/network_1018.pickle"
LOG_PATH = "./PROMISE/bug_log_"+datetime.datetime.now().strftime('%Y-%m-%d-%H%M')+".txt"
SOFTWARE = ["ant-1.3","ant-1.4","ant-1.5","ant-1.6","ant-1.7","camel-1.0","camel-1.2","camel-1.4",
            "camel-1.6","ivy-1.0","ivy-1.1","ivy-1.2","jedit-3.2","jedit-4.0","jedit-4.1","jedit-4.2",
            "jedit-4.3","log4j-1.0","log4j-1.1","log4j-1.2","lucene-2.0","lucene-2.2","lucene-2.4",
            "poi-1.5","poi-2.0","poi-2.5","poi-3.0","synapse-1.0","synapse-1.1","synapse-1.2",
            "velocity-1.4","velocity-1.5","velocity-1.6","xalan-2.4","xalan-2.5","xalan-2.6","xalan-2.7",
            "xerces-1.1","xerces-1.2","xerces-1.3","xerces-1.4.4"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run ProNE.")
    parser.add_argument('-graph', nargs='?', default='data/blogcatalog.ungraph', help='Graph path')
    parser.add_argument('-emb1', nargs='?', default='emb/blogcatalog.emb', help='Output path of sparse embeddings')
    parser.add_argument('-emb2', nargs='?', default='emb/blogcatalog_enhanced.emb', help='Output path of enhanced embeddings')
    parser.add_argument('-dimension', type=int, default=64, help='Number of dimensions. Default is 128.')
    parser.add_argument('-step', type=int, default=10, help='Step of recursion. Default is 10.')
    parser.add_argument('-theta', type=float, default=0.5, help='Parameter of ProNE. Default is 0.5.')
    parser.add_argument('-mu', type=float, default=0.2, help='Parameter of ProNE. Default is 0.2')
    return parser.parse_args()


def parse_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    context = root.find('context')
    container = context.find("container")
    namespace = container.findall("namespace")
    map_list = []
    for n in namespace:
        class_name = n.findall("type")
        temp_list = []
        for c in class_name:
            dependencies = c.find("dependencies")
            depend = dependencies.findall("depends-on")
            depend_name = [x.get("name") for x in depend]
            temp_list.append([c.get("name"), depend_name])
        map_list.extend(temp_list)
    return map_list


def generate_network(r_list: list, path):
    value = 0
    class_dict = {}
    for r in r_list:
        if r[0] not in class_dict:
            class_dict[r[0]] = value
            value += 1
    with open(path, mode="w") as f:
        for r in r_list:
            class_value = class_dict[r[0]]
            for c in r[1]:
                if c not in class_dict:
                    class_dict[c] = value
                    value += 1
                print(str(class_value) + " " + str(class_dict[c]), file=f)
    return class_dict


def extract_network():
    with open(BUG_PATH, mode="rb") as f:
        dataset = pickle.load(f)
    args = parse_args()
    for dir_path, dir_names, filenames in os.walk(NETWORK_PATH):
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            swn = os.path.splitext(filename)[0]
            if ext == ".odem":
                network_list = parse_xml(dir_path + "/" + filename)
                c_dict = generate_network(network_list, dir_path + "/" + swn + ".txt")
                model = ProNE(dir_path + "/" + swn + ".txt", args.emb1, args.emb2, args.dimension)
                features_matrix = model.pre_factorization(model.matrix0, model.matrix0)
                embeddings_matrix = model.chebyshev_gaussian(model.matrix0, features_matrix, args.step, args.mu,
                                                             args.theta)
                for file in dataset[SOFTWARE.index(swn)]:
                    if len(file) == 25:
                        class_name = file[0]
                        try:
                            class_index = c_dict[class_name]
                            file.append(embeddings_matrix[class_index])
                        except KeyError:
                            print(class_name + " has no dependency")
                            file.append(np.zeros(64))

    with open(TARGET_PATH, mode="wb") as f:
        pickle.dump(dataset, f)


extract_network()
