import pickle

import javalang.javalang as javalang
import os
import csv

DATA_PATH = "./PROMISE/source code/"
BUG_PATH = "PROMISE/bug-data/"
TARGET_PATH = "PROMISE/AST_encoding/datasets_1018.pickle"
LOG_PATH = "./PROMISE/bug_log_1018.txt"
TYPE = ["type_parameters", "parameters", "throws", "body", "types", "path", "static", "wildcard",
"modifiers", "annotations", "extends", "implements", "dimensions", "arguments", "sub_type",
 "return_type", "type", "declarators", "initializers", "initializer", "label", "condition", "then_statement",
 "else_statement", "control", "goto", "lock", "block", "resources", "catches", "finally_block", "cases",
 "statements", "case", "init", "update", "var", "iterable",  "method", "type_arguments",
"qualifier", "selectors", "index", "constructor_type_arguments", "constants", "declarations", "default"]
TYPE_ALL = ["type_parameters", "name", "parameters", "throws", "body", "package", "imports", "types", "path", "static", "wildcard",
"modifiers", "annotations", "extends", "implements", "dimensions", "arguments", "sub_type", "value", "values",
 "return_type", "type", "declarators", "initializers", "initializer", "varargs", "label", "condition", "then_statement",
 "else_statement", "control", "goto", "expression", "lock", "block", "resources", "catches", "finally_block", "cases",
 "statements", "case", "init", "update", "var", "iterable", "expressionl", "if_true", "if_false", "operator", "operandl",
 "operandr", "method", "type_arguments", "prefix_operators", "postfix_operators", "qualifier", "selectors", "member",
 "index", "constructor_type_arguments", "constants", "declarations", "default"]
total_word = []
type_copy = ["type_parameters", "parameters", "throws", "body", "types", "path", "static", "wildcard",
"modifiers", "annotations", "extends", "implements", "dimensions", "arguments", "sub_type",
 "return_type", "type", "declarators", "initializers", "initializer", "label", "condition", "then_statement",
 "else_statement", "control", "goto", "lock", "block", "resources", "catches", "finally_block", "cases",
 "statements", "case", "init", "update", "var", "iterable",  "method", "type_arguments",
"qualifier", "selectors", "index", "constructor_type_arguments", "constants", "declarations", "default"]


def preorder_traverse_nodename(node, node_list):
    # 如果节点是一个javalang.tree.Node对象，打印节点的类型和属性
    if isinstance(node, javalang.ast.Node):
        node_list.append(type(node).__name__)
        for child in node.children:
            preorder_traverse_nodename(child, node_list)

    elif isinstance(node, list) or isinstance(node, set):
        for child in node:
            preorder_traverse_nodename(child, node_list)
    # 如果节点是一个javalang.tree.Node对象，遍历节点的子节点


def inorder_traverse_nodename(node, stack, node_list):

    if isinstance(node, list) or isinstance(node, set) or hasattr(node, 'children'):
        stack.append(node)
    elif isinstance(node, javalang.ast.Node):
        node_list.append(type(node).__name__)

    if isinstance(node, list) or isinstance(node, set):
        for child in node:
            inorder_traverse_nodename(child, stack, node_list)
        item = stack.pop()
    # 如果节点是一个javalang.tree.Node对象，遍历节点的子节点
    elif isinstance(node, javalang.ast.Node):
        for child in node.children:
            inorder_traverse_nodename(child, stack, node_list)
        item = stack.pop()
        # print(type(item).__name__, end=",", file=f)
        node_list.append(type(item).__name__)


def levelorder_traverse_nodename(node, queue, node_list):
    if isinstance(node, javalang.ast.Node):
        queue.append(node)

    while len(queue) != 0:
        node = queue.pop(0)
        if isinstance(node, list) or isinstance(node, set):
            for child in node:
                queue.append(child)
        elif isinstance(node, javalang.ast.Node):
            node_list.append(type(node).__name__)
            for child in node.children:
                queue.append(child)


def preorder_traverse(node, node_list):
    # 如果节点是一个javalang.tree.Node对象，打印节点的类型和属性
    if isinstance(node, javalang.ast.Node):
        node_list.append(type(node).__name__)

    elif isinstance(node, str) and node:
        node_list.append(node)
    elif isinstance(node, int) or isinstance(node, float):
        node_list.append(node)

    if isinstance(node, list) or isinstance(node, set):
        for child in node:
            preorder_traverse(child, node_list)
    # 如果节点是一个javalang.tree.Node对象，遍历节点的子节点
    elif isinstance(node, javalang.ast.Node):
        for attr in node.attrs:
            if attr in TYPE:
                preorder_traverse(getattr(node, attr), node_list)


def inorder_traverse(node, stack, node_list):

    if isinstance(node, list) or isinstance(node, set) or hasattr(node, 'children'):
        stack.append(node)
    else:
        if isinstance(node, javalang.ast.Node):
            node_list.append(type(node).__name__)
        elif isinstance(node, str) and node:
            node_list.append(node)
        elif isinstance(node, int) or isinstance(node, float):
            node_list.append(node)

    if isinstance(node, list) or isinstance(node, set):
        for child in node:
            inorder_traverse(child, stack, node_list)
        item = stack.pop()
    # 如果节点是一个javalang.tree.Node对象，遍历节点的子节点
    elif isinstance(node, javalang.ast.Node):
        for attr in node.attrs:
            if attr in TYPE:
                inorder_traverse(getattr(node, attr), stack, node_list)
        item = stack.pop()
        # print(type(item).__name__, end=",", file=f)
        node_list.append(type(item).__name__)


def levelorder_traverse(node, queue, node_list):
    if isinstance(node, javalang.ast.Node) or isinstance(node, list) or isinstance(node, set):
        queue.append(node)
    elif isinstance(node, str) and node:
        queue.append(node)
    elif isinstance(node, int) or isinstance(node, float):
        queue.append(node)

    while len(queue) != 0:
        node = queue.pop(0)
        if isinstance(node, str) and node:
            node_list.append(node)
        elif isinstance(node, int) or isinstance(node, float):
            node_list.append(node)
        elif isinstance(node, list) or isinstance(node, set):
            for child in node:
                queue.append(child)
        elif isinstance(node, javalang.ast.Node):
            node_list.append(type(node).__name__)
            for attr in node.attrs:
                if attr in TYPE:
                    ele = getattr(node, attr)
                    if isinstance(ele, list) or isinstance(ele, set):
                        queue.extend(list(ele))
                    else:
                        queue.append(ele)




# def depth_count(node, depth_list, count):
#     if isinstance(node, javalang.tree.Node):
#         count += 1
#         for child in node.children:
#             if child is None or isinstance(child, set) or isinstance(child, str):
#                 continue
#             if isinstance(child, list) and len(child) != 0:
#                 for c in child:
#                     depth_count(c, depth_list, count)
#             else:
#                 depth_count(child, depth_list, count)
#         depth_list.append(count)


def process_file(csv_path):
    # if os.path.exists(result_path):  # 检查是否存在结果文件
    #     os.remove(result_path)
    # 记录所有缺陷信息的文件名
    file_list = []  # 文件信息及遍历结果
    dir_name, file_name = os.path.split(csv_path)
    dir_name = dir_name.split('/')[-1]
    file_name = '.'.join(file_name.split('.')[:-1])
    dir_temp = dir_name + '/' + file_name + '/'  # 提取ant/ant-1.3/的相对路径
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            file_list.append(row)
    # 将文件名转成路径并访问
    for file in file_list:
        path = file[0].replace('.', '/')
        file_path = DATA_PATH + dir_temp + path + '.java'
        try:
            with open(file_path, "r") as f1:
                text = f1.read()
        except Exception as e:
            with open(LOG_PATH, "a") as f2:
                print(str(e) + " 找不到文件 ", file=f2)
            continue
        try:
            tree = javalang.parse.parse(text)  # 解析java文件为AST
        except Exception as e:
            with open(LOG_PATH, "a") as f2:
                print(file_path + "/" + file[0] + " 语法错误 " + str(e), file=f2)
            continue
        stack = []
        pre_list = []
        in_list = []
        queue = []
        lev_list = []
        # with open(result_path, "a") as f3:  # 写入目标文件
        preorder_traverse(tree, pre_list)
        inorder_traverse(tree, stack, in_list)
        levelorder_traverse(tree, queue, lev_list)
        assert len(pre_list) == len(in_list) == len(lev_list)
        # TODO the effect of different traverse sequence
        file.append(pre_list)
        file.append(in_list)
        file.append(lev_list)
        global total_word
        total_word.extend(pre_list)
    return file_list


def traverse_directory(s_directory, t_directory):
    if os.path.exists(LOG_PATH):  # 检查是否存在日志文件
        os.remove(LOG_PATH)
    all_file = []
    all_version = []
    for dir_path, dir_names, filenames in os.walk(s_directory):
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            swn = os.path.splitext(filename)[0]
            if ext == ".csv":
                # print(swn, end=",")
                all_version.append(swn)
                list1 = process_file(dir_path + "/" + filename)
                print(filename + " is done")
                all_file.append(list1)
    with open(t_directory, mode='wb') as fp:
        pickle.dump(all_file, fp)
    print("finish!")
    print(all_version)


def simplify_wordfrequency(word_list: list):
    word_frequency = {}
    for word in word_list:
        word = word.lower()
        if word not in word_frequency:
            word_frequency[word] = 1
        else:
            word_frequency[word] += 1
    print(f"origin number of word: {len(word_frequency)}")
    delete_word = [k for k in word_frequency if word_frequency[k] < 5]
    include_word = list(word_frequency.keys())
    for w in delete_word:
        include_word.remove(w)
    print(f"simplify number of word: {len(word_frequency)-len(delete_word)}")
    with open(TARGET_PATH, mode="rb") as f:
        data = pickle.load(f)
    for project in data:
        for f in project:
            if len(f) == 25:
                f[22] = [x.lower() for x in f[22] if x.lower() in include_word]
                f[23] = [x.lower() for x in f[23] if x.lower() in include_word]
                f[24] = [x.lower() for x in f[24] if x.lower() in include_word]
    with open("PROMISE/AST_encoding/datasets0711_simplify.pickle", mode="wb") as f:
        pickle.dump(data, f)


traverse_directory(BUG_PATH, TARGET_PATH)
# simplify_wordfrequency(total_word)
# process_file("PROMISE/bug-data/ant/ant-1.3.csv", "PROMISE/token/ant-1.3.txt")
# b_tree = ast_to_binary_tree(tree)
# stack = []
# preorder_traverse(tree)
# print()
# inorder_traverse(tree, stack)
# with open("PROMISE/source code/camel/camel-1.2/org/apache/camel/CamelContext.java", "r") as f1:
#     text = f1.read()
# with open("PROMISE/source code/test.java", "r") as f2:
#     text = f2.read()
# tree = javalang.parse.parse("public class A{public static void main(String[] args){int i=0; while(i<10){i++;} }}")
# stack = []
# pre_list = []
# in_list = []
# lev_list = []
# queue = []
# preorder_traverse(tree, pre_list)
# inorder_traverse(tree, stack, in_list)
# levelorder_traverse(tree, queue, lev_list)
# print(pre_list)
# print(in_list)
# print(lev_list)
