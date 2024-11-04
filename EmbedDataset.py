import pickle
import random

import math
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F
from dat.util.misc import NestedTensor


DATA_PATH = "PROMISE/AST_encoding/network_1018.pickle"
WORD_MAP_PATH = "PROMISE/AST_encoding/wordid_1018.pickle"
SOFTWARE = ["ant-1.3","ant-1.4","ant-1.5","ant-1.6","ant-1.7","camel-1.0","camel-1.2","camel-1.4",
            "camel-1.6","ivy-1.0","ivy-1.1","ivy-1.2","jedit-3.2","jedit-4.0","jedit-4.1","jedit-4.2",
            "jedit-4.3","log4j-1.0","log4j-1.1","log4j-1.2","lucene-2.0","lucene-2.2","lucene-2.4",
            "poi-1.5","poi-2.0","poi-2.5","poi-3.0","synapse-1.0","synapse-1.1","synapse-1.2",
            "velocity-1.4","velocity-1.5","velocity-1.6","xalan-2.4","xalan-2.5","xalan-2.6","xalan-2.7",
            "xerces-1.1","xerces-1.2","xerces-1.3","xerces-1.4.4"]
FILE_MAX_SIZE = 13608
EMBED_DIM = 256
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def wordNumCount(source, target=None):
    file_list = []
    maxfile = 0
    with open(DATA_PATH, mode='rb') as file:
        data = pickle.load(file)
    if type(source) == str:
        file_list.extend(data[SOFTWARE.index(source)])
    else:
        for x in source:
            file_list.extend(data[SOFTWARE.index(x)])
    if target is not None:
        if type(target) == str:
            file_list.extend(data[SOFTWARE.index(target)])
        else:
            for x in source:
                file_list.extend(data[SOFTWARE.index(x)])

    for f in file_list:
        if len(f) == 26:
            if len(f[22]) > maxfile:
                maxfile = len(f[22])
    return maxfile


class EmbedDataset(Dataset):
    def __init__(self, version, flag, word_num: int):
        self.version = version
        assert flag in ["train", "test"]
        self.flag = flag
        self.data = self.__load_data__(DATA_PATH, WORD_MAP_PATH, self.version, word_num)

    def __getitem__(self, index):
        # 根据索引返回数据
        return self.data[0][index], self.data[1][index]

    def __len__(self):
        # 返回数据的长度
        return len(self.data[1])

    def __load_data__(self, defect_data, map_file, version, word_num):
        with open(defect_data, mode='rb') as file:
            data = pickle.load(file)
        with open(map_file, mode='rb') as file:
            word_id = pickle.load(file)
        wordidx_num = len(word_id)
        w0 = [k for k, v in word_id.items() if v == 0]
        word_id[w0[0]] = wordidx_num
        emb = torch.nn.Embedding(wordidx_num+1, EMBED_DIM, padding_idx=0)
        # 将选择的版本合并
        source_list = []
        if type(version) == str:
            source_list.extend(data[SOFTWARE.index(version)])
        else:
            for x in version:
                source_list.extend(data[SOFTWARE.index(x)])

        ast = []
        label = []

        for f in source_list:
            if len(f) == 25:
                pre_map = np.array([word_id[x.lower()]+1 for x in f[22]])  # 有的文件由于语法错误或找不到无AST
                in_map = np.array([word_id[x.lower()]+1 for x in f[23]])
                if word_num - len(pre_map) > 0:
                    if (word_num - len(pre_map)) % 2 == 0:
                        pad1 = pad2 = (word_num - len(pre_map)) // 2
                    else:
                        pad1 = math.ceil((word_num - len(pre_map))/2)
                        pad2 = math.floor((word_num - len(pre_map))/2)
                    pre_map = np.pad(pre_map, (pad1, pad2), 'constant')
                    in_map = np.pad(in_map, (pad1, pad2), 'constant')

                else:
                    pre_map = pre_map[: word_num]
                    in_map = in_map[: word_num]
                pre_embed = emb(torch.tensor(pre_map).long())
                in_embed = emb(torch.tensor(in_map).long())

                ast.append(torch.stack([pre_embed, in_embed], dim=0))
                label.append([0, 1.0] if float(f[21]) else [1.0, 0])  # 二分类交叉熵要求的one-hot编码
        return torch.stack(ast, dim=0), np.array(label)


class EmbedDatasetWithMask(Dataset):
    def __init__(self, version, flag, pad_type, word_num=None, batch_size=None):
        assert flag in ["train", "test", "network"]
        self.version = version
        self.flag = flag
        self.word_num = word_num
        self.wordidx_num = 6110
        self.sample_len = None
        self.data = self.load_data(DATA_PATH, WORD_MAP_PATH, self.version)
        if pad_type is None and batch_size is not None:
            self.data = self.batch_pad(self.data, batch_size)
        elif pad_type == "bottom":
            self.data = self.cut_pad(self.data, word_num, pad_type)
        elif pad_type == "side":
            self.data = self.cut_pad(self.data, word_num, pad_type)

    def __getitem__(self, index):
        # 根据索引返回数据
        return self.data[0][index], self.data[1][index], self.data[2][index], self.data[3][index]

    def __len__(self):
        # 返回数据的长度
        return len(self.data[2])

    def load_data(self, defect_data, map_file, version):
        sample_len = []
        with open(map_file, mode='rb') as file:
            word_id = pickle.load(file)
        wordidx_num = len(word_id)
        self.wordidx_num = wordidx_num
        w0 = [k for k, v in word_id.items() if v == 0]
        word_id[w0[0]] = wordidx_num
        # emb = torch.nn.Embedding(wordidx_num+1, EMBED_DIM, padding_idx=0)
        # 将选择的版本合并
        source_list = []
        with open(defect_data, mode='rb') as file:
            data = pickle.load(file)
        if type(version) == str:
            source_list.extend(data[SOFTWARE.index(version)])
        else:
            for x in version:
                source_list.extend(data[SOFTWARE.index(x)])
        sample = []
        net = []
        label = []
        max_map = 0
        for f in source_list:
            if len(f) == 26:
                pre_map = np.array([word_id[x.lower()] for x in f[22]])  # 有的文件由于语法错误或找不到无AST
                in_map = np.array([word_id[x.lower()] for x in f[23]])
                lev_map = np.array([word_id[x.lower()] for x in f[24]])
                sample_len.append(len(pre_map))
                if len(pre_map) > max_map:
                    max_map = len(pre_map)
                sample.append([pre_map, in_map, lev_map])
                net.append(f[25])
                label.append([0, 1.0] if float(f[21]) else [1.0, 0])  # 二分类交叉熵要求的one-hot编码
        # if self.flag == "train":
        #     sample, net, label = self.undersample(sample, net, label)
        #     sample, net, label = self.groupsample(sample, net, label, 0.5)
        #     sample, label = self.create_sample(sample, label)
        #     sample_temp = [np.stack(x, axis=0) for x in sample]
        #     sample_flatten = []
        #     for x in sample_temp:
        #         sample_pad = np.pad(x, ((0, 0), (0, max_map-x.shape[1])))
        #         sample_flatten.append(sample_pad.flatten())
        #     sm = SMOTE(random_state=42)
        #     label_smote = [x[1] for x in label]
        #     sample_smote, label_smote = sm.fit_resample(sample_flatten, np.array(label_smote))
        #     # sample, label_smote = self.create_sample(sample, label)
        #     sample_list = []
        #     for x in sample_smote:
        #         x_reshape = np.array(x).reshape(3, -1)
        #         sample_list.append([np.trim_zeros(xx, trim='b') for xx in x_reshape])
        #     label = []
        #     for x in label_smote:
        #         label.append([0, 1.0] if x else [1.0, 0])
        #     sample = sample_list
        # sample = self.file_scale(sample, 17)
        self.sample_len = sample_len
        return sample, net, np.array(label)

    def file_scale(self, sample, file_min_len):
        new_sample = []
        for x in sample:
            if len(x[0]) < file_min_len:
                new_sample.append(np.tile(np.array(x), 16))
            else:
                new_sample.append(np.array(x))
        return new_sample

    def groupsample(self, sample, net, label, normal_rate):
        defect_label = []
        normal_label = []
        sample_index = []
        data_num = len(label)
        for i in range(data_num):
            if label[i][1] == 1:
                defect_label.append(i)
            else:
                normal_label.append(i)
        normal_num = int(normal_rate * data_num)
        defect_num = data_num-normal_num

        label_index = np.random.randint(0, len(defect_label), data_num-normal_num)
        data_index = [defect_label[i] for i in label_index]

        label_index = np.random.randint(0, len(normal_label), normal_num)
        data_index += [normal_label[i] for i in label_index]
        for x in range(len(sample)):
            if x not in data_index:
                sample_index.append(x)
        sample = [sample[x] for x in sample_index]
        net = [net[x] for x in sample_index]
        label = [label[x] for x in sample_index]
        return sample, net, label

    def undersample(self, sample, net, label):
        defect_label = []
        normal_label = []
        sample_index = []
        for i in range(len(label)):
            if label[i][1] == 1:
                defect_label.append(i)
            else:
                normal_label.append(i)
        gap = len(defect_label) - len(normal_label)
        if gap > 0:
            label_index = np.random.randint(0, len(defect_label), gap)
            data_index = [defect_label[i] for i in label_index]
        else:
            label_index = np.random.randint(0, len(normal_label), -gap)
            data_index = [normal_label[i] for i in label_index]
        for x in range(len(sample)):
            if x not in data_index:
                sample_index.append(x)
        sample = [sample[x] for x in sample_index]
        net = [net[x] for x in sample_index]
        label = [label[x] for x in sample_index]
        print("number of dataset sample: %d" % len(sample))
        return sample, net, label

    def oversample(self, sample, net, label):
        defect_label = []
        normal_label = []
        for i in range(len(label)):
            if label[i][1] == 1:
                defect_label.append(i)
            else:
                normal_label.append(i)
        gap = len(defect_label)-len(normal_label)
        if gap > 0:
            label_index = np.random.randint(0, len(normal_label), gap)
            data_index = [normal_label[i] for i in label_index]
        else:
            label_index = np.random.randint(0, len(defect_label), -gap)
            data_index = [defect_label[i] for i in label_index]

        sample = sample + [sample[x] for x in data_index]
        net = net + [net[x] for x in data_index]
        label = label + [label[x] for x in data_index]
        return sample, net, label

    def pad(self, integer_map: list, word_num, pad_type):
        pad_result = []
        assert pad_type in ["bottom", "side", "none"]
        if pad_type == "none":
            pad_result = integer_map
            mask = np.ones(integer_map[0].shape[0], dtype=np.uint8)
        else:
            if word_num - len(integer_map[0]) > 0:
                if pad_type == "side":
                    if (word_num - len(integer_map[0])) % 2 == 0:
                        pad1 = pad2 = (word_num - len(integer_map[0])) // 2
                    else:
                        pad1 = math.ceil((word_num - len(integer_map[0])) / 2)
                        pad2 = math.floor((word_num - len(integer_map[0])) / 2)
                    mask = np.concatenate(
                        [np.zeros(pad1, dtype=np.uint8), np.ones(len(integer_map[0]), dtype=np.uint8),
                         np.zeros(pad2, dtype=np.uint8)])[None, :]

                elif pad_type == "bottom":
                    pad1 = 0
                    pad2 = word_num - len(integer_map[0])
                    mask = np.concatenate([np.ones(len(integer_map[0]), dtype=np.uint8),
                                           np.zeros(pad2, dtype=np.uint8)])[None, :]

                for x in integer_map:
                    pad_result.append(np.pad(x, (pad1, pad2), 'constant'))
            else:
                for x in integer_map:
                    pad_result.append(x[: word_num])
                mask = np.ones(word_num, dtype=np.uint8)[None, :]
        mask = np.repeat(mask, 64, axis=0).T
        return np.stack(pad_result), mask

    def create_sample(self, sample, label):
        defect_list = []
        normal_list = []
        for i in range(len(label)):
            if label[i][1] == 1:
                defect_list.append(sample[i])
            else:
                normal_list.append(sample[i])
        gap = len(normal_list) - len(defect_list)
        if gap > 0:
            cs = self.perturbation(defect_list, gap)
            return sample + cs, np.concatenate([label, np.tile(np.array([0, 1.0]), (len(cs), 1))], axis=0)
        else:
            return sample, label

    def perturbation(self, sample: list, gap: int):
        p_list = []
        p_type = random.choice(["outCross", "outPlus", "mutation"])
        num = abs(gap)
        while num:
            x = random.choice(sample)
            if p_type == "inCross":
                x1, x2, x3 = x
                point = random.randint(1, len(x1))
                temp = x1[:point]
                x1[:point] = x2[:point]
                x2[:point] = x3[:point]
                x3[:point] = temp
                p_list.append([x1, x2, x3])
            elif p_type == "outCross":
                x1, x2, x3 = x
                y1, y2, y3 = random.choice(sample)
                point = random.randint(1, min(len(x1), len(y1)))
                x1, _ = self.swap(x1[:point], y1[:point])
                x2, _ = self.swap(x2[:point], y2[:point])
                x3, _ = self.swap(x3[:point], y3[:point])
                p_list.append([x1, x2, x3])
            elif p_type == "outPlus":
                x1, x2, x3 = x
                y1, y2, y3 = random.choice(sample)
                point = random.randint(1, len(y1))
                x1 = np.append(x1, y1[point:])
                x2 = np.append(x2, y2[point:])
                x3 = np.append(x3, y3[point:])
                # if len(x1) > self.word_num:
                #     x1 = x1[:self.word_num]
                #     x2 = x2[:self.word_num]
                #     x3 = x3[:self.word_num]
                p_list.append([x1, x2, x3])
            elif p_type == "mutation":
                x1, x2, x3 = x
                point = np.random.randint(0, len(x1), int(len(x1)*0.4))
                for i in point:
                    x1[i] = random.randint(1, self.wordidx_num)
                    x2[i] = random.randint(1, self.wordidx_num)
                    x3[i] = random.randint(1, self.wordidx_num)
                p_list.append([x1, x2, x3])
            num -= 1

        return p_list

    def swap(self, x, y):
        temp = x
        x = y
        y = temp
        return x, y

    def batch_pad(self, data, batch_size):
        feature, net, label = data
        feature_lens = dict(zip(list(range(len(feature))), [len(x[0]) for x in feature]))
        feature_lens_sorted = sorted(feature_lens.items(), key=lambda e: e[1])

        len_list = [feature_lens_sorted[i:i + batch_size] for i in range(0, len(feature_lens_sorted), batch_size)]
        sorted_id = [x[0] for x in feature_lens_sorted]
        paded_feature = []
        paded_mask = []
        net_sc = []
        new_label = []
        n = 0
        for l in [sorted_id[i:i + batch_size] for i in range(0, len(sorted_id), batch_size)]:
            pad_num = max([x[1] for x in len_list[n]])
            if pad_num < 15:
                pad_num = 15
            for index in l:
                f, m = self.pad(feature[index], pad_num, pad_type="side")
                paded_feature.append(f)
                paded_mask.append(m)
                net_sc.append(net[index])
                new_label.append(label[index])
            n += 1
        return paded_feature, paded_mask, net_sc, new_label

    def cut_pad(self, data, word_num, pad_type):
        masks = []
        ast = []
        sample, net, label = data
        for _ in sample:
            pad_result, mask = self.pad(_, word_num, pad_type)
            masks.append(mask)
            ast.append(pad_result)
        return ast, masks, net, label


class StratifiedShuffleSplit(Sampler):
    def __init__(self, data_source, batch_size, none_defect_rate):
        super(StratifiedShuffleSplit, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.all_sample = self.create_sampler(none_defect_rate)

    def __iter__(self):
        batch = []
        for index in self.all_sample:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return math.floor(len(self.data_source[0])/self.batch_size)

    def create_sampler(self, none_defect_rate):
        defect_index = []
        none_defect_index = []
        all_sample = []
        _, _, _, label = self.data_source

        for i in range(len(label)):
            if label[i][1] == 1:
                defect_index.append(i)
            else:
                none_defect_index.append(i)

        if none_defect_rate is None:
            none_defect_rate = len(defect_index) / len(label)  # 根据正负例比例决定抽样比例
        batch_size_threshold = self.batch_size * none_defect_rate

        for i in range(len(label)):  # 按照batch_size大小排列样本，确保每个batch中有相应比例的正负例
            if i % self.batch_size < batch_size_threshold:
                n = random.randint(0, len(none_defect_index) - 1)
                all_sample.append(none_defect_index[n])
            else:
                n = random.randint(0, len(defect_index) - 1)
                all_sample.append(defect_index[n])
        return all_sample


def train_model(device, dataloader, model, loss_fn, optimizer, scheduler=None, emb=None, writer=None, epoch=0):
    size = len(dataloader)
    model.train()
    total_loss = 0
    num = epoch * size
    for batch, (x, mask, net, y) in enumerate(dataloader):
        correct = 0
        if emb is not None:
            x = emb(x)
        X = NestedTensor(x.float(), mask)
        y = y.float()
        net = net.float().to(device)
        X, y = X.to(device), y.to(device)
        model = model.cuda()
        optimizer.zero_grad()
        prop = model(X, net)
        # prep = F.softmax(pred, dim=1)
        #
        y_pred = F.softmax(prop, dim=1).argmax(1).cpu().tolist()
        y_true = y.cpu().numpy()
        for i in range(len(y_true)):
            if y_pred[i] == y_true[i][1]:
                correct += 1

        loss = loss_fn(prop, y)
        loss.backward(retain_graph=True)
        optimizer.step()
        # if batch % 100 == 0:
        loss = loss.item()
        total_loss += loss
        if writer is not None:
            writer.add_scalar("Loss/training loss", loss, num)
        num += 1
        # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    if scheduler is not None:
        scheduler.step()
    print(f"loss: {total_loss/size:>7f}")


def test_model(device, dataloader, model, loss_fn, emb=None, writer=None):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    current = 0
    y_prop = []
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for (x, mask, net, y) in dataloader:
            if emb is not None:
                x = emb(x)
            # x = x.permute(1, 0, 2, 3)
            correct = 0
            y_true = []
            X = NestedTensor(x.float(), mask)
            y = y.float()
            net = net.float().to(device)
            X, y = X.to(device), y.to(device)
            prop = model(X, net)

            current_loss = loss_fn(prop, y).item()
            test_loss += current_loss

            # print(f"test_loss: {current_loss:>7f}")

            for yy in y.cpu().numpy():
                if yy[0] == 0:
                    y_true.append(1)
                else:
                    y_true.append(0)

            y_pred = F.softmax(prop, dim=1).argmax(1).cpu().tolist()
            for i in range(len(y_true)):
                if y_pred[i] == y_true[i]:
                    correct += 1

            y_pred_list.extend(y_pred)
            y_true_list.extend(y_true)
            prop_softmax = F.softmax(prop, dim=1)
            y_prop.extend(prop_softmax.cpu().numpy()[:, 1])
            if writer is not None:
                print("Loss/testing loss: " + str(current_loss))
                writer.add_scalar("Loss/testing loss", current_loss, current)

            current += 1
            # correct += torch.eq(y, pred).sum().item()/2
    # correct += (y_pred == y.true).type(torch.float).sum().item()
    test_loss /= num_batches
    if writer is not None:
        writer.add_text("results", "Avg_test_loss: " + str(test_loss), 0)
    print("Avg_test_loss: " + str(test_loss))
    return y_true_list, y_pred_list, y_prop
