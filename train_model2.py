import os
import torch
import torchvision
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score, matthews_corrcoef, \
    precision_score, recall_score
from torch import optim, nn
from torch.utils.data import DataLoader
from EmbedDataset import wordNumCount, EmbedDatasetWithMask, train_model, test_model, StratifiedShuffleSplit
from dat.backbone import build_backbone
from dat.MFA import MFA
from dat.deformable_transformer import build_deforamble_transformer
import time
import pickle
from dat_config import get_args_parser

epochs = 20
BATCH_SIZE = 8
WORD_MAP_PATH = "PROMISE/AST_encoding/wordid_1018.pickle"
SOURCE = ["ant-1.4"]
TARGET = ["ant-1.6"]
TOTAL = ["ant-1.4", "ant-1.6", "ant-1.7", "camel-1.2", "camel-1.4", "camel-1.6", "jedit-3.2", "jedit-4.0", "jedit-4.1",
         "lucene-2.0", "lucene-2.2", "lucene-2.4", "poi-1.5", "poi-2.5", "poi-3.0", "velocity-1.4", "velocity-1.5",
         "velocity-1.6", "xalan-2.4", "xalan-2.5", "xalan-2.6"]
SAVE_PATH = 'savemodel/deformableTR/'

EMBED_DIM = 64
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
with open(WORD_MAP_PATH, mode='rb') as file:
    word_id = pickle.load(file)
wordidx_num = len(word_id)
print(f"number of word type: {wordidx_num}")


def execution_time(step: str, start):
    end = time.time()
    print(step + f" execution time: {(end - start)/60:.2f} minutes")


args = get_args_parser().parse_args()
start_time = time.time()
word_num = wordNumCount(SOURCE, TARGET)
print("sequence word num: %d" % word_num)
glove_model = torch.load("PROMISE/AST_encoding/glove3_1018_64.pt")
map_vector = (glove_model["_focal_embeddings.weight"] + glove_model["_context_embeddings.weight"]).cpu()
pad_tensor = torch.zeros(map_vector.shape[1], dtype=torch.float64)
map_vector = torch.cat((map_vector, map_vector[0:1]), dim=0)
map_vector[0] = pad_tensor
emb = torch.nn.Embedding.from_pretrained(map_vector.cpu(), freeze=False, padding_idx=0)

# emb = torch.nn.Embedding(wordidx_num + 1, EMBED_DIM, padding_idx=0)

train = EmbedDatasetWithMask(SOURCE, "train", "bottom", word_num=word_num)
test = EmbedDatasetWithMask(TARGET, "test", "bottom", word_num=word_num)
Stratified_train = StratifiedShuffleSplit(train.data, BATCH_SIZE, 0.5)
train_dataloader = DataLoader(train, batch_sampler=Stratified_train)
# train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test, shuffle=True)

backbone = build_backbone(args)
transformer = build_deforamble_transformer(args)

model = MFA(
    backbone,
    transformer,
    num_classes=2,
    num_queries=args.num_queries,
    num_feature_levels=args.num_feature_levels,
)

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model)

# 加载断点继续训练
# state_dict = torch.load(SAVE_PATH + '#'.join(SOURCE) + "_" + TARGET + ".pth")
# model.load_state_dict(state_dict)
def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out

# for name, param in model.named_parameters():
#     print(name, param.requires_grad)
model = model.to(device)
param_dicts = [
    {
        "params":
            [p for n, p in model.named_parameters()
             if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
        "lr": args.lr,
    },
    {
        "params": [p for n, p in model.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
        "lr": args.lr_backbone,
    },
    {
        "params": [p for n, p in model.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
        "lr": args.lr * args.lr_linear_proj_mult,
    }
]
execution_time("parameterInitialize", start_time)
optimizer = optim.AdamW(param_dicts, lr=args.lr)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1, last_epoch=-1)

#  训练
num = 0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_model(device, train_dataloader, model, criterion, optimizer, scheduler, emb)
    execution_time("train", start_time)
    # y_true, y_pred, y_prop = test_model(device, test_dataloader, model, criterion)

# if not os.path.exists(SAVE_PATH):
#     os.makedirs(SAVE_PATH)
# torch.save(model.state_dict(), SAVE_PATH + '#'.join(SOURCE) + "_" + TARGET + ".pth")

# state_dict = torch.load(SAVE_PATH + '#'.join(SOURCE) + "_" + TARGET + ".pth")
# model.load_state_dict(state_dict)

y_true, y_pred, y_prop = test_model(device, test_dataloader, model, criterion, emb)
auc = roc_auc_score(y_true, y_prop)
print(f"AUC:{auc:>0.4f}")
acc = accuracy_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print("TN: %d, FP: %d, FN: %d, TP: %d" % (tn, fp, fn, tp))
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')
mcc = matthews_corrcoef(y_true, y_pred)
print("accuracy: %.3f, precision: %.3f, recall: %.3f, f1: %.3f, mcc: %.3f," % (acc, precision, recall, f1, mcc))
# accuracy, precision, recall, f1, mcc, TN, TP, FP, FN = MyEvaluate.metric(y_true, y_pred)
# MyEvaluate_show.metric(y_true, y_pred)
# print("accuracy: %.2f, precision: %.2f, recall: %.2f, f1: %.2f, mcc: %.2f," % (accuracy, precision, recall, f1, mcc))
# print("TN: %d, TP: %d, FP: %d, FN: %d" % (TN, TP, FP, FN))
# print("Done!")
