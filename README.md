This is a code copy of our proposed Multi-scale feature fusion with attention mechanism for software defect prediction (MFA) model. 
train_mode2 is model training and prediction.
MFA is the model structure, and dat is the deformable transformer. 

PROMISE contains the dataset and related word embedding files. network_1018.pickle includes all datasets we used. Minimal datasets.zip is a copy of original datasets.

The AST.py file uses javalang to parse the source code and generate a triple traversal sequence.

The dependency.py embeds Class Dependency Networks using ProNE.

Please get the ProNE model and javalang program from the corresponding links.
Javalang:https://github.com/c2nes/javalang
ProNE:https://github.com/THUDM/ProNE

Training and testing this model by simply run the trian_model2.py
