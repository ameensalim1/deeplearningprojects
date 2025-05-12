import sys
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy

# --- Configuration ---
# Choose dataset: "MNIST", "CIFAR", "ORL" (must match pickle filename in data/stage_3_data/)
DATASET_CHOICE = "ORL" 
METHOD_CHOICE = "CNN"
OUTPUT_CSV   = "output/architecture_comparison.csv"

# 1. Setup Dataset Loader
print(f"Setting up dataset: {DATASET_CHOICE}")
dataset = Dataset_Loader(dName=f"{DATASET_CHOICE} Dataset", 
                         dDescription=f"Loading {DATASET_CHOICE}",
                         dataset_name=DATASET_CHOICE) # This 'dataset_name' is key for pickle file
data = dataset.load() # Load data to populate image_channels, image_height, etc.
if data['train'] is None or data['test'] is None:
    raise RuntimeError(f"Failed to load train or test split for {DATASET_CHOICE}")
N, C, H, W = data['train']['X'].shape
num_classes = len(np.unique(data['train']['y']))

architectures = [
    {"name":"Arch_A","conv_feats":[16,32], "fc_dim":128,"kernel":3},
    {"name":"Arch_B","conv_feats":[32,64], "fc_dim":256,"kernel":3},
    {"name":"Arch_C","conv_feats":[64,128],"fc_dim":512,"kernel":5},
    {"name":"Arch_D","conv_feats":[128,256],"fc_dim":256,"kernel":3},
]

results = []
methods = []

# Loop over architectures
for arch in architectures:
    print(f"\n--- Training {arch['name']} ---")
    # a) build a fresh Method_CNN
    method = Method_CNN(
      mName=f"{arch['name']}_{DATASET_CHOICE}", 
      mDescription="sweep",
      input_channels=C,
      num_classes=num_classes,
      image_size=(H, W),
      optimizer_cls=torch.optim.Adam
    )
    # b) overwrite conv1 & conv2
    in_c = C
    for i, out_c in enumerate(arch["conv_feats"], start=1):
        setattr(method, f"conv{i}",
                torch.nn.Conv2d(in_c, out_c,
                                kernel_size=arch["kernel"],
                                padding=arch["kernel"]//2))
        in_c = out_c
    # c) recompute flatten & FC
    # c) recompute flatten & FC (now matches your forward exactly)
    dummy = torch.zeros(1, C, H, W)
    with torch.no_grad():
        x = method.relu1   (method.conv1(dummy))
        x = method.pool1   (x)
        x = method.relu2   (method.conv2(x))
        x = method.pool2   (x)
    flat_n = x.view(1, -1).shape[1]
    method.flattened_size = flat_n
    method.fc1 = torch.nn.Linear(flat_n, arch["fc_dim"])
    method.fc2 = torch.nn.Linear(arch["fc_dim"], num_classes)

    # d) prepare and run
    evaluator = Evaluate_Accuracy("eval","")
    saver = Result_Saver(arch["name"], "")
    saver.result_destination_folder_path = "output/stage_3_results/"
    saver.result_destination_file_name     = f"{arch['name']}_{DATASET_CHOICE}"

    setting = Setting_Train_Test_Split("split","")
    setting.prepare(dataset, method, saver, evaluator)
    scores = setting.load_run_save_evaluate()
    scores["architecture"] = arch["name"]
    results.append(scores)
    methods.append(method)

# 4) tabulate
df = pd.DataFrame(results).set_index("architecture")
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV)
print(df)

plt.figure()
for arch, loss_hist in zip(architectures, [m.train_losses for m in methods]):
    plt.plot(loss_hist, label=arch["name"])
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title(f"{METHOD_CHOICE} Train Loss Curves on {DATASET_CHOICE}")
plt.legend()
plt.grid(True)
plt.savefig(f"output/{DATASET_CHOICE}_all_arch_train_loss.png")

"""
print("Setting up CNN method...")
method = Method_CNN(
    mName=f"CNN_on_{DATASET_CHOICE}", 
    mDescription="CNN for image classification",
    input_channels=dataset.image_channels,
    num_classes=dataset.num_classes,
    image_size=(dataset.image_height, dataset.image_width)
)

# 3. Setup Result Saver and Evaluator
result_saver = Result_Saver('stage_3_cnn_results', 'Pickled experiment results') # rName and rType
result_saver.result_destination_folder_path = 'output/stage_3_results/'
result_saver.result_destination_file_name = f'{method.method_name}_results' # .pkl will be added by save()
evaluator = Evaluate_Accuracy(eName='Accuracy Evaluator', eDescription='')

# 4. Setup Setting
print("Setting up Train/Test Split Setting...")
settings = Setting_Train_Test_Split(
    sName=f'TrainTestSplit_{METHOD_CHOICE}_on_{DATASET_CHOICE}',
    sDescription='Basic train/test split experiment'
)
settings.prepare(
    sDataset=dataset, 
    sMethod=method,
    sResult=result_saver,
    sEvaluate=evaluator
)

# 5. Run everything
print("Starting experiment run...")
final_scores = settings.load_run_save_evaluate()

if final_scores:
    print(f"Experiment completed. Final Scores: {final_scores}")
else:
    print("Experiment run failed or returned no scores.")
"""