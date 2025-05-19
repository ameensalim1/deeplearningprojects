import sys
import os
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy

# --- Configuration ---
# Choose dataset: "MNIST", "CIFAR", "ORL" (must match pickle filename in data/stage_3_data/)
DATASET_CHOICE = "MNIST" 
METHOD_CHOICE = "CNN"

# 1. Setup Dataset Loader
print(f"Setting up dataset: {DATASET_CHOICE}")
dataset = Dataset_Loader(dName=f"{DATASET_CHOICE} Dataset", 
                         dDescription=f"Loading {DATASET_CHOICE} image data",
                         dataset_name=DATASET_CHOICE) # This 'dataset_name' is key for pickle file
dataset.load() # Load data to populate image_channels, image_height, etc.


print("Setting up CNN method for Architecture 2 (MNIST)...")
method = Method_CNN(
    mName=f"CNN_Arch2_on_{DATASET_CHOICE}", 
    mDescription="CNN Architecture 2 for image classification",
    input_channels=dataset.image_channels,
    num_classes=dataset.num_classes,
    image_size=(dataset.image_height, dataset.image_width),
    conv_channels=(32, 64),         # Architecture 2: Conv. Block Channel (32, 64)
    fc_hidden_size=256,             # Architecture 2: Hidden Fully Connected Layer: 256
    kernel_size=3                   # Architecture 2: Kernel: 3x3 (default, but explicit)
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
