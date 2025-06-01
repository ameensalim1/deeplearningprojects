'''
Main script for GCN Node Classification experiments
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
import sys

# Add the project root to Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
current_dir   = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.Method_GCN import Method_GCN
from code.stage_5_code.Result_Saver import Result_Saver
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_5_code.Setting_Graph_Node_Classification import Setting_Graph_Node_Classification


def run_gcn_experiment(dataset_name):
    """
    Run GCN experiment on a specific dataset
    """
    print(f"\n{'='*60}")
    print(f"Running GCN experiment on {dataset_name.upper()} dataset")
    print(f"{'='*60}")
    
    # ---- Step 1: Setup Dataset ----
    data = Dataset_Loader()
    data.dataset_name = dataset_name
    data.dataset_source_folder_path = f'data/stage_5_data/stage_5_data/{dataset_name}'
    data.dataset_source_file_name = dataset_name
    data.dataset_description = f'{dataset_name} node classification dataset'

    # ---- Step 2: Setup Method ----
    method = Method_GCN('GCN', 'Graph Convolutional Network for Node Classification')
    
    # ---- Step 3: Setup Result Saver ----
    result = Result_Saver('saver', 'result saver')
    result.result_destination_folder_path = 'result/stage_5_result/'
    result.result_destination_file_name = f'{dataset_name}_gcn'
    result.fold_count = 0
    
    # Ensure result directory exists
    os.makedirs(result.result_destination_folder_path, exist_ok=True)
    
    # ---- Step 4: Setup Evaluator ----
    evaluate = Evaluate_Accuracy('accuracy', 'classification accuracy')
    
    # ---- Step 5: Setup Setting ----
    setting = Setting_Graph_Node_Classification('graph_node_classification', 'Graph-based node classification setting')
    
    # ---- Step 6: Connect Components ----
    setting.prepare(data, method, result, evaluate)
    
    # ---- Step 7: Print Setup Summary ----
    setting.print_setup_summary()
    
    # ---- Step 8: Run Experiment ----
    accuracy, detailed_results = setting.load_run_save_evaluate()
    
    print(f"\nExperiment completed for {dataset_name.upper()}")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    
    return accuracy, detailed_results


def main():
    """
    Main function to run all experiments
    """
    print("Starting GCN Node Classification Experiments")
    print("=" * 60)
    
    # List of datasets to experiment with
    datasets = ['cora', 'pubmed', 'citeseer']
    results_summary = {}
    
    # Run experiments on all datasets
    for dataset_name in datasets:
        try:
            accuracy, detailed_results = run_gcn_experiment(dataset_name)
            results_summary[dataset_name] = {
                'accuracy': accuracy,
                'test_accuracy': detailed_results['test_accuracy'],
                'epochs': len(detailed_results['train_losses'])
            }
        except Exception as e:
            print(f"Error running experiment on {dataset_name}: {str(e)}")
            results_summary[dataset_name] = {'error': str(e)}
    
    # Print final summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for dataset, result in results_summary.items():
        if 'error' in result:
            print(f"{dataset.upper():10s}: ERROR - {result['error']}")
        else:
            print(f"{dataset.upper():10s}: {result['accuracy']:.4f} accuracy ({result['epochs']} epochs)")
    
    print(f"\nAll results saved to: result/stage_5_result/")
    print("Files generated:")
    for dataset in datasets:
        if dataset in results_summary and 'error' not in results_summary[dataset]:
            print(f"  - {dataset}_learning_curve.png")
            print(f"  - {dataset}_report.txt")
            print(f"  - {dataset}_gcn_0 (pickle file)")


if __name__ == '__main__':
    main() 