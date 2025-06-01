'''
Concrete SettingModule class for Graph Node Classification
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
from code.base_class.setting import setting


class Setting_Graph_Node_Classification(setting):
    """
    Setting class for graph-based node classification experiments
    """
    
    def __init__(self, sName=None, sDescription=None):
        super(Setting_Graph_Node_Classification, self).__init__(sName, sDescription)
        self.dataset_name = None
        
    def load_run_save_evaluate(self):
        """
        Main pipeline: load graph data -> run GCN -> save results -> evaluate
        """
        
        # Load graph dataset
        print(f"Loading {self.dataset.dataset_name} dataset...")
        loaded_data = self.dataset.load()
        
        # Prepare data for GCN method
        self.method.data = loaded_data
        self.dataset_name = self.dataset.dataset_name
        
        # Run GCN method
        print(f"Running GCN on {self.dataset_name}...")
        learned_result = self.method.run()
        
        # Save raw result
        self.result.data = learned_result
        self.result.save()
        
        # Save learning curves plot
        curves_path = f"result/stage_5_result/{self.dataset_name}_learning_curve.png"
        self.method.save_learning_curves(self.dataset_name, curves_path)
        
        # Save detailed report
        self.save_detailed_report(learned_result)
        
        # Evaluate accuracy
        self.evaluate.data = {
            'true_y': learned_result['true_y'], 
            'pred_y': learned_result['pred_y']
        }
        
        accuracy = self.evaluate.evaluate()
        print(f"Final test accuracy for {self.dataset_name}: {accuracy:.4f}")
        
        return accuracy, learned_result

    def save_detailed_report(self, result):
        """
        Save detailed performance report to text file
        """
        report_path = f"result/stage_5_result/{self.dataset_name}_report.txt"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(f"GCN Performance Report - {self.dataset_name.upper()} Dataset\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Method: Graph Convolutional Network (GCN)\n\n")
            
            f.write("Final Results:\n")
            f.write(f"Test Accuracy: {result['test_accuracy']:.4f}\n\n")
            
            f.write("Training Summary:\n")
            f.write(f"Total Epochs: {len(result['train_losses'])}\n")
            f.write(f"Final Train Loss: {result['train_losses'][-1]:.4f}\n")
            f.write(f"Final Train Accuracy: {result['train_accuracies'][-1]:.4f}\n")
            f.write(f"Final Validation Loss: {result['val_losses'][-1]:.4f}\n")
            f.write(f"Final Validation Accuracy: {result['val_accuracies'][-1]:.4f}\n\n")
            
            f.write(f"Best Validation Accuracy: {max(result['val_accuracies']):.4f}\n")
            f.write(f"Best Training Accuracy: {max(result['train_accuracies']):.4f}\n\n")
            
            # Additional stats
            f.write("Model Configuration:\n")
            f.write(f"Hidden Dimensions: {self.method.hidden_dim}\n")
            f.write(f"Dropout Rate: {self.method.dropout}\n")
            f.write(f"Learning Rate: {self.method.learning_rate}\n")
            f.write(f"Weight Decay: {self.method.weight_decay}\n")
            
        print(f"Detailed report saved to {report_path}") 