'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        # Compute all four metrics
        acc   = accuracy_score(y_true, y_pred)
        prec  = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec   = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1    = f1_score(y_true, y_pred, average='macro', zero_division=0)

        return {
            'accuracy':    acc,
            'precision':   prec,
            'recall':      rec,
            'f1_score':    f1
        }
        