'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        """
        Returns a dict with:
          - accuracy
          - macro-precision
          - macro-recall
          - macro-f1
        """
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']
        print('evaluating performance...')
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred,
            average="macro",
            zero_division=0
        )
        return {
            "accuracy":        acc,
            "precision_macro": p,
            "recall_macro":    r,
            "f1_macro":        f1
        }