'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = '/Users/ameensalim/ECS189G/project/ECS189G_Winter_2022_Source_Code_Template/data/stage_2_data/'
    dataset_source_file_name = 'train.csv'
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X = []
        y = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(' ')]
            X.append(elements[:-1])
            y.append(elements[-1])
        f.close()
        return {'X': X, 'y': y}