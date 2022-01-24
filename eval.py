import os
import numpy as np

from processor import dataloader
from hmm.model import ClassificationModel


def eval(model, X, Y):
    predictions = model.predict(X)
    accuracy    = 0
    for pred, truth in zip(predictions, Y):
        # print('{}, {}'.format(pred, truth))
        if pred[0][0] == truth:
            accuracy += 1
    return accuracy / len(Y)
    
if __name__ == '__main__':
    dataset_path = './datasets/digit-recogntion'
    eval_path   = os.path.join(dataset_path, 'test')

    model_path = os.path.join('outputs', 'mymodel')
    
    model = ClassificationModel(states_length=10, mixtures=5, num_labels=10)
    model.load(model_path)
    
    X, Y  = dataloader(eval_path)

    acc = eval(model, X, Y)
    print('accuracy: {}'.format(acc))