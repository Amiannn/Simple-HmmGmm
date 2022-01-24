import os
import numpy as np

from processor import dataloader
from hmm.model import ClassificationModel


def train(model, X, Y, epochs):
    model.train(X, Y, epochs)
    return model
    
if __name__ == '__main__':
    dataset_path = './datasets/digit-recogntion'
    train_path   = os.path.join(dataset_path, 'train')

    model = ClassificationModel(states_length=10, mixtures=2, num_labels=10)
    X, Y  = dataloader(train_path)

    train(model, X, Y, epochs=20)

    save_path = os.path.join('outputs', 'mymodel')
    model.save(save_path)
