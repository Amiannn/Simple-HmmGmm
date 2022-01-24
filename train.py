import os
import numpy as np

from hmm.model import ClassificationModel

def train(model, X, Y, epochs):
    model.train(X, Y, epochs)
    return model
    
if __name__ == '__main__':
    observationsA = [
        ((np.sin(np.linspace(np.random.random(1)[0], np.pi + np.random.random(1)[0], 100)) + 1) / 2).reshape(
            -1, 1
        ) for i in range(20)
    ]

    observationsB = [
        ((np.cos(np.linspace(np.random.random(1)[0], np.pi + np.random.random(1)[0], 100)) + 1) / 2).reshape(
            -1, 1
        ) for i in range(20)
    ]

    train_A = observationsA[:10]
    train_B = observationsB[:10]

    test_A  = observationsA[10:]
    test_B  = observationsB[10:]

    model = ClassificationModel(states_length=5, num_labels=2)
    
    train(model, [train_A, train_B], ['A', 'B'], 10)
    save_path = os.path.join('outputs', 'mymodel')
    model.save(save_path)
