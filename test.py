import os
import numpy as np

from hmm.model import ClassificationModel

def test(model, X):
    result = model.predict(X)
    return result

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

    model_path = os.path.join('outputs', 'mymodel')
    
    model = ClassificationModel(states_length=5, num_labels=2)
    model.load(model_path)

    print(test(model, [test_B[0]]))