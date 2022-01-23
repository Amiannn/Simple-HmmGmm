import numpy as np

from hmm.hmm import HMM


def train(model, observation):
    
    ...

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

    hmms = [HMM(5) for i in range(10)]

    hmms[0].train(train_A, 10)
    hmms[1].train(train_B, 10)
    
    scoreAA = np.sum([hmms[0].viterbi(test)[0] for test in test_A]) / 10
    scoreAB = np.sum([hmms[0].viterbi(test)[0] for test in test_B]) / 10

    scoreBA = np.sum([hmms[1].viterbi(test)[0] for test in test_A]) / 10
    scoreBB = np.sum([hmms[1].viterbi(test)[0] for test in test_B]) / 10

    print(scoreAA)
    print(scoreAB)

    print(scoreBA)
    print(scoreBB)