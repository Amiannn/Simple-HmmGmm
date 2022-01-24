import os
import json
import numpy as np

from hmm.hmm import HMM

class ClassificationModel():
    def __init__(self, states_length, mixtures, num_labels):
        self.reset_model(states_length, mixtures, num_labels, None)

    def reset_model(self, states_length, mixtures, num_labels, index2label):
        self.hmms = [HMM(states_length, mixtures) for _ in range(num_labels)]
        self.index2label = index2label
        self.mixtures    = mixtures

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        config = {
            'states_length': self.hmms[0].states.length,
            'mixtures'     : self.mixtures,
            'num_labels'   : len(self.hmms),
            'index2label'  : self.index2label
        }
        # save config json
        with open(os.path.join(path, 'cfg.json'), 'w', encoding='utf8') as fr:
            json.dump(config, fr, ensure_ascii=False, indent=4)

        # save hmm-gmm
        for i in range(len(self.hmms)):
            self.hmms[i].save(os.path.join(path, 'hmm_{}'.format(i)))

    def load(self, path):
        config = None
        # load config file
        with open(os.path.join(path, 'cfg.json'), 'r', encoding='utf8') as fr:
            config = json.load(fr)
            
        # reset model
        self.reset_model(
            config['states_length'],
            config['mixtures'],
            config['num_labels'],
            config['index2label']
        )
        # set hmms
        for i in range(len(self.hmms)):
            self.hmms[i].load(os.path.join(path, 'hmm_{}'.format(i)))

    def train(self, X, Y, epochs):
        observations_dict = dict([[y, []] for y in set(Y)])
        self.index2label  = dict([[i, k]  for i, k in zip(range(len(observations_dict)), observations_dict.keys())])

        for index in range(len(Y)):
            label = Y[index]
            observations_dict[label].append(X[index])

        for index, label in enumerate(observations_dict):
            observations = observations_dict[label]
            print('{}label:[{}]{}'.format('='*10, label, '='*10))
            self.hmms[index].train(observations, epochs)

    def predict(self, X):
        result = []
        for x in X:
            probs = []
            # find the max prob hmm
            for index in range(len(self.hmms)):
                prob = self.hmms[index].viterbi(x)[0]
                probs.append([prob, index])
            probs = sorted(probs, reverse=True, key=lambda p: p[0])
            predict_label = self.index2label[str(probs[0][1])]
            result.append(predict_label)
        return result