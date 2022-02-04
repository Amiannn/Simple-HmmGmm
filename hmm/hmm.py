import os
import numpy as np

from hmm.states import States

class HMM():
    def __init__(self, states_length, mixtures):
        self.states = States(states_length, mixtures)
        np.seterr(divide = 'ignore')

    def _logaddexp(self, a):
        value = 0.0
        for i in range(a.shape[0]):
            value = np.logaddexp(a[i], value)
        return value

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        self.states.save(path)

    def load(self, path):
        self.states.load(path)

    def generate(self, sequence_length):
        pi = np.exp(self.states.initials_log)
        a  = self.states.transition_matrix
        b  = [self.states.emission(i) for i in range(self.states.length)]
        
        states_index = np.arange(self.states.length)
        states_seq   = []

        # start state
        state = np.random.choice(states_index, 1, p=pi)[0]
        # start state's emission
        emiss = b[state].sample(1)[0]
        states_seq.append([state, emiss])

        for t in range(1, sequence_length):
            # next state and emission
            state = np.random.choice(states_index, 1, p=a[state, :])[0]
            emiss = b[state].sample(1)[0]
            states_seq.append([state, emiss])

        return states_seq

    def forward(self, observation):
        obs_length = observation.shape[0]

        # initial probability
        pi_log = self.states.initials_log

        # transition probability
        a_log  = np.log(self.states.transition_matrix)

        # emission probability
        b_log  = np.array([
            self.states.emission(i).score_samples(
                observation
            ) for i in range(self.states.length)
        ])

        alpha_log  = np.zeros([self.states.length, obs_length])
        
        # when t equal to 0
        alpha_log[:, 0] = pi_log + b_log[:, 0]

        # when t bigger than 0
        for t in range(1, obs_length):
            for j in range(self.states.length):
                alpha_log[j, t] = self._logaddexp(alpha_log[:, t - 1] + a_log[:, j]) + b_log[j, t]

        # forward probability
        prob_log = self._logaddexp(alpha_log[:, -1])
        return prob_log, alpha_log

    def backward(self, observation):
        obs_length = observation.shape[0]

        # initial probability
        pi_log = self.states.initials_log

        # transition probability
        a_log  = np.log(self.states.transition_matrix)

        # emission probability
        b_log  = np.array([
            self.states.emission(i).score_samples(
                observation
            ) for i in range(self.states.length)
        ])

        beta_log  = np.zeros([self.states.length, obs_length])
        
        # when t equal to T - 1
        beta_log[:, -1] = 1

        # when t smaller than T - 1
        for t in range(obs_length - 2, -1, -1):
            for i in range(self.states.length):
                beta_log[i, t] = self._logaddexp(beta_log[:, t + 1] + a_log[i, :] + b_log[:, t + 1])

        # backward probability
        beta_log[:, 0] = pi_log + b_log[:, 0]
        prob_log = self._logaddexp(beta_log[:, 0])
        return prob_log, beta_log


    def viterbi(self, observation):
        obs_length = observation.shape[0]

        # initial probability
        pi_log = self.states.initials_log

        # transition probability
        a_log  = np.log(self.states.transition_matrix)

        # emission probability
        b_log  = np.array([
            self.states.emission(i).score_samples(
                observation
            ) for i in range(self.states.length)
        ])

        # when t equal to 0
        delta_log = pi_log + b_log[:, 0]
        # back pointers
        bp_matrix = np.zeros([self.states.length, obs_length])

        for t in range(obs_length):
            # tranpose A, more easy to caluate delta
            delta_log = delta_log + a_log.T
            
            max_delta_a     = np.max(delta_log, axis=1)
            bp_matrix[:, t] = np.argmax(delta_log, axis=1)
            
            # shape (n, )
            delta_log   = max_delta_a + b_log[:, t]

        prob_log = delta_log[-1]
        
        # state sequence
        state_q     = np.zeros(obs_length, dtype=np.int32)
        state_q[-1] = self.states.length - 1

        # back tracking
        for t in range(obs_length - 1, 0, -1):
            state_q[t - 1] = bp_matrix[state_q[t], t]

        return prob_log, state_q

    def train(self, observations, epoch):
        # transistion matrix
        a = self.states.transition_matrix
        # emission prob
        b = self.states.emission

        # get uniform state segmentations
        states_sequences = [[] for s in range(self.states.length)]
        for observation in observations:
            states_sequence = np.array_split(observation, self.states.length)
            for s in range(self.states.length):
                states_sequences[s].append(states_sequence[s])

        # initialize emission (gaussian mixture model)
        for s in range(self.states.length):
            states_sequence = np.concatenate(states_sequences[s], axis=0)
            b(s).fit(states_sequence)

        # training HMM and GMM
        for e in range(epoch):
            # decode log prob
            decode_log_prob = 0.0
            # record viterbi decode path
            states_hists = np.zeros(self.states.length, dtype=np.int32)
            # get viterbi state segmentations
            states_sequences = [[] for s in range(self.states.length)]
            for observation in observations:
                decode_prob, decode_states = self.viterbi(observation)
                states_hist  = np.histogram(decode_states, bins=np.arange(self.states.length + 1))[0]
                states_accum = np.add.accumulate(states_hist)
                
                # new state alignment
                for s in range(self.states.length):
                    start = states_accum[s] - states_hist[s]
                    end   = states_accum[s]
                    states_sequences[s].append(observation[start:end, :])
                # accumulate
                states_hists    = states_hists + states_hist
                decode_log_prob = decode_log_prob + decode_prob
            
            # update emission (gaussian mixture model)
            for s in range(self.states.length):
                states_sequence = np.concatenate(states_sequences[s], axis=0)
                b(s).fit(states_sequence)

            # update transition
            for s in range(self.states.length - 1):
                a[s, s + 1] = len(observations) / states_hists[s]
                a[s, s]     = 1 - a[s, s + 1]

            print('epoch: [{}], log_prob:{}'.format(e, decode_log_prob))
        return decode_log_prob