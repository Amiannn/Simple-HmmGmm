import numpy as np

from hmm.hmm import HMM


hmm = HMM(5)

x = np.sin(np.linspace(0, np.pi, 100)) + 1

xs = np.array_split(x, hmm.states.length)

for index, x in enumerate(xs):
    x = x.reshape(-1, 1)
    hmm.states.emission(index).fit(x)

sequence = hmm.generate(100)


x = []

for state, em in sequence:
    x.append(em[0])
    print(state, end='')
print()

x  = np.array(x).reshape(-1, 1)
print(x.shape)

print(hmm.viterbi(x))