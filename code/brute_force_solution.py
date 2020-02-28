import numpy as np
import utils, compute
from itertools import combinations_with_replacement

def find_permutations(letters, m):
    permuts = [[]]
    for i in range(0, m):
        permuts_tmp = []

        for j in permuts:
            for l in letters:
                tmp = j[:]
                tmp += [l]

                permuts_tmp.append(tmp)
        permuts = permuts_tmp

    return permuts


def find_max(x, combs, W, T):
    max_val, ans = 0, 0
    for c in combs:
        val = compute.compute_probability(x, c, W, T)
        if val > max_val:
            max_val = val
            ans = c
    return ans


if __name__ == "__main__":
    X, W, T = utils.load_decode_input()
    label, features = utils.read_data_struct()

    letters = [i for i in range(0, 26)]

    m = 3
    permuts = find_permutations(letters, m)

    print(find_max(features[:3], permuts, W, T))



