import numpy as np
import utils, compute
from itertools import combinations_with_replacement


def get_combination(letters, m):
    comb = combinations_with_replacement(letters, m)
    # for i in comb:
    #     i = list(i)
    return [list(i) for i in comb]


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
    
    m = 5
    combs = get_combination(letters, m)
    
    print(find_max(features[:3], combs, W, T))



