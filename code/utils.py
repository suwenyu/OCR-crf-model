def read_data(filename):
    # read data from file, return list of tuples, 
    # and each tuple contains word(list of chars) and img features([128, 1]*len(word))
    # ex. (['a', 'k', 'e'], [[1, 0, 0 ... ], [0, 1, 0 ...], [0, 0, 0 ...]])
    data = []
    tmp_id = 1
    label = []
    features = []
    
    f = open(filename, 'r')
    for line in f:
        line_list = line.rstrip().split(' ')
        _id, letter, word_id = line_list[0], line_list[1], int(line_list[3])
        feature = line_list[5:]
        
        if word_id != tmp_id:
            data.append((label, features))
            tmp_id = word_id
            label = []
            features = []

        label.append(letter)
        features.append(feature)
    return data


if __name__ == '__main__':
    train_data = read_data('../data/test.txt')
    test_data = read_data('../data/test.txt')