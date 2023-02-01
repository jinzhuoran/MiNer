import argparse

from utils import *


def e2t(path, e2id, t2id):
    entity2type = {}
    with open(path) as f:
        for line in f:
            entity, type = line.strip('\n').split('\t')
            if e2id[entity] not in entity2type:
                entity2type[e2id[entity]] = []
            entity2type[e2id[entity]].append(t2id[type])
    return entity2type


def occurrence(data_path, req_p=True):
    if os.path.exists(os.path.join(data_path, 'p.pt')) and req_p:
        p = torch.load(os.path.join(data_path, 'p.pt'))
        return None, p
    e2id = read_id(os.path.join(data_path, 'entities.tsv'))
    t2id = read_id(os.path.join(data_path, 'types.tsv'))
    entity2type = e2t(os.path.join(data_path, 'ET_train.txt'), e2id, t2id)
    counts = [[0] * len(t2id) for _ in range(len(t2id))]
    for index, value in entity2type.items():
        for i in range(len(value)):
            for j in range(i + 1, len(value)):
                counts[value[i]][value[j]] += 1
                counts[value[j]][value[i]] += 1
    counts = torch.FloatTensor(counts)
    count = len(counts)
    temp = counts + 0.000001
    p = torch.FloatTensor(count, count)
    for i in range(count):
        p[i] = temp[i] / temp[i].sum()
    torch.save(p, os.path.join(data_path, 'p.pt'))
    torch.save(counts, os.path.join(data_path, 'counts.pt'))
    return counts, p


def mining(e2t, index, p):
    r = torch.zeros(p.size()[0], dtype=torch.float)
    for i in e2t[index]:
        r = r + p[i]
    r[e2t[index]] = 0
    r = r / torch.sum(r)
    return r


def eval(data_path):
    counts = torch.load(os.path.join(data_path, 'counts.pt'))
    e2id = read_id(os.path.join(data_path, 'entities.tsv'))
    r2id = read_id(os.path.join(data_path, 'relations.tsv'))
    r2id['type'] = len(r2id)
    t2id = read_id(os.path.join(data_path, 'types.tsv'))
    g, train_label, all_true, train_id, valid_id, test_id = \
        load_graph(data_path, e2id, r2id, t2id)
    predict = torch.zeros(len(e2id), len(t2id), dtype=torch.float)

    train = e2t(os.path.join(data_path, 'ET_train.txt'), e2id, t2id)
    test = e2t(os.path.join(data_path, 'ET_test.txt'), e2id, t2id)

    for index, value in test.items():
        if index not in train:
            continue
        rank = mining(train, index, counts)
        predict[index] = rank
    MRR = evaluate(os.path.join(data_path, 'ET_test.txt'), predict, all_true, e2id, t2id)
    return MRR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='YAGO43kET')
    parser.add_argument('--save_dir', type=str, default='save')
    args, _ = parser.parse_known_args()
    args = vars(args)
    data_path = os.path.join(args['data_dir'], args['dataset'], 'clean')
    set_logger(args)
    MRR = eval(data_path)
