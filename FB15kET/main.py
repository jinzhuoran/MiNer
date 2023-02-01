import argparse

import torch.cuda

torch.cuda.set_device(3)
from dgl.dataloading import NodeDataLoader, MultiLayerFullNeighborSampler, MultiLayerNeighborSampler

from MiNer2 import MiNer2
from MiNer3 import MiNer3
from MiNer4 import MiNer4
from RGCN import RGCN
from CompGCN import CompGCN
from CET import CET
from utils import *
from loss import FNA, TCR
from statistics import occurrence
import time


def main(args):
    use_cuda = args['cuda'] and torch.cuda.is_available()
    data_path = os.path.join(args['data_dir'], args['dataset'], 'clean')
    save_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))
    save_path = os.path.join(args['save_dir'], args['dataset'], save_time)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # graph
    e2id = read_id(os.path.join(data_path, 'entities.tsv'))
    r2id = read_id(os.path.join(data_path, 'relations.tsv'))
    r2id['type'] = len(r2id)
    t2id = read_id(os.path.join(data_path, 'types.tsv'))
    num_entity = len(e2id)
    num_rels = len(r2id)
    num_types = len(t2id)
    num_nodes = num_entity + num_types
    g, train_label, all_true, train_id, valid_id, test_id = load_graph(data_path, e2id, r2id, t2id,
                                                                       args['load_ET'], args['load_KG'])
    if args['num_layers'] == 2:
        if args['neighbor_sampling']:
            train_sampler = MultiLayerNeighborSampler([args['neighbor_num1'], args['neighbor_num2']], replace=True)
        else:
            train_sampler = MultiLayerFullNeighborSampler(args['num_layers'])
        test_sampler = MultiLayerNeighborSampler([args['neighbor_num1'] * 5, -1], replace=True)
    elif args['num_layers'] == 3:
        if args['neighbor_sampling']:
            train_sampler = MultiLayerNeighborSampler(
                [args['neighbor_num1'], args['neighbor_num1'], args['neighbor_num2']], replace=True)
        else:
            train_sampler = MultiLayerFullNeighborSampler(args['num_layers'])
        test_sampler = MultiLayerNeighborSampler([args['neighbor_num1'] * 5, args['neighbor_num1'] * 5, -1],
                                                 replace=True)
    elif args['num_layers'] == 4:
        if args['neighbor_sampling']:
            train_sampler = MultiLayerNeighborSampler(
                [args['neighbor_num1'], args['neighbor_num1'], args['neighbor_num1'], args['neighbor_num2']],
                replace=True)
        else:
            train_sampler = MultiLayerFullNeighborSampler(args['num_layers'])
        test_sampler = MultiLayerNeighborSampler(
            [args['neighbor_num1'] * 5, args['neighbor_num1'] * 5, args['neighbor_num1'] * 5, -1], replace=True)
    else:
        if args['neighbor_sampling']:
            train_sampler = MultiLayerNeighborSampler([args['neighbor_num']] * args['num_layers'], replace=True)
        else:
            train_sampler = MultiLayerFullNeighborSampler(args['num_layers'])
        test_sampler = MultiLayerFullNeighborSampler(args['num_layers'])
    train_dataloader = NodeDataLoader(
        g, train_id, train_sampler,
        batch_size=args['train_batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=6
    )
    valid_dataloader = NodeDataLoader(
        g, valid_id, test_sampler,
        batch_size=args['test_batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=6
    )
    test_dataloader = NodeDataLoader(
        g, test_id, test_sampler,
        batch_size=args['test_batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=6
    )

    # model
    if args['model'] == 'MiNer':
        if args['num_layers'] == 2:
            model = MiNer2(args['hidden_dim'], num_nodes, num_rels, num_types, alpha=args['alpha'], beta=args['beta'])
        elif args['num_layers'] == 3:
            model = MiNer3(args['hidden_dim'], num_nodes, num_rels, num_types, alpha=args['alpha'], beta=args['beta'])
        elif args['num_layers'] == 4:
            model = MiNer4(args['hidden_dim'], num_nodes, num_rels, num_types, alpha=args['alpha'], beta=args['beta'])
    elif args['model'] == 'CET':
        model = CET(args, num_nodes, num_rels, num_types)
    elif args['model'] == 'RGCN':
        model = RGCN(args['hidden_dim'], num_nodes, num_rels, num_types, num_layers=args['num_layers'],
                     num_bases=args['num_bases'], activation=args['activation'], regularizer=args['regularizer'],
                     self_loop=args['selfloop'])
    elif args['model'] == 'CompGCN':
        model = CompGCN(num_nodes, num_rels * 2, num_types, args['hidden_dim'], num_layers=args['num_layers'],
                        activation=args['activation'], self_loop=args['selfloop'])
    else:
        raise ValueError('No such model')

    if use_cuda:
        device = torch.device('cuda')
        model = model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['occurrence']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'lr': args['lr']},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'lr': 1e-6}]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters)

    _, occ = occurrence(data_path)
    occ = occ.to(device)
    tcr = TCR(occurrence=occ, num_examp=num_entity, num_classes=num_types, lambd=args['lambd'], omega=args['omega'],
              gama=args['gama'])

    fna = FNA(args['eta'])
    # training
    max_valid_mrr = 0
    model.train()
    for epoch in range(args['max_epoch']):
        log = []
        for input_nodes, output_nodes, blocks in train_dataloader:
            label = train_label[output_nodes, :]
            if use_cuda:
                blocks = [b.to(torch.device('cuda')) for b in blocks]
                label = label.cuda()
            predict = model(blocks)

            if args['loss'] == 'TCR':
                label[predict > args['threshold']] = 1
                pos_loss, neg_loss = fna(predict, label)
                regular = tcr(output_nodes, predict, epoch)
                loss = pos_loss + neg_loss + regular
            elif args['loss'] == 'FNA':
                pos_loss, neg_loss = fna(predict, label)
                loss = pos_loss + neg_loss
            else:
                raise ValueError('loss %s is not defined' % args['loss'])

            log.append({
                "loss": loss.item(),
                "pos_loss": pos_loss.item(),
                "neg_loss": neg_loss.item(),
            })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = sum([_['loss'] for _ in log]) / len(log)
        avg_pos_loss = sum([_['pos_loss'] for _ in log]) / len(log)
        avg_neg_loss = sum([_['neg_loss'] for _ in log]) / len(log)
        logging.debug('epoch %d: loss: %f\tpos_loss: %f\tneg_loss: %f' %
                      (epoch, avg_loss, avg_pos_loss, avg_neg_loss))

        if epoch != 0 and epoch % args['valid_epoch'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pkl'))
            model.eval()
            with torch.no_grad():
                predict = torch.zeros(num_entity, num_types, dtype=torch.half)
                for input_nodes, output_nodes, blocks in valid_dataloader:
                    if use_cuda:
                        blocks = [b.to(torch.device('cuda')) for b in blocks]
                    predict[output_nodes] = model(blocks).cpu().half()
                valid_mrr = evaluate(os.path.join(data_path, 'ET_valid.txt'), predict, all_true, e2id, t2id)
            model.train()
            if valid_mrr < max_valid_mrr:
                if step < args['max_step']:
                    step += 1
                else:
                    logging.debug('early stop')
                    break
            else:
                step = 0
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pkl'))
                max_valid_mrr = valid_mrr

    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pkl')))
        model.eval()
        predict = torch.zeros(num_entity, num_types, dtype=torch.half)

        for input_nodes, output_nodes, blocks in test_dataloader:
            if use_cuda:
                blocks = [b.to(torch.device('cuda')) for b in blocks]
            predict[output_nodes] = model(blocks).cpu().half()
        evaluate(os.path.join(data_path, 'ET_test.txt'), predict, all_true, e2id, t2id)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MiNer')
    parser.add_argument('--loss', type=str, default='TCR')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--neighbor_num1', type=int, default=30)
    parser.add_argument('--neighbor_num2', type=int, default=10)

    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--eta', type=float, default=4.0)
    parser.add_argument('--lambd', type=float, default=3)
    parser.add_argument('--omega', type=float, default=0.7)
    parser.add_argument('--gama', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.95)

    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='FB15kET')
    parser.add_argument('--load_ET', action='store_true', default=True)
    parser.add_argument('--load_KG', action='store_true', default=True)
    parser.add_argument('--neighbor_sampling', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_step', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--valid_epoch', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.5)

    parser.add_argument('--regularizer', type=str, default='basis')
    parser.add_argument('--num_bases', type=int, default=-1)
    parser.add_argument('--activation', type=str, default='none')
    parser.add_argument('--selfloop', action='store_true', default=False)

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        params = vars(get_params())
        set_logger(params)
        main(params)
    except Exception as e:
        logging.exception(e)
        raise
