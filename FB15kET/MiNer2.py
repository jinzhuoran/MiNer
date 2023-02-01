import torch
import torch.nn as nn

from MHA import MHA


class MiNer2(nn.Module):
    def __init__(self, hidden_dim, num_nodes, num_rels, num_types, alpha=0.3, beta=0.3):
        super(MiNer2, self).__init__()
        self.embedding_dim = hidden_dim
        self.embedding_range = 10 / self.embedding_dim
        self.num_rels = num_rels
        self.alpha = alpha
        self.beta = beta

        self.MulLayer = MulLayer(self.alpha)
        self.AGGLayer = AGGLayer()
        self.MTILayer = MTILayer(self.embedding_dim, num_types, self.beta)
        self.entity = nn.Parameter(torch.randn(num_nodes, self.embedding_dim))
        nn.init.uniform_(tensor=self.entity, a=-self.embedding_range, b=self.embedding_range)
        self.relation = nn.Parameter(torch.randn(num_rels, self.embedding_dim))
        nn.init.uniform_(tensor=self.relation, a=-self.embedding_range, b=self.embedding_range)
        self.device = torch.device('cuda')

    def forward(self, blocks):
        src1 = torch.index_select(self.entity, 0, blocks[0].srcdata['id'])
        etype1 = blocks[0].edata['etype']
        relations1 = torch.index_select(self.relation, 0, etype1 % self.num_rels)
        relations1[etype1 >= self.num_rels] = relations1[etype1 >= self.num_rels] * -1
        agg = self.AGGLayer(blocks[0], src1, relations1)
        src2 = torch.index_select(self.entity, 0, blocks[1].srcdata['id'])
        etype2 = blocks[1].edata['etype']
        relations2 = torch.index_select(self.relation, 0, etype2 % self.num_rels)
        relations2[etype2 >= self.num_rels] = relations2[etype2 >= self.num_rels] * -1
        output2 = self.MTILayer(blocks[1], src2, relations2, agg).sigmoid()
        return output2


class MTILayer(nn.Module):
    def __init__(self, embedding_dim, num_types, beta):
        super(MTILayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_types = num_types
        self.fc = nn.Linear(embedding_dim, num_types)
        self.mha1 = MHA(5, 1.0)
        self.mha2 = MHA(5, 1.0)
        self.beta = beta

    def message_func(self, edges):
        h_embedding = edges.src['h']
        r_embedding = edges.data['h']
        agg = edges.src['agg']
        return {'msg': h_embedding + r_embedding, 'agg': agg + r_embedding}

    def reduce_func(self, nodes):
        msg = nodes.mailbox['msg']
        agg = nodes.mailbox['agg']

        msg1 = self.fc(torch.relu(msg))
        msg2 = self.fc(torch.relu(msg.mean(1, keepdim=True)))
        predict1 = torch.cat([msg1, msg2], dim=1)

        agg1 = self.fc(torch.relu(agg))
        agg2 = self.fc(torch.relu(agg.mean(1, keepdim=True)))
        predict2 = torch.cat([agg1, agg2], dim=1)

        predict = self.beta * self.mha1(predict1) + (1 - self.beta) * self.mha2(predict2)
        return {'predict': predict}

    def forward(self, graph, src_embedding, edge_embedding, agg):
        assert len(edge_embedding) == graph.num_edges(), print('every edge should have a type')
        with graph.local_scope():
            graph.srcdata['h'] = src_embedding
            graph.srcdata['agg'] = agg
            graph.edata['h'] = edge_embedding
            graph.update_all(self.message_func, self.reduce_func)
            return graph.dstdata['predict']


class AGGLayer(nn.Module):
    def __init__(self):
        super(AGGLayer, self).__init__()

    def message_func(self, edges):
        h_embedding = edges.src['h1']
        r_embedding = edges.data['h1']
        return {'msg1': h_embedding + r_embedding}

    def reduce_func(self, nodes):
        msg = nodes.mailbox['msg1']
        agg = msg.mean(1)
        return {'agg': agg}

    def forward(self, graph, src_embedding, edge_embedding):
        assert len(edge_embedding) == graph.num_edges(), print('every edge should have a type')
        with graph.local_scope():
            graph.srcdata['h1'] = src_embedding
            graph.edata['h1'] = edge_embedding
            graph.update_all(self.message_func, self.reduce_func)
            return graph.dstdata['agg']


class MulLayer(nn.Module):
    def __init__(self, alpha=0.3):
        super(MulLayer, self).__init__()
        self.alpha = alpha

    def message_func(self, edges):
        h_embedding = edges.src['h1']
        r_embedding = edges.data['h1']
        return {'msg1': h_embedding + r_embedding}

    def reduce_func(self, nodes):
        h = nodes.data['h1']
        msg = nodes.mailbox['msg1']
        agg = msg.mean(1)
        return {'agg': self.alpha * h + (1 - self.alpha) * agg}

    def forward(self, graph, src_embedding, dst_embedding, edge_embedding):
        assert len(edge_embedding) == graph.num_edges(), print('every edge should have a type')
        with graph.local_scope():
            graph.srcdata['h1'] = src_embedding
            graph.edata['h1'] = edge_embedding
            graph.dstdata['h1'] = dst_embedding

            graph.update_all(self.message_func, self.reduce_func)
            return graph.dstdata['agg']
