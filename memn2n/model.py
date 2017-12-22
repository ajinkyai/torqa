import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
torch.manual_seed(0)


class MemN2N(nn.Module):
    def __init__(self, settings):
        super(MemN2N, self).__init__()
        self.settings = settings
        embedding_dim = settings["embedding_dim"]
        max_hops = settings['max_hops']
        num_vocab = settings['num_vocab']

        max_story_size = settings['max_story_size']
        # List of embedding matrices. This list will contain A, B, C
        # and W. We are using a single list because of the extensive
        # parameter sharing.
        self.embeddings= [nn.Embedding(num_vocab, embedding_dim, padding_idx=0)
                                            for _ in range(max_hops+1)]
        for emb in self.embeddings:
            emb.weight.data.normal_(0, 0.1)
        self.embedding_weights = nn.ParameterList([e.weight for e in self.embeddings])
        # For T_A and T_C
        self.T_embeddings = nn.ParameterList([nn.Parameter(torch.randn(max_story_size,
                                                                       embedding_dim).normal_(0, 0.1))
                                              for _ in range(max_hops+1)])

        self.train_loader = None
        self.test_loader = None
        self.optim = None
        self.scheduler = None
        self.ce_fn = None


    def position_encode(self, m):
        # Dim of m is 32, 27, 7, 20
        if m.dim() == 4:
            J = m.size(2)   # size of sentence. No. of words in sentence.
            d = m.size(3)   # embedding dim
        else:
            J = m.size(2-1)   # size of sentence. No. of words in sentence.
            d = m.size(3-1)   # embedding dim

        jmat = torch.arange(1, J + 1).unsqueeze(1).expand(J, d) / J
        kmat = torch.arange(1, d + 1).unsqueeze(0).expand(J, d) / d
        t1 = 1 - jmat
        t2 = kmat * (1 - 2*jmat)
        l = t1 - t2
        l = Variable(l)
        if self.settings['cuda']:
            l.cuda()
        rv = l.expand_as(m) * m
        return rv

    def forward(self, story, query):
        max_hops = self.settings['max_hops']
        B = self.embeddings[0]

        u_list = []
        u1 = B(query)
        # u1 = self.position_encode(u1)
        u_list.append(torch.sum(u1, 1))
        for hop in range(max_hops):
            A  = self.embeddings[hop]
            TA = self.T_embeddings[hop]
            C  = self.embeddings[hop + 1]
            TC = self.T_embeddings[hop + 1]
            orig_story_size = story.size()
            m  = A(story.view(story.size(0), -1))
            m  = m.view(orig_story_size + (m.size(-1), ))
            # encode here. m dim is (32, 26, 7, 20)
            # m = self.position_encode(m)
            m  = torch.sum(m, 2)
            m += TA

            uTm = m * u_list[-1].unsqueeze(1).expand_as(m)
            uTm = torch.sum(uTm, 2)
            proba = F.softmax(uTm)  # b x i x edim

            c  = C(story.view(story.size(0), -1))
            c = c.view(orig_story_size + (c.size(-1),))
            # c = self.position_encode(c)
            c  = torch.sum(c, 2)
            c += TC

            pc = proba.unsqueeze(2).expand_as(c) * c
            o = torch.sum(pc, 1)

            # next u
            u_next = u_list[-1] + o
            u_list.append(u_next)

        W = self.embeddings[-1]
        ahat = u_list[-1]@W.weight.t()
        return ahat, F.softmax(ahat)



    def fit(self, train_data):
        config = self.settings
        max_epochs = config['max_epochs']
        decay_interval = self.settings['decay_interval']
        decay_ratio = self.settings['decay_ratio']
        lambda_lr_update = lambda epoch: decay_ratio ** max(0, epoch // decay_interval)
        self.optim = torch.optim.SGD(self.parameters(), lr=config['lr'])
        self.scheduler = LambdaLR(self.optim, lr_lambda=[lambda_lr_update])
        self.ce_fn = nn.CrossEntropyLoss(size_average=False)
        if config['cuda']:
            self.ce_fn   = self.ce_fn.cuda()
            self.mem_n2n = self.mem_n2n.cuda()

        self.train_loader = DataLoader(train_data,
                                       batch_size=config['batch_size'],
                                       num_workers=1,
                                       shuffle=True)

        for epoch in range(max_epochs):
            # loss = self._train_single_epoch(epoch)

            for step, (story, query, answer) in enumerate(self.train_loader):
                story = Variable(story)
                query = Variable(query)
                answer = Variable(answer)

                if config['cuda']:
                    story = story.cuda()
                    query = query.cuda()
                    answer = answer.cuda()

                self.optim.zero_grad()
                loss = self.ce_fn(self(story, query)[0], answer)
                loss.backward()
                nn.utils.clip_grad_norm(self.parameters(), 40.0)

                self.optim.step()
            self.scheduler.step()

            if (epoch+1) % 10 == 0:
                train_acc = self.tst(train_data)
                print('epoch: {}, loss: {}, train acc: {}, lr:{}'.format(epoch+1, loss.data[0], train_acc, self.scheduler.get_lr()))
        print(train_acc)

    def tst(self, data, train=False):
        correct = 0
        config = self.settings
        if not train:
            self.test_loader = DataLoader(data,
                                       batch_size=config['batch_size'],
                                       num_workers=1,
                                       shuffle=False)
        loader = self.train_loader if data == "train" else self.test_loader
        for step, (story, query, answer) in enumerate(loader):
            story = Variable(story)
            query = Variable(query)
            answer = Variable(answer)

            if self.settings['cuda']:
                story = story.cuda()
                query = query.cuda()
                answer = answer.cuda()

            pred_prob = self(story, query)[1]
            pred = pred_prob.data.max(1)[1]
            correct += pred.eq(answer.data).sum()

        acc = correct / len(loader.dataset)
        return acc

