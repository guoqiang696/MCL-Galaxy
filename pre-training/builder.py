# import torch
# import torch.nn as nn
# import torchvision.models as models
# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))
#
#
# class MoCo(nn.Module):
#     def __init__(self):
#         """
#         dim: feature dimension (default: 128)
#         K: queue size; number of negative keys (default: 65536)
#         m: moco momentum of updating key encoder (default: 0.999)
#         T: softmax temperature (default: 0.07)
#         """
#         super(MoCo, self).__init__()
#         # create the queue
#         self.register_buffer("queue", torch.randn(4, 8))
#
#         self.queue = nn.functional.normalize(self.queue, dim=0)
#         # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
#         # print(self.queue_ptr)
#         # print(type(self.queue_ptr))


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import numpy
import time
import random
import numpy as np


class CreateRandomPage:
    def __init__(self, begin, end, needcount):
        self.begin = begin
        self.end = end
        self.needcount = needcount
        self.resultlist = []
        self.count = 0

    def createrandompage(self):
        tempInt = random.randint(self.begin, self.end)
        if (self.count < self.needcount):
            if (tempInt not in self.resultlist):
                self.resultlist.append(tempInt)
                self.count += 1
            return self.createrandompage()
        return self.resultlist


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, base_queue, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 16384)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.queue = base_queue
        self.encoder_q = base_encoder(num_classes=128)
        self.T = T
        a = base_queue.numpy()
        print("Primitive matrix：", a.shape)
        print("a:", a)
        time1 = time.time()
        D = numpy.cov(a)
        time2 = time.time()
        print("Time to calculate the covariance：", time2 - time1)
        print("invD.shape:", D.shape)
        print("The covariance matrix is：", D)
        invD = numpy.linalg.inv(D)
        time3 = time.time()
        print("Compute the inverse of the covariance matrix：", time3 - time2)
        print("invD.shape:", invD.shape)
        print("The inverse of the covariance matrix is given by：", invD)
        self.invD = invD
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)

        for param in self.encoder_q.parameters():
            param.requires_grad = False  # not update by gradient

    def forward(self, im_q, output, target):
        """
        Input:
            im_q: a batch of query images
        Output:
            queue_logits: 1×N, every picture loss
        """
        # compute query features

        q = self.encoder_q(im_q)  # queries: NxC    256(batch_size) × 128(特征维度)
        # print("q.size():", q.size())      # q.size(): torch.Size([256, 128])
        # print("q", q)
        q = torch.nn.functional.normalize(q, dim=1)

        # Einstein sum is more intuitive
        # compute logits: NxK

        # print("self.queue.clone().detach().size():", self.queue.clone().detach().size())
        # print(self.queue.clone().detach())

        time1 = time.time()
        q = q.T
        my_list = []
        num = 64
        resultlist = random.sample(range(0, 16384), num)
        # print(resultlist)
        # print("q.shape[1]:", q.shape[1])
        for i in range(0, q.shape[1]):
            ans = 0.0
            for j in resultlist:
                tp = q[:, i].cpu() - self.queue[:, j]
                ans += numpy.sqrt(numpy.dot(numpy.dot(tp, self.invD), tp.T))
            ans /= num
            # print("type(target[i].numpy())", type(target[i].cpu().numpy()))
            # print("type(ans):", type(ans))
            # print((target[i].cpu().numpy(), ans))
            # v = target[i].cpu().numpy()

            # print([v, output[i], ans])
            my_list.append([i, output[i], ans])

        # for i in range(0, q.shape[1]):
        #     ans = 0.0
        #     for j in resultlist:
        #         ans += cos_sim(q[:, i].cpu(), self.queue[:, j])
        #     ans /= num
        #     v = target[i].cpu().numpy()
        #     # print([v, output[i], ans])
        #     my_list.append([v, output[i], ans])
        time2 = time.time()
        # queue_logits = torch.einsum('nc,ck->n', [q, self.queue.clone().detach()])
        # apply temperature
        # logits /= self.T
        # mmax = max(my_list)
        # mmin = min(my_list)
        # mmav = sum(my_list) / 256
        # print("my_list.index(mmax):", my_list.index(mmax))
        # print("my_list.index(mmin):", my_list.index(mmin))
        # print("max(my_list):", max(my_list))
        # print("min(my_list):", min(my_list))
        # print("mmav:", mmav)
        # print("mmax - mmin:", mmax - mmin)
        # print(my_list)
        result = sorted(my_list, key=lambda x: (x[2], x[0]))

        ind_list_2 = []
        for i in result[-64:-1]:
            ind = i[0]
            ind_list_2.append(ind)
        # print("ind_list_2:", ind_list_2)


        for i in ind_list_2:
            if abs(output[i][2]) < 1.0 or abs(output[i][3]) < 1.0:
                output[i][2] = -5.0
                output[i][3] = 5.0

        return output


def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a, b) / (a_norm * b_norm)
    return cos
