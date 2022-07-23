import jittor
from jittor import nn
import math

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def execute(self, x):
        norm = x.pow(self.power).sum(1, keepdims=True).pow(1. / self.power)
        out = x.divide(norm + 1e-7)
        return out



# class PatchNCELoss(nn.Module):
class BidirectionalNCE1(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_dtype = jittor.bool
        self.cosm = math.cos(0.25)
        self.sinm = math.sin(0.25)

    def execute(self, feat_q, feat_k):

        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = nn.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        cosx = l_pos
        sinx = jittor.sqrt(1.0 - jittor.pow(cosx, 2))
        l_pos = cosx * self.cosm - sinx * self.sinm

        batch_dim_for_bmm = int(batchSize / 64)
        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = nn.bmm(feat_q, feat_k.transpose(2, 1))

        # just fill the diagonal with very small number, which is exp(-10)
        diagonal = jittor.init.eye(npatches, dtype=self.mask_dtype)[None, :, :]
        #l_neg_curbatch.masked_fill_(diagonal, -10.0)
        diagonal = diagonal.repeat([l_neg_curbatch.shape[0],1,1])
        l_neg_curbatch = jittor.masked_fill(l_neg_curbatch, diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = jittor.concat((l_pos, l_neg), dim=1) / 0.05
        loss = nn.cross_entropy_loss(out, jittor.zeros(out.size(0), dtype='int32').mean().stop_grad())

        return loss



class BidirectionalNCE1_(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_dtype = jittor.bool
        self.l2_norm = Normalize(2)

    def execute(self, feat_q, feat_k):

        bs, dim, h, w = feat_q.shape
        feat_q = feat_q.view(bs, dim, -1).view(-1, h*w)
        feat_k = feat_k.view(bs, dim, -1).view(-1, h*w)


        feat_q = self.l2_norm(feat_q)  # 3, 410, 64, 64
        feat_k = self.l2_norm(feat_k)
        feat_k = feat_k.detach()

        # print('***', feat_q.shape)

        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]

        l_pos = nn.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # batch_dim_for_bmm = self.args.batch_size
        batch_dim_for_bmm = bs

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = nn.bmm(feat_q, feat_k.transpose(2, 1))
        diagonal = jittor.init.eye(npatches, dtype=self.mask_dtype)[None, :, :]

        #l_neg_curbatch.masked_fill_(diagonal, -10.0)
        
        diagonal = diagonal.repeat([l_neg_curbatch.shape[0],1,1])
        l_neg_curbatch = jittor.masked_fill(l_neg_curbatch, diagonal, -10.0)

        l_neg = l_neg_curbatch.view(-1, npatches)


        out = jittor.concat((l_pos, l_neg), dim=1) / 0.05
        loss = nn.cross_entropy_loss(out, jittor.zeros(out.size(0), dtype='int32').stop_grad(),reduction='none')

        # print('***', l_pos.shape, l_neg.shape, loss)
        # 1/0
        return loss



class BidirectionalNCE2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mask_dtype = 'bool'
        self.l2_norm = Normalize(2)

    def execute(self, feat_q, feat_k):

        feat_q = self.l2_norm(feat_q)
        feat_k = self.l2_norm(feat_k)
        feat_k = feat_k.detach()

        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]

        l_pos = nn.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)


        batch_dim_for_bmm = self.args.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = nn.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = jittor.init.eye(npatches, dtype=self.mask_dtype)[None, :, :]
        diagonal = diagonal.repeat([l_neg_curbatch.shape[0],1,1])
        l_neg_curbatch = jittor.masked_fill(l_neg_curbatch, diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = jittor.contrib.concat((l_pos, l_neg), dim=1) / self.args.nce_T


        loss = self.cross_entropy_loss(out, jittor.zeros(out.size(0), dtype='int32').stop_grad())

        return loss


class SRNCE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mask_dtype = 'bool'
        self.l2_norm = Normalize(2)

    def execute(self, feat_q, feat_k, feat_c):
        feat_q = self.l2_norm(feat_q)
        feat_k = self.l2_norm(feat_k)
        feat_c = self.l2_norm(feat_c)
        feat_k = feat_k.detach()
        feat_c = feat_c.detach()

        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]

        l_pos1 = nn.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos1 = l_pos1.view(batchSize, 1)
        sim_q_c = nn.bmm(feat_c.view(batchSize, 1, -1), feat_q.view(batchSize, -1, 1))
        sim_q_c = sim_q_c.view(batchSize, 1)
        pos_weight = nn.bmm(feat_c.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        pos_weight = pos_weight.view(batchSize, 1)

        l_pos1 = l_pos1 - (0.7 * (1 - pos_weight))
        l_pos2 = (1 - sim_q_c) - (0.7 * (1 - pos_weight))

        batch_dim_for_bmm = self.args.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        trans_feat_q = feat_q.transpose(2, 1)
        trans_feat_k = feat_k.transpose(2, 1)
        feat_c = feat_c.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = nn.bmm(feat_k, trans_feat_q)

        diagonal = jittor.init.eye(npatches, dtype=self.mask_dtype)[None, :, :]
        diagonal = diagonal.repeat([l_neg_curbatch.shape[0],1,1])
        l_neg_curbatch = jittor.masked_fill(l_neg_curbatch, diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out1 = jittor.concat((l_pos1, l_pos2, l_neg), dim=1) / self.args.nce_T

        loss1 = nn.cross_entropy_loss(out1, jittor.zeros(out1.size(0), dtype='int32').stop_grad(),reduction='none')
        out2 = l_pos2 / self.args.nce_T

        loss2 = nn.cross_entropy_loss(out2, jittor.zeros(out2.size(0), dtype='int32').stop_grad(),reduction='none')

        loss = loss1 + loss2
        return loss



def test():
    _Normalize = Normalize()
    print(_Normalize)

    _BidirectionalNCE1 = BidirectionalNCE1()
    print(_BidirectionalNCE1)

    ########### test opt ############
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='# of discrim filters in first conv layer')
    parser.add_argument('--nce_T', type=float, default=10.0, help='weight for feature matching loss')
    opt = parser.parse_args()
    _BidirectionalNCE2 = BidirectionalNCE2(opt)
    print(_BidirectionalNCE2)
    _SRNCE = SRNCE(opt)
    print(_SRNCE)

    inp = jittor.rand((4,16, 64, 64))
    inp2 = jittor.rand((4,16, 64, 64))
    inp3 = jittor.rand((4,16, 64, 64))
    out = _Normalize(inp)
    out = _BidirectionalNCE1(inp, inp2)
    #out = _BidirectionalNCE2(inp, inp2)
    out = _SRNCE(inp, inp2, inp3)

if __name__ == '__main__':
    test()
    pass