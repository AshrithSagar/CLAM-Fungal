import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.utils import initialize_weights
import numpy as np

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), attention_labels_loss_fn=None, subtyping=False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.instance_classifier = nn.Linear(size[1], 2)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.attention_labels_loss_fn = attention_labels_loss_fn
        self.attention_loss_positive = nn.L1Loss(reduction='sum')
        self.attention_loss_negative = nn.L1Loss(reduction='sum')
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifier = self.instance_classifier.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()

    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier, label, bool_annot, patch_annot, semi_supervised, alpha_weight, weight_alpha, training):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        if training:
            bool_annot = bool_annot.item()
        else:
            bool_annot = None

        # Get instance
        if semi_supervised and bool_annot:
            all_instances = h
        else:
            top_p_ids = torch.topk(A.squeeze(), self.k_sample)[1]
            top_n_ids = torch.topk(-A.squeeze(), self.k_sample)[1]
            top_p = torch.index_select(h, dim=0, index=top_p_ids)
            top_n = torch.index_select(h, dim=0, index=top_n_ids)
            all_instances = torch.cat([top_p, top_n], dim=0)

        logits = classifier(all_instances)

        if semi_supervised and bool_annot:
            logits = logits.view(len(h), 2)
        else:
            logits = logits.view(2*self.k_sample, 2)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)

        # Get target labels
        if semi_supervised and bool_annot:
            all_targets = patch_annot[0]

            if label:
                postive_patch_ids = np.where(patch_annot.cpu() == 1)[0]
                negative_patch_ids = np.where(patch_annot.cpu() == 0)[0]

                positive_preds = [A[idx] for idx in postive_patch_ids]
                negative_preds = [A[idx] for idx in negative_patch_ids]
                loss_pos_L1 = attention_loss_positive(positive_preds, np.ones((len(postive_patch_ids),), dtype=int))
                loss_neg_L1 = attention_loss_negative(negative_preds, np.ones((len(negative_patch_ids),), dtype=int))
            else:
                loss_pos_L1 = None
                loss_neg_L1 = None

        else:
            if label:
                p_targets = self.create_positive_targets(self.k_sample, device)  # 1's
            else:
                p_targets = self.create_negative_targets(self.k_sample, device)  # 0's
            n_targets = self.create_negative_targets(self.k_sample, device)  # 0's
            all_targets = torch.cat([p_targets, n_targets], dim=0)

            loss_pos_L1 = None
            loss_neg_L1 = None

        instance_loss = self.instance_loss_fn(logits, all_targets)

        if alpha_weight and semi_supervised:
            if bool_annot:
                instance_loss *= weight_alpha[0]
            else:
                instance_loss *= weight_alpha[1]
        return instance_loss, all_preds, all_targets, loss_pos_L1, loss_neg_L1

    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A.squeeze(), self.k_sample)[1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
#         print("top_p", top_p.shape)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
#         print("logits", logits.shape)
#         print("p_targets", p_targets.shape)
        instance_loss = self.instance_loss_fn(logits.squeeze(), p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, bool_annot=None, patch_annot=None, semi_supervised=False, alpha_weight=False, weight_alpha=None, label=None, instance_eval=False, return_features=False, attention_only=False, training=True):
        device = h.device
#         print("h.shape", h.shape)
        A, h = self.attention_net(h)  # NxK
#         print("A.shape", A.shape)
#         print("h.shape", h.shape)
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            classifier = self.instance_classifier
            instance_loss, preds, targets, loss_pos_L1, loss_neg_L1 = self.inst_eval(A, h, classifier, label, bool_annot, patch_annot, semi_supervised, alpha_weight, weight_alpha, training)

        if semi_supervised and bool_annot:
            A_attention_preds = A.view([77])
            attention_labels_loss = self.attention_labels_loss_fn(A_attention_preds, targets.float())
        else:
            attention_labels_loss = None

        M = torch.mm(A.view(1, 77), h.view(77, 512))
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        if instance_eval:
            results_dict = {'instance_loss': instance_loss,
            'loss_pos_L1': loss_pos_L1, 'loss_neg_L1': loss_neg_L1,
            'inst_labels': np.array(targets.cpu().numpy()),
            'inst_preds': np.array(preds.cpu().numpy())}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
#         print("Y_hat shape", Y_hat.shape)
        return logits, Y_prob, Y_hat, A_raw, results_dict, attention_labels_loss

class CLAM_MB(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A.view(self.n_classes, 24)[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A.view(self.n_classes, 24)[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A.view(self.n_classes, 24), h.view(24, 512))
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
