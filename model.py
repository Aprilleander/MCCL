import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class SharpLogitLoss(nn.Module):
    def __init__(self, T=0.2, alpha_max=1.0, warmup_epochs=4, total_epochs=15,current_epoch =0):
        super().__init__()
        self.T = T
        self.alpha_max = alpha_max
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = current_epoch

    def update(self, epoch):
        self.current_epoch = epoch

    def forward(self,logits,targets,prototypes):
        """
        logits: [B, C], raw logits from the classifier
        """
        if self.alpha_max == 0:
            return 0.0
        # soft targets from current logits (detached)
        with torch.no_grad():
            teacher = prototypes[targets]
            soft_targets = F.softmax(teacher.detach() / self.T, dim=1)
        # student log_probs
        student_log_probs = F.log_softmax(logits / self.T, dim=1)
        # KL divergence loss
        kl_loss = F.kl_div(student_log_probs, soft_targets, reduction='batchmean')*self.T
        # Warmup alpha
        if self.current_epoch < self.warmup_epochs:
            alpha = self.alpha_max * self.current_epoch / max(1, self.warmup_epochs)
        else:
            alpha = self.alpha_max
        return alpha * kl_loss
    
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1, num_classes=2):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, input, target, similarity,smoothing = 0.5):
        target_smooth = F.one_hot(target,input.size(1)).float()*(1-smoothing) +smoothing*similarity#F.one_hot(similarity,input.size(1)).float()#s1/input.size(0)#coef# / self.num_classes
        return torch.nn.CrossEntropyLoss()(input, target_smooth)

def KL_loss(cls_x,memory_x):

    kl_loss = (F.kl_div(F.log_softmax(cls_x, dim=1), memory_x, reduction='batchmean'))
    return kl_loss

def top1_sharpness_loss(logits):
    top1, _ = logits.max(dim=1, keepdim=True)    # [B, 1]
    mean_others = (logits.sum(dim=1, keepdim=True) - top1) / (logits.size(1) - 1)
    loss = F.relu(mean_others - top1).mean()     # 目标是 top1 >> 其他平均
    return loss

"""
def prototype_separation_loss(prototypes, temperature=0.1, base_temperature=0.1,local_rank =0):
    num_classes = prototypes.size(0)
    labels = torch.arange(0, num_classes).cuda(local_rank,non_blocking=True)
    labels = labels.contiguous().view(-1, 1)

    mask = (1- torch.eq(labels, labels.T).float()).cuda()

    logits = torch.div(torch.matmul(prototypes, prototypes.T), temperature)

    mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
    mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]

    # loss
    loss = temperature / base_temperature * mean_prob_neg.mean()
    return loss
"""


class SimpleRelationConsistencyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha  # 关系一致性权重
        self.beta = beta    # 原型一致性权重
        
    def forward(self, soft_labels, relation_matrix, prototype_similarity, known_class_mask):
        """
        Args:
            soft_labels: [8, 101] 软标签分布
            relation_matrix: [8, 8] 连续值关系矩阵
            prototype_similarity: [8, 50] 与已知prototype的相似度
            known_class_mask: [101] 已知类别的掩码
        """
        # 1. 从软标签计算预测的关系矩阵 (使用余弦相似度)
        normalized_labels = F.normalize(soft_labels, p=2, dim=1)
        pred_relation = torch.mm(normalized_labels, normalized_labels.t())
        
        # 2. 关系一致性损失 (MSE)
        relation_loss = F.mse_loss(pred_relation, relation_matrix)
        
        # 3. 原型一致性损失
        known_indices = torch.where(known_class_mask)[0]
        if len(known_indices) > 0:
            known_probs = soft_labels[:, known_indices]
            prototype_loss = F.mse_loss(known_probs, prototype_similarity)
        else:
            prototype_loss = torch.tensor(0.0)
        
        # 总损失
        total_loss = self.alpha * relation_loss + self.beta * prototype_loss
        
        return total_loss, {
            "relation_loss": relation_loss,
            "prototype_loss": prototype_loss
        }

        

def prototype_separation_loss(prototypes, temperature=0.1,local_rank=0):
    # 确保L2归一化
    prototypes = F.normalize(prototypes, p=2, dim=-1)
    
    # 计算相似度矩阵
    sim_matrix = torch.mm(prototypes, prototypes.t()) / temperature
    
    # 创建掩码（排除对角线）
    mask = ~torch.eye(len(prototypes), dtype=torch.bool, device=prototypes.device)
    
    # 计算负对相似度
    neg_sim = sim_matrix.masked_select(mask).view(len(prototypes), -1)
    
    # 稳定logsumexp计算
    max_vals = neg_sim.max(dim=1, keepdim=True).values
    log_neg_sum = torch.log(torch.exp(neg_sim - max_vals).sum(dim=1)) + max_vals.squeeze()
    
    # 计算损失（最大化分离 → 最小化负对相似度）
    loss = log_neg_sum.mean()  # 等价于推远所有负对
    
    return loss


def max_margin_loss(logits, margin=5.0):
    # logits: [B, 174]
    max_logit, _ = logits.max(dim=1, keepdim=True)                # [B, 1]
    second_logit = logits.clone()
    second_logit[torch.arange(logits.size(0)), logits.argmax(1)] = -1e9
    second_max, _ = second_logit.max(dim=1, keepdim=True)         # [B, 1]
    # 目标是 max_logit - second_max >= margin
    loss = F.relu(margin - (max_logit - second_max)).mean()
    return loss

def info_nce_logits_for(features, confusion_factor, args,local_rank,is_code=False):

    b_ = 0.5 * int(features.size(0))
    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda(local_rank,non_blocking=True)

    features = F.normalize(features, dim=1)
    if is_code:
        dist=torch.cdist(features, features,p=2)
        similarity_matrix =-dist/(dist.sum(dim=1)+1e-10)
    else:
        similarity_matrix=-torch.cdist(features, features)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(local_rank,non_blocking=True)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    confusion_factor = confusion_factor[~mask].view(confusion_factor.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    pos_confs= confusion_factor[labels.bool()].view(confusion_factor.shape[0], -1)


    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    neg_confs= confusion_factor[~labels.bool()].view(confusion_factor.shape[0], -1)


    logits = torch.cat([positives, negatives], dim=1)
    log_confs = torch.cat([pos_confs, neg_confs], dim=1)

    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(local_rank,non_blocking=True)

    logits = logits / args.temperature
    return logits, labels, log_confs

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
            
        # prototypes
        self.prototype_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.prototype_layer.weight_g.data.fill_(1)
        self.prototype_layer.weight_g.requires_grad = False
        print('prototype size: ', self.prototype_layer.weight_v.size())           

        print('randomly initialize prototypes...')
        print(self.prototype_layer.weight_v)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        
        prototypes = self.prototype_layer.weight_v.clone()
        normed_prototypes = F.normalize(prototypes, dim=-1, p=2)
        
        return x_proj, logits


def temporal_contrast(strong,weak,backbone,cfg):
    loss_unlabel = torch.nn.CrossEntropyLoss(reduction="none")
    loss_l2 = torch.nn.MSELoss(reduction="none")
    
    strong_temporal = temporal_aug(strong,cfg)
    strong = F.dropout2d(strong, 0.2)

    student,s_cls_x,s_t_feat,s_s_feat= backbone(strong)
    idx = torch.randperm(strong.size(0))
    strong_resort = strong_temporal[idx,:]
    strong = torch.cat([strong, strong_resort], dim=0)
    with torch.no_grad():
        teacher_output = backbone(weak)
        teacher, t_cls_x, t_t_feat, t_s_feat = (x.detach() for x in teacher_output)
    backbone.train()
    mask_ratio = np.random.beta(cfg.TRAIN.BETA,cfg.TRAIN.BETA)

    tube = TubeMasking(input_size=cfg.DATA.WINDOWS_SIZE, mask_ratio=mask_ratio)
    tubemask = tube.tubemask()
    tubemask = tubemask[None].repeat(teacher.size(0), axis=0)
    tubemask = torch.Tensor(tubemask).cuda(non_blocking=True)
    student_mix,mix_cls_x, mix_t_feat, mix_s_feat = backbone(strong,tubemask)
    
    pseudo = torch.zeros(teacher.size(0),teacher.size(-1)).cuda()
    pseudo_label = torch.softmax(teacher / 1, dim=-1)
    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(cfg.TRAIN.DELTA).float()
    pseudo_label2 = pseudo_label[idx,:]
    max_probs2, _ = torch.max(pseudo_label2, dim=-1)
    mask2 = max_probs2.ge(cfg.TRAIN.DELTA).float()
    mask_mix = mask_ratio *mask + (1-mask_ratio)*mask2
    for i in range(teacher.size(0)):
        if mask[i] > 0:
            pseudo[i][targets_u[i]] = 1
        else:
            pseudo[i] = pseudo_label[i]
    y_1 = pseudo
    y_2 = pseudo[idx,:]
    y_label = mask_ratio * y_1 + (1-mask_ratio) * y_2
    loss_mix = (torch.mean(loss_l2(student_mix, y_label), dim=1)*mask_mix).mean()
    loss_un = (loss_unlabel(student, targets_u)*mask).mean()
    return loss_mix, loss_un

    
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1,wait_ratio_epochs=0, ramp_ratio_teacher_epochs=10,init_ratio=0.0, final_ratio=1.0):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.ratio_schedule = np.concatenate((
            np.zeros(wait_ratio_epochs),
            np.linspace(init_ratio,
                        final_ratio, ramp_ratio_teacher_epochs),
            np.ones(nepochs - wait_ratio_epochs - ramp_ratio_teacher_epochs) * final_ratio
        ))

    def forward(self, student_output, teacher_output, epoch,cfg):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        ratio_epoch = self.ratio_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        
        teacher_label = []
        for i in range(self.ncrops):
            top2 = torch.topk(teacher_out[i], k=2, dim=-1, largest=True)[0]
            top2_div = top2[:, 0] / (top2[:, 1] + 1e-6)
            filter_number = int(len(teacher_out[i])*ratio_epoch)
            topk_filter = torch.topk(top2_div, k=filter_number, largest=True)[1]
            pseudo_label = F.one_hot(teacher_out[i].argmax(dim=-1), num_classes=cfg.MODEL.NUM_CLASSES)
            pseudo_label = pseudo_label.float()
            teacher_out[i][topk_filter] = pseudo_label[topk_filter]
            teacher_label.append(teacher_out[i])

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_label):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss
