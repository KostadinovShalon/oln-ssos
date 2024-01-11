import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from mmdet.core import bbox2roi, bbox2result
from mmdet.models import HEADS

from FrEIA.framework import InputNode, OutputNode, Node, GraphINN
from FrEIA.modules import PermuteRandom, GLOWCouplingBlock

from vos.models.roi_heads.oln_mask_vos_roi_head import OLNMaskKMeansVOSRoIHead
from vos.models.roi_heads.oln_vos_roi_head import OLNKMeansVOSRoIHead

warnings.simplefilter(action='ignore', category=FutureWarning)


@HEADS.register_module()
class OLNKMeansFFSRoIHead(OLNKMeansVOSRoIHead):

    def __init__(self,
                 nll_loss_weight=0.1,
                 *args,
                 **kwargs):
        """
        VOS BBox Head

        Args:
            vos_sample_per_class: queue size for each class to form the Gaussians
            start_epoch: starting epoch where VOS is going to be applied
            logistic_regression_hidden_dim: hidden dimension for the logistic regression layer (phi in Eq. 5)
            negative_sampling_size: number of samples from the multivariate Gaussian where the lowest k samples are
                considered negative (ood)
            bottomk_epsilon_dist: lowest k elements to use from `negative_sampling_size` samples form
                the multivariate Gaussian to be considered as negative
            ood_loss_weight: uncertainty loss weight
        """
        super(OLNKMeansFFSRoIHead, self).__init__(*args, **kwargs)

        self.weight_energy = torch.nn.Linear(self.k, 1).cuda()
        torch.nn.init.uniform_(self.weight_energy.weight)
        self.nll_loss_weight = nll_loss_weight

        self.in1 = InputNode(1024, name='input1')
        self.layer1 = Node(self.in1, GLOWCouplingBlock, {'subnet_constructor': self.subnet_fc, 'clamp': 2.0},
                           name=F'coupling_{0}')
        self.layer2 = Node(self.layer1, PermuteRandom, {'seed': 0}, name=F'permute_{0}')
        self.layer3 = Node(self.layer2, GLOWCouplingBlock, {'subnet_constructor': self.subnet_fc, 'clamp': 2.0},
                           name=F'coupling_{1}')
        self.layer4 = Node(self.layer3, PermuteRandom, {'seed': 1}, name=F'permute_{1}')
        self.out1 = OutputNode(self.layer4, name='output1')
        self.flow_model = GraphINN([self.in1, self.layer1, self.layer2, self.layer3, self.layer4, self.out1])

    def subnet_fc(self, c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, 2048), nn.ReLU(), nn.Linear(2048, c_out))

    def NLLLoss(self, z, sldj):
        """Negative log-likelihood loss assuming isotropic gaussian with unit norm.
          Args:
             k (int or float): Number of discrete values in each input dimension.
                E.g., `k` is 256 for natural images.
          See Also:
              Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
        """
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) - np.log(256) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()
        return nll

    def NLL(self, z, sldj):
        """Negative log-likelihood loss assuming isotropic gaussian with unit norm.
          Args:
             k (int or float): Number of discrete values in each input dimension.
                E.g., `k` is 256 for natural images.
          See Also:
              Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
        """
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) - np.log(256) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll
        return nll

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, sampling_weak_results=None, gt_weak_bboxes=None, gt_weak_pseudo_labels=None):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(None,
                                        bbox_results['bbox_pred'],
                                        bbox_results['bbox_score'],
                                        rois,
                                        *bbox_targets)
        if sampling_weak_results is not None and len(sampling_weak_results) > 0:
            weak_rois = bbox2roi([res.bboxes for res in sampling_weak_results])
            bbox_weak_results = self._bbox_forward(x, weak_rois)
            bbox_weak_targets = self.bbox_head.get_targets(sampling_weak_results,
                                                           gt_weak_bboxes, gt_weak_pseudo_labels, self.train_cfg)
        else:
            bbox_weak_results = None
            bbox_weak_targets = None
        # VOS STARTS HERE
        ood_loss, pseudo_loss, nll_loss = self._ood_forward_train(bbox_results, bbox_targets, bbox_weak_results,
                                                                  bbox_weak_targets,
                                                                  device=x[0].device)
        loss_bbox["loss_ood"] = self.ood_loss_weight * ood_loss
        loss_bbox["loss_pseudo_class"] = self.pseudo_label_loss_weight * pseudo_loss
        loss_bbox["loss_nll"] = self.nll_loss_weight * nll_loss
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def log_sum_exp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        import math
        # TODO: torch.max(value, dim=None) threw an error at time of writing
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(
                F.relu(self.weight_energy.weight) * torch.exp(value0),dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            # if isinstance(sum_exp, Number):
            #     return m + math.log(sum_exp)
            # else:
            return m + torch.log(sum_exp)

    def _ood_forward_train(self, bbox_results, bbox_targets, bbox_weak_results, bbox_weak_targets, device):
        selected_fg_samples = (bbox_targets[0] != self.k).nonzero().view(-1)
        indices_numpy = selected_fg_samples.cpu().numpy().astype(int)

        z, sldj = self.flow_model(bbox_results['shared_bbox_feats'][indices_numpy,].detach().cuda())
        nll_loss = self.NLLLoss(z, sldj)

        # if bbox_weak_results is not None:
        #     weak_selected_fg_samples = (bbox_weak_targets[0] != self.k).nonzero().view(-1)
        #     weak_indices_numpy = weak_selected_fg_samples.cpu().numpy().astype(int)
        #     weak_gt_classes_numpy = bbox_weak_targets[0].cpu().numpy().astype(int)
        #
        #     weak_box_features = []
        #     for index in weak_indices_numpy:
        #         weak_box_features.append(bbox_weak_results['shared_bbox_feats'][index].view(1, -1))
        #     if len(weak_box_features) == 0:
        #         weak_box_features = None
        #     else:
        #         weak_box_features = torch.cat(weak_box_features, dim=0)
        # else:
        #     weak_box_features = None
        #     weak_indices_numpy = None
        #     weak_gt_classes_numpy = None

        gt_box_features = []
        for index in indices_numpy:
            gt_box_features.append(bbox_results['shared_bbox_feats'][index].view(1, -1))
        gt_box_features = torch.cat(gt_box_features, dim=0)

        ood_reg_loss = torch.zeros(1).to(device)
        loss_pseudo_score = torch.zeros(1).cuda()
        if self.kmeans is not None:
            if not self.use_all_proposals_ood:
                # gt_pseudo_labels = self.kmeans.predict(gt_box_features.detach().cpu())
                gt_pseudo_logits = self.pseudo_score(gt_box_features)
            else:
                # gt_pseudo_labels = self.kmeans.predict(bbox_results['shared_bbox_feats'].detach().cpu())
                gt_pseudo_logits = self.pseudo_score(bbox_results['shared_bbox_feats'])
            gt_pseudo_labels = bbox_targets[0][selected_fg_samples]

            # if weak_box_features is not None:
            #     weak_pseudo_logits = self.pseudo_score(weak_box_features)
            #     weak_pseudo_labels = bbox_weak_targets[0][weak_selected_fg_samples]
            #     gt_pseudo_logits = torch.cat([gt_pseudo_logits, weak_pseudo_logits], dim=0)
            #     gt_pseudo_labels = torch.cat([gt_pseudo_labels, weak_pseudo_labels], dim=0)
            loss_pseudo_score = self.loss_pseudo_cls(gt_pseudo_logits, gt_pseudo_labels.long())
            if self.epoch >= self.start_epoch:
                with torch.no_grad():
                    z_randn = torch.randn((self.negative_sampling_size, 1024), dtype=torch.float32).cuda()
                    negative_samples, _ = self.flow_model(z_randn, rev=True)
                    # negative_samples = torch.sigmoid(negative_samples)
                    _, sldj_neg = self.flow_model(negative_samples)
                    nll_neg = self.NLL(z_randn, sldj_neg)
                    cur_samples, index_prob = torch.topk(nll_neg, self.bottomk_epsilon_dist)
                    ood_samples = negative_samples[index_prob].view(1, -1)
                    # ood_samples = torch.squeeze(ood_samples)
                    del negative_samples
                    del z_randn
                energy_score_for_fg = self.log_sum_exp(gt_pseudo_logits, 1)

                # Now we need to get the class logits for the negative samples.
                predictions_ood = self.pseudo_score(ood_samples)
                energy_score_for_bg = self.log_sum_exp(predictions_ood, 1)

                input_for_loss = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                id_labels_size = energy_score_for_fg.shape[0]
                labels_for_loss = torch.cat((torch.ones(id_labels_size).to(device),
                                             torch.zeros(len(ood_samples)).to(device)), -1)

                output = self.logistic_regression_layer(input_for_loss.view(-1, 1))
                ood_reg_loss = F.binary_cross_entropy_with_logits(
                    output.view(-1), labels_for_loss)

        return ood_reg_loss, loss_pseudo_score, nll_loss


@HEADS.register_module()
class OLNMaskKMeansFFSRoIHead(OLNKMeansFFSRoIHead, OLNMaskKMeansVOSRoIHead):
    pass
