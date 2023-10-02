import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmdet.core import bbox2roi, bbox2result
from mmdet.models import StandardRoIHead, HEADS


def bbox2result_ood(bboxes, labels, ood_scores, inter_feats, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        ood_scores (torch.Tensor | np.ndarray): shape (n, )  [1 -> in distribution]
        inter_feats (torch.Tensor | np.ndarray): shape (n, k + 1)
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            ood_scores = ood_scores.detach().cpu().numpy()
            inter_feats = inter_feats.detach().cpu().numpy()
        bboxes_ood = np.hstack((bboxes, ood_scores[:, None], inter_feats))
        # bboxes_ood is now an N x (4 + 1 + 1 + (K + 1)) tensor
        return [bboxes_ood[labels == i, :] for i in range(num_classes)]


class FeaturesQueue:

    def __init__(self, num_classes=20, per_class_size=1000):
        self.num_classes = num_classes
        self.per_class_size = per_class_size
        self.data = [list() for _ in range(num_classes)]

    def push_features(self, x, class_indices):
        """

        Args:
            x: (N, *) feature tensor
            class_indices: N-list of class indices

        """

        assert x.shape[0] == class_indices.shape[0]
        for _x, class_idx in zip(x, class_indices):
            self.push_feature(_x.view(-1), class_idx)

    def push_feature(self, x, class_idx):
        if len(self.data[class_idx]) < self.per_class_size:
            self.data[class_idx].append(x)
        else:
            self.data[class_idx][:-1] = self.data[class_idx][1:]
            self.data[class_idx][-1] = x

    @property
    def is_ready(self):
        return sum(len(fts) for fts in self.data) == self.num_classes * self.per_class_size

    def get_data_tensor(self):
        if not self.is_ready:
            return
        return torch.stack([torch.stack(d, 0) for d in self.data], 0)  # K x Q x C


@HEADS.register_module()
class VOSRoIHead(StandardRoIHead):

    def __init__(self,
                 vos_samples_per_class=1000,
                 start_epoch=0,
                 logistic_regression_hidden_dim=512,
                 negative_sampling_size=10000,
                 bottomk_epsilon_dist=1,
                 ood_loss_weight=0.1,
                 linear_logistic=False,
                 use_queue=True,
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
        super(VOSRoIHead, self).__init__(*args, **kwargs)
        self.vos_samples_per_class = vos_samples_per_class
        self.start_epoch = start_epoch
        self.bottomk_epsilon_dist = bottomk_epsilon_dist
        self.negative_sampling_size = negative_sampling_size
        self.ood_loss_weight = ood_loss_weight

        self.linear_logistic = linear_logistic
        if linear_logistic:
            self.logistic_regression_layer = nn.Linear(1, 2)
        else:
            self.logistic_regression_layer = nn.Sequential(
                nn.Linear(1, logistic_regression_hidden_dim),
                nn.ReLU(),
                nn.Linear(logistic_regression_hidden_dim, 1)
            )

        self.weight_energy = nn.Linear(self.bbox_head.num_classes, 1)
        nn.init.uniform_(self.weight_energy.weight)
        self.epoch = 0

        self.use_queue = use_queue
        if use_queue:
            self.queue = FeaturesQueue(self.bbox_head.num_classes, self.vos_samples_per_class)
        else:
            self.data_dict = torch.zeros(self.bbox_head.num_classes, self.vos_samples_per_class, 1024).cuda()
            self.number_dict = {}
            for i in range(self.bbox_head.num_classes):
                self.number_dict[i] = 0

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, fts = self.bbox_head(bbox_feats)

        # FEATS ARE CHANGED HERE
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, shared_bbox_feats=fts)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        # VOS STARTS HERE
        ood_loss = self._ood_forward_train(bbox_results, bbox_targets, device=x[0].device)
        loss_bbox["loss_ood"] = self.ood_loss_weight * ood_loss
        f = torch.nn.MSELoss()
        if self.linear_logistic:
            loss_dummy = f(self.logistic_regression_layer(torch.zeros(1).cuda()), self.logistic_regression_layer.bias)
            loss_dummy1 = f(self.weight_energy(torch.zeros(self.bbox_head.num_classes).cuda()), self.weight_energy.bias)
        else:
            loss_dummy = f(self.logistic_regression_layer(torch.zeros(1).cuda()),
                           self.logistic_regression_layer(torch.zeros(1).cuda()))
            loss_dummy1 = f(self.weight_energy(torch.zeros(self.bbox_head.num_classes).cuda()),
                            self.weight_energy(torch.zeros(self.bbox_head.num_classes).cuda()))
        loss_bbox["loss_dummy"] = loss_dummy
        loss_bbox["loss_dummy1"] = loss_dummy1
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _ood_forward_train(self, bbox_results, bbox_targets, device):
        n_classes = bbox_results['cls_score'].shape[1]
        selected_fg_samples = (bbox_targets[0] != n_classes - 1).nonzero().view(-1)
        indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
        gt_classes_numpy = bbox_targets[0].cpu().numpy().astype(int)

        if self.use_queue:
            self.queue.push_features(bbox_results['shared_bbox_feats'][indices_numpy].detach(),
                                     gt_classes_numpy[indices_numpy])
            queue_ready = self.queue.is_ready
        else:
            sum_temp = 0
            for index in range(self.bbox_head.num_classes):
                sum_temp += self.number_dict[index]
            queue_ready = sum_temp >= (n_classes - 1) * self.vos_samples_per_class
            if not queue_ready:
                for index in indices_numpy:
                    dict_key = gt_classes_numpy[index]
                    if self.number_dict[dict_key] < self.vos_samples_per_class:
                        self.data_dict[dict_key][self.number_dict[dict_key]] = bbox_results['shared_bbox_feats'][index].detach()
                        self.number_dict[dict_key] += 1
            else:
                for index in indices_numpy:
                    dict_key = gt_classes_numpy[index]
                    self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                          bbox_results['shared_bbox_feats'][index].detach().view(1, -1)), 0)
        ood_reg_loss = torch.zeros(1).to(device)
        if queue_ready and self.epoch >= self.start_epoch:
            # Getting means
            if self.use_queue:
                q = self.queue.get_data_tensor()  # K x N x C
                means = torch.mean(q, dim=1)  # K x C
                X = q - means[:, None]  # K x N x C
                X = X.view(-1, X.size(-1))  # KN x C, there could be an easier way to implement this

                cov_mat = torch.mm(X.t(), X) / len(X)  # C x C
                # For stability
                cov_mat += 1e-4 * torch.eye(self.bbox_head.fc_out_channels, device=device)

                dists = [torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov_mat)
                         for mean in means]

                negative_samples = [dist.rsample((self.negative_sampling_size,)) for dist in dists]
                prob_density = [dist.log_prob(neg_sample) for dist, neg_sample in zip(dists, negative_samples)]

                indices_prob = [torch.topk(- p, self.bottomk_epsilon_dist)[1] for p in prob_density]
                ood_samples = [neg_sample[index] for neg_sample, index in zip(negative_samples, indices_prob)]
                ood_samples = torch.cat(ood_samples, dim=0)  # N x C
            else:
                # the covariance finder needs the data to be centered.
                for index in range(n_classes - 1):
                    if index == 0:
                        X = self.data_dict[index] - self.data_dict[index].mean(0)
                        mean_embed_id = self.data_dict[index].mean(0).view(1, -1)
                    else:
                        X = torch.cat((X, self.data_dict[index] - self.data_dict[index].mean(0)), 0)
                        mean_embed_id = torch.cat((mean_embed_id,
                                                   self.data_dict[index].mean(0).view(1, -1)), 0)

                # add the variance.
                temp_precision = torch.mm(X.t(), X) / len(X)
                # for stable training.
                temp_precision += 0.0001 * torch.eye(self.bbox_head.fc_out_channels, device=device)

                for index in range(n_classes - 1):
                    new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                        mean_embed_id[index], covariance_matrix=temp_precision)
                    negative_samples = new_dis.rsample((self.negative_sampling_size,))
                    prob_density = new_dis.log_prob(negative_samples)

                    # keep the data in the low density area.
                    cur_samples, index_prob = torch.topk(- prob_density, self.bottomk_epsilon_dist)
                    if index == 0:
                        ood_samples = negative_samples[index_prob]
                    else:
                        ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
                    del new_dis
                    del negative_samples

            if len(ood_samples) > 0:
                energy_score_for_fg = self.log_sum_exp(bbox_results['cls_score'][selected_fg_samples][:, :-1], 1)

                # Now we need to get the class logits for the negative samples.
                predictions_ood = self.bbox_head.forward_cls_branch(ood_samples)
                energy_score_for_bg = self.log_sum_exp(predictions_ood[:, :-1], 1)

                input_for_loss = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                labels_for_loss = torch.cat((torch.ones(len(selected_fg_samples)).cuda(),
                                             torch.zeros(len(ood_samples)).cuda()), -1)

                output = self.logistic_regression_layer(input_for_loss.view(-1, 1))
                if self.linear_logistic:
                    criterion = torch.nn.CrossEntropyLoss()  # weight=weights_fg_bg)
                    ood_reg_loss = criterion(output, labels_for_loss.long())
                else:
                    ood_reg_loss = F.binary_cross_entropy_with_logits(
                        output.view(-1), labels_for_loss,
                        pos_weight=torch.tensor(len(ood_samples)/len(selected_fg_samples)).cuda())
        return ood_reg_loss

    def log_sum_exp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation

        value.exp().sum(dim, keepdim).log()
        """
        # TODO: torch.max(value, dim=None) threw an error at time of writing
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(
                F.relu(self.weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            # if isinstance(sum_exp, Number):
            #     return m + math.log(sum_exp)
            # else:
            return m + torch.log(sum_exp)

    # def simple_test_ood(self,
    #                     x,
    #                     proposals):
    #     """Test only det bboxes without augmentation."""
    #     rois = bbox2roi(proposals)
    #     bbox_results = self._bbox_forward(x, rois)
    #
    #     # split batch bbox prediction back to each image
    #     cls_score = bbox_results['cls_score']
    #     energy = self.log_sum_exp(cls_score[:, :-1], 1)
    #     ood_scores = self.logistic_regression_layer(energy.view(-1, 1))
    #     num_proposals_per_img = tuple(len(p) for p in proposals)
    #     cls_score = cls_score.split(num_proposals_per_img, 0)
    #     ood_scores = ood_scores.split(num_proposals_per_img, 0)
    #
    #     det_ood_scores = []
    #     for i in range(len(proposals)):
    #         if isinstance(ood_scores[i], list):
    #             ood_scores[i] = sum(ood_scores[i]) / float(len(ood_scores[i]))
    #         det_ood_scores.append(F.sigmoid(ood_scores[i]) if cls_score is not None else None)
    #
    #     # apply bbox post-processing to each image individually
    #     return det_ood_scores

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        # OOD
        inter_feats = cls_score  # N x (K + 1)
        energy = torch.logsumexp(cls_score[:, :-1], 1)
        ood_scores = self.logistic_regression_layer(energy.view(-1, 1))

        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        ood_scores = ood_scores.split(num_proposals_per_img, 0)
        inter_feats = inter_feats.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_ood_scores = []
        det_inter_feats = []
        for i in range(len(proposals)):
            oods = F.sigmoid(ood_scores[i][:, 0]) if cls_score is not None else None
            det_bbox, det_label, _ood_scores, _inter_feats = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                oods,
                inter_feats[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            det_ood_scores.append(_ood_scores)
            det_inter_feats.append(_inter_feats)
        return det_bboxes, det_labels, det_ood_scores, det_inter_feats

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels, det_ood_scores, det_inter_feats = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        # det_ood_scores = self.simple_test_ood(x, proposal_list)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, det_ood_scores, segm_results,
            else:
                return det_bboxes, det_labels, det_ood_scores

        bbox_results = [
            bbox2result_ood(det_bboxes[i], det_labels[i], det_ood_scores[i], det_inter_feats[i],
                            self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))
