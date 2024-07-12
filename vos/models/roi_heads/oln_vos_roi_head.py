import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from mmdet.core import bbox2roi, bbox2result
from mmdet.models import HEADS, OlnRoIHead, build_roi_extractor

warnings.simplefilter(action='ignore', category=FutureWarning)


def bbox2result_ood(bboxes, labels, ood_scores, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        ood_scores (torch.Tensor | np.ndarray): shape (n, )  [1 -> in distribution]
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
        bboxes_ood = np.hstack((bboxes, ood_scores[:, None]))
        # bboxes_ood is now an N x (4 + 1 + 1 + (K + 1)) tensor
        return [bboxes_ood]


@HEADS.register_module()
class OLNKMeansVOSRoIHead(OlnRoIHead):

    def __init__(self,
                 start_epoch=0,
                 logistic_regression_hidden_dim=512,
                 vos_samples_per_class=1000,
                 negative_sampling_size=10000,
                 bottomk_epsilon_dist=1,
                 ood_loss_weight=0.1,
                 pseudo_label_loss_weight=1.0,
                 k=5,
                 recalculate_pseudolabels_every_epoch=1,
                 k_means_minibatch=True,
                 repeat_ood_sampling=4,
                 use_all_proposals_ood=False,
                 weak_bbox_test_confidence=0.5,
                 pseudo_bbox_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(type='RoIAlign', output_size=3, sampling_ratio=0),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
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
        super(OLNKMeansVOSRoIHead, self).__init__(*args, **kwargs)
        self.vos_samples_per_class = vos_samples_per_class
        self.start_epoch = start_epoch
        self.bottomk_epsilon_dist = bottomk_epsilon_dist
        self.negative_sampling_size = negative_sampling_size
        self.ood_loss_weight = ood_loss_weight
        self.weak_bbox_test_confidence = weak_bbox_test_confidence

        self.k = k
        self.recalculate_pseudolabels_every_epoch = recalculate_pseudolabels_every_epoch
        self.k_means_minibatch = k_means_minibatch
        self.repeat_ood_sampling = repeat_ood_sampling

        self.logistic_regression_layer = nn.Sequential(
            nn.Linear(1, logistic_regression_hidden_dim),
            nn.ReLU(),
            nn.Linear(logistic_regression_hidden_dim, 1)
        )

        self.epoch = 0

        self.data_dict = torch.zeros(self.k, self.vos_samples_per_class, 1024).cuda()
        self.number_dict = {}
        for i in range(self.k):
            self.number_dict[i] = 0

        # self.samples_for_covariance = 20 * 1024
        self.ft_minibatches = []

        self.pseudo_score = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.k)
        )

        for m in self.pseudo_score.modules():
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
        ft_size = pseudo_bbox_roi_extractor['roi_layer']['output_size'] ** 2 * pseudo_bbox_roi_extractor['out_channels']
        self.means = nn.Parameter((torch.zeros(k, ft_size)), requires_grad=False)
        self.cov = None
        self.kmeans = None

        self.loss_pseudo_cls = torch.nn.CrossEntropyLoss()
        self.use_all_proposals_ood = use_all_proposals_ood

        self.post_epoch_features = []
        self.post_epoch_weak_features = []
        self.pseudo_label_loss_weight = pseudo_label_loss_weight
        self.bbox_head.num_classes = self.k

        self.pseudo_bbox_roi_extractor = build_roi_extractor(pseudo_bbox_roi_extractor)

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_ann_ids=None,
                      gt_pseudo_labels=None,
                      gt_weak_bboxes=None,
                      gt_weak_bboxes_labels=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposal_list (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            gt_ann_ids (None | Tensor) : .
            gt_pseudo_labels (None | Tensor) : .
            gt_weak_bboxes (None | Tensor) : .
            gt_weak_bboxes_labels (None | Tensor) : .

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        sampling_results = []
        sampling_weak_results = []
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_pseudo_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_pseudo_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
                if gt_weak_bboxes:
                    weak_assign_results = self.bbox_assigner.assign(
                        proposal_list[i], gt_weak_bboxes[i], None,
                        gt_weak_bboxes_labels[i])
                    weak_sampling_result = self.bbox_sampler.sample(
                        weak_assign_results,
                        proposal_list[i],
                        gt_weak_bboxes[i],
                        gt_weak_bboxes_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_weak_results.append(weak_sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_pseudo_labels,
                                                    img_metas,
                                                    sampling_weak_results,
                                                    gt_weak_bboxes,
                                                    gt_weak_bboxes_labels)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(x, sampling_results,
                                                        bbox_results['bbox_feats'],
                                                        gt_masks, img_metas)
                losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, bbox_score, fts = self.bbox_head(bbox_feats)

        # FEATS ARE CHANGED HERE
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_score=bbox_score, shared_bbox_feats=fts,
            bbox_feats=bbox_feats)
        return bbox_results

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
        ood_loss, pseudo_loss = self._ood_forward_train(bbox_results, bbox_targets, bbox_weak_results,
                                                        bbox_weak_targets,
                                                        device=x[0].device)
        loss_bbox["loss_ood"] = self.ood_loss_weight * ood_loss
        loss_bbox["loss_pseudo_class"] = self.pseudo_label_loss_weight * pseudo_loss
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def accumulate_pseudo_labels(self, fts, rois):
        bbox_feats = self.pseudo_bbox_roi_extractor(
            fts[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_feats = bbox_feats.flatten(1)
        self.post_epoch_features.append(bbox_feats)

    def accumulate_weak_pseudo_labels(self, fts, rois):
        bbox_feats = self.pseudo_bbox_roi_extractor(
            fts[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_feats = bbox_feats.flatten(1)
        self.post_epoch_weak_features.append(bbox_feats)

    def calculate_pseudo_labels(self):
        # if len(self.post_epoch_weak_features) > 0:
        #     weak_fts = torch.cat(self.post_epoch_weak_features, dim=0)
        #     fts = torch.cat([fts, weak_fts], dim=0)
        if self.means.sum().cpu().item() == 0:
            self.kmeans = MiniBatchKMeans(n_clusters=self.k, n_init=1, batch_size=1024)
        else:
            self.kmeans = MiniBatchKMeans(n_clusters=self.k, n_init=1, batch_size=1024,
                                          init=self.means.data.cpu())

        current_iter = 0
        dev = self.post_epoch_features[0].device
        data_to_fit = torch.zeros((1024, self.post_epoch_features[0].shape[1])).to(dev)
        last_index = 0
        while current_iter < len(self.post_epoch_features):
            iter_fts = self.post_epoch_features[current_iter]
            if iter_fts is None:
                current_iter += 1
                continue
            n_fts = iter_fts.shape[0]
            if last_index + n_fts < 1024:
                data_to_fit[last_index:(last_index + n_fts)] = iter_fts
                last_index += n_fts
                current_iter += 1
            else:
                fts_to_use = 1024 - last_index
                if fts_to_use > 0:
                    data_to_fit[last_index:] = iter_fts[:fts_to_use]
                self.kmeans.partial_fit(data_to_fit.cpu())
                last_index = n_fts - fts_to_use
                data_to_fit = torch.zeros((1024, self.post_epoch_features[0].shape[1])).to(dev)
                data_to_fit[:last_index] = iter_fts[fts_to_use:]
                current_iter += 1
        labels = []
        for iter_fts in self.post_epoch_features:
            if iter_fts is None:
                continue
            _labels = self.kmeans.predict(iter_fts.cpu())
            labels.append(torch.tensor(_labels).to(dev))
        labels = torch.cat(labels).cpu()
        self.means.data = torch.tensor(self.kmeans.cluster_centers_).to(dev)
        self.post_epoch_features = []
        self.post_epoch_weak_features = []
        total_samples = sum(self.kmeans.counts_)
        cw = total_samples / (self.k * self.kmeans.counts_)
        # self.loss_pseudo_cls = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=fts.device))
        return labels

    def _ood_forward_train(self, bbox_results, bbox_targets, bbox_weak_results, bbox_weak_targets, device):
        selected_fg_samples = (bbox_targets[0] != self.k).nonzero().view(-1)
        indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
        gt_classes_numpy = bbox_targets[0].cpu().numpy().astype(int)

        if bbox_weak_results is not None:
            weak_selected_fg_samples = (bbox_weak_targets[0] != self.k).nonzero().view(-1)
            weak_indices_numpy = weak_selected_fg_samples.cpu().numpy().astype(int)
            weak_gt_classes_numpy = bbox_weak_targets[0].cpu().numpy().astype(int)

            weak_box_features = []
            for index in weak_indices_numpy:
                weak_box_features.append(bbox_weak_results['shared_bbox_feats'][index].view(1, -1))
            if len(weak_box_features) == 0:
                weak_box_features = None
            else:
                weak_box_features = torch.cat(weak_box_features, dim=0)
        else:
            weak_box_features = None
            weak_indices_numpy = None
            weak_gt_classes_numpy = None

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

            if weak_box_features is not None:
                weak_pseudo_logits = self.pseudo_score(weak_box_features)
                weak_pseudo_labels = bbox_weak_targets[0][weak_selected_fg_samples]
                gt_pseudo_logits = torch.cat([gt_pseudo_logits, weak_pseudo_logits], dim=0)
                gt_pseudo_labels = torch.cat([gt_pseudo_labels, weak_pseudo_labels], dim=0)
            loss_pseudo_score = self.loss_pseudo_cls(gt_pseudo_logits, gt_pseudo_labels.long())

            sum_temp = 0
            for index in range(self.k):
                sum_temp += self.number_dict[index]
            queue_ready = sum_temp >= self.k * self.vos_samples_per_class
            if not queue_ready:
                for index in indices_numpy:
                    fts = bbox_results['shared_bbox_feats'][index].detach()
                    dict_key = gt_classes_numpy[index]
                    if self.number_dict[dict_key] < self.vos_samples_per_class:
                        self.data_dict[dict_key][self.number_dict[dict_key]] = fts
                        self.number_dict[dict_key] += 1
                if weak_indices_numpy is not None:
                    for index in weak_indices_numpy:
                        fts = bbox_weak_results['shared_bbox_feats'][index].detach()
                        dict_key = weak_gt_classes_numpy[index]
                        if self.number_dict[dict_key] < self.vos_samples_per_class:
                            self.data_dict[dict_key][self.number_dict[dict_key]] = fts
                            self.number_dict[dict_key] += 1
            else:
                for index in indices_numpy:
                    fts = bbox_results['shared_bbox_feats'][index].detach()
                    dict_key = gt_classes_numpy[index]
                    self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                          fts.view(1, -1)), 0)
                if weak_indices_numpy is not None:
                    for index in weak_indices_numpy:
                        fts = bbox_weak_results['shared_bbox_feats'][index].detach()
                        dict_key = weak_gt_classes_numpy[index]
                        self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                              fts.view(1, -1)), 0)
                if self.epoch >= self.start_epoch:
                    for index in range(self.k):
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
                    ood_samples = None
                    for index in range(self.k):
                        new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                            mean_embed_id[index], covariance_matrix=temp_precision)
                        for _ in range(self.repeat_ood_sampling):
                            negative_samples = new_dis.rsample((self.negative_sampling_size,))
                            prob_density = new_dis.log_prob(negative_samples)

                            # keep the data in the low density area.
                            cur_samples, index_prob = torch.topk(- prob_density, self.bottomk_epsilon_dist)
                            if ood_samples is None:
                                ood_samples = negative_samples[index_prob]
                            else:
                                ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
                        del new_dis
                        del negative_samples

                    energy_score_for_fg = torch.logsumexp(gt_pseudo_logits, 1)

                    # Now we need to get the class logits for the negative samples.
                    predictions_ood = self.pseudo_score(ood_samples)
                    energy_score_for_bg = torch.logsumexp(predictions_ood, 1)

                    input_for_loss = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                    id_labels_size = energy_score_for_fg.shape[0]
                    labels_for_loss = torch.cat((torch.ones(id_labels_size).to(device),
                                                 torch.zeros(len(ood_samples)).to(device)), -1)

                    output = self.logistic_regression_layer(input_for_loss.view(-1, 1))
                    ood_reg_loss = F.binary_cross_entropy_with_logits(
                        output.view(-1), labels_for_loss)

        return ood_reg_loss, loss_pseudo_score

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False,
                           with_ood=True):
        """Test only det bboxes without augmentation."""
        rpn_score = torch.cat([p[:, -1:] for p in proposals], 0)
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        bbox_score = bbox_results['bbox_score']
        # OOD
        inter_feats = self.pseudo_score(bbox_results['shared_bbox_feats'])  # N x (K + 1)
        energy = torch.logsumexp(inter_feats, 1)
        ood_scores = self.logistic_regression_layer(energy.view(-1, 1))

        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_score = bbox_score.split(num_proposals_per_img, 0)
        ood_scores = ood_scores.split(num_proposals_per_img, 0)
        rpn_score = rpn_score.split(num_proposals_per_img, 0)
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
        for i in range(len(proposals)):
            oods = F.sigmoid(ood_scores[i][:, 0]) if cls_score is not None else None
            det_bbox, det_label, _ood_scores = self.bbox_head.get_bboxes(
                rois[i],
                inter_feats[i],
                bbox_pred[i],
                oods,
                bbox_score[i],
                rpn_score[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg,
                with_ood=with_ood)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            det_ood_scores.append(_ood_scores)
        if with_ood:
            return det_bboxes, det_labels, det_ood_scores
        else:
            return det_bboxes, det_labels, None

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    with_ood=True):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels, det_ood_scores = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale, with_ood=with_ood)
        # det_ood_scores = self.simple_test_ood(x, proposal_list)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, det_ood_scores, segm_results,
            else:
                return det_bboxes, det_labels, det_ood_scores
        if with_ood:
            bbox_results = [
                bbox2result_ood(det_bboxes[i], det_labels[i], det_ood_scores[i],
                                self.bbox_head.num_classes)
                for i in range(len(det_bboxes))
            ]
        else:
            bbox_results = [
                bbox2result(det_bboxes[i], det_labels[i], self.bbox_head.num_classes)
                for i in range(len(det_bboxes))
            ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))
