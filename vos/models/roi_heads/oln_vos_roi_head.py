import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans, KMeans

from mmdet.core import bbox2roi
from mmdet.models import HEADS, OlnRoIHead, build_loss
import warnings
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
        return [bboxes_ood[labels == i, :] for i in range(num_classes)]


@HEADS.register_module()
class OLNKMeansVOSRoIHead(OlnRoIHead):

    def __init__(self,
                 start_epoch=0,
                 logistic_regression_hidden_dim=512,
                 negative_sampling_size=10000,
                 bottomk_epsilon_dist=1,
                 ood_loss_weight=0.1,
                 k=5,
                 recalculate_pseudolabels_every_epoch=1,
                 k_means_minibatch=True,
                 repeat_ood_sampling=4,
                 use_all_proposals_ood=False,
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
        self.start_epoch = start_epoch
        self.bottomk_epsilon_dist = bottomk_epsilon_dist
        self.negative_sampling_size = negative_sampling_size
        self.ood_loss_weight = ood_loss_weight

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

        self.data_list = []

        self.samples_for_covariance = 20 * 1024
        self.ft_minibatches = []
        self.k_means_batches_to_restart = 20
        self.kmeans_minibatches_passed = 0

        self.pseudo_score = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.k)
        )

        for m in self.pseudo_score.modules():
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

        self.means = None
        self.cov = None
        self.kmeans = None

        self.loss_pseudo_cls = torch.nn.CrossEntropyLoss()
        self.use_all_proposals_ood = use_all_proposals_ood

        self.post_epoch_features = []

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
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'],
                                        bbox_results['bbox_score'],
                                        rois,
                                        *bbox_targets)
        # VOS STARTS HERE
        ood_loss, pseudo_loss = self._ood_forward_train(bbox_results, bbox_targets, device=x[0].device)
        loss_bbox["loss_ood"] = self.ood_loss_weight * ood_loss
        loss_bbox["loss_pseudo_class"] = pseudo_loss
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def accumulate_pseudo_labels(self, fts, rois):
        bbox_feats = self.bbox_roi_extractor(
            fts[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        _, _, _, shared_fts = self.bbox_head(bbox_feats)
        self.post_epoch_features.append(shared_fts)

    def calculate_pseudo_labels(self):
        fts = torch.cat(self.post_epoch_features, dim=0)
        if self.means is None:
            self.kmeans = MiniBatchKMeans(n_clusters=self.k, n_init=1, batch_size=1024).fit(fts.cpu())
        else:
            self.kmeans = MiniBatchKMeans(n_clusters=self.k, n_init=1, batch_size=1024,
                                          init=self.kmeans.cluster_centers_).fit(fts.cpu())
        self.means = torch.tensor(self.kmeans.cluster_centers_).to(fts.device).detach()
        self.post_epoch_features = []
        total_samples = sum(self.kmeans.counts_)
        cw = total_samples / (self.k * self.kmeans.counts_)
        self.loss_pseudo_cls = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=fts.device))

    def _ood_forward_train(self, bbox_results, bbox_targets, device):
        n_classes = bbox_results['cls_score'].shape[1]
        selected_fg_samples = (bbox_targets[0] != n_classes - 1).nonzero().view(-1)
        indices_numpy = selected_fg_samples.cpu().numpy().astype(int)

        gt_box_features = []
        for index in indices_numpy:
            gt_box_features.append(bbox_results['shared_bbox_feats'][index].view(1, -1))
            self.data_list.append(bbox_results['shared_bbox_feats'][index].detach().view(1, -1))
        gt_box_features = torch.cat(gt_box_features, dim=0)
        if len(self.data_list) > self.samples_for_covariance:
            self.data_list = self.data_list[-self.samples_for_covariance:]

        if self.kmeans is not None and len(self.data_list) >= self.samples_for_covariance:
            data_tensor = torch.cat(self.data_list, dim=0)
            X = []
            data_labels = self.kmeans.predict(data_tensor.cpu()) if self.k_means_minibatch \
                else self.kmeans.labels_
            for i, mean in enumerate(self.means):
                label_idx = data_labels == i
                label_idx = torch.tensor(label_idx).to(data_tensor.device)
                X.append(data_tensor[label_idx] - mean)
            X = torch.cat(X, dim=0).detach()
            # add the variance.
            self.cov = torch.mm(X.t(), X) / len(X)
            # for stable training.
            self.cov += 0.0001 * torch.eye(self.bbox_head.fc_out_channels, device=device)
            if not self.k_means_minibatch:
                self.data_list = []

        ood_reg_loss = torch.zeros(1).to(device)
        loss_pseudo_score = torch.zeros(1).cuda()
        ood_samples = []
        if self.kmeans is not None:
            if not self.use_all_proposals_ood:
                gt_pseudo_labels = self.kmeans.predict(gt_box_features.detach().cpu())
                gt_pseudo_logits = self.pseudo_score(gt_box_features)
            else:
                gt_pseudo_labels = self.kmeans.predict(bbox_results['shared_bbox_feats'].detach().cpu())
                gt_pseudo_logits = self.pseudo_score(bbox_results['shared_bbox_feats'])
            gt_pseudo_labels = torch.tensor(gt_pseudo_labels).to(gt_box_features.device)
            loss_pseudo_score = self.loss_pseudo_cls(gt_pseudo_logits, gt_pseudo_labels.long())

            if self.epoch >= self.start_epoch and self.cov is not None:
                for i, mean in enumerate(self.means):
                    new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                        mean, covariance_matrix=self.cov)
                    repeat_factor = 10 if self.use_all_proposals_ood else 1
                    for _ in range(self.repeat_ood_sampling * repeat_factor):
                        negative_samples = new_dis.rsample((self.negative_sampling_size,))
                        prob_density = new_dis.log_prob(negative_samples)

                        # keep the data in the low density area.
                        cur_samples, index_prob = torch.topk(- prob_density, self.bottomk_epsilon_dist)
                        ood_samples.append(negative_samples[index_prob])
                        del negative_samples
                    del new_dis
                ood_samples = torch.cat(ood_samples, dim=0).to(torch.float32)

                energy_score_for_fg = torch.logsumexp(gt_pseudo_logits, 1)

                # Now we need to get the class logits for the negative samples.
                predictions_ood = self.pseudo_score(ood_samples)
                energy_score_for_bg = torch.logsumexp(predictions_ood, 1)

                input_for_loss = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                id_labels_size = len(selected_fg_samples) if not self.use_all_proposals_ood else \
                    len(bbox_results['shared_bbox_feats'])
                labels_for_loss = torch.cat((torch.ones(id_labels_size).to(device),
                                           torch.zeros(len(ood_samples)).to(device)), -1)

                output = self.logistic_regression_layer(input_for_loss.view(-1, 1))
                ood_reg_loss = F.binary_cross_entropy_with_logits(
                    output.view(-1), labels_for_loss
                    , pos_weight=torch.tensor(len(ood_samples)/len(selected_fg_samples)).cuda())
        return ood_reg_loss, loss_pseudo_score

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
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
                cls_score[i],
                bbox_pred[i],
                oods,
                bbox_score[i],
                rpn_score[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            det_ood_scores.append(_ood_scores)
        return det_bboxes, det_labels, det_ood_scores

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels, det_ood_scores = self.simple_test_bboxes(
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
            bbox2result_ood(det_bboxes[i], det_labels[i], det_ood_scores[i],
                            self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))
