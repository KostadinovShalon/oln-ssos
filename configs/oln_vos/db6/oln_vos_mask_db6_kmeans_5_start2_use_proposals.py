_base_ = "./oln_vos_mask_db6_kmeans_5.py"

model = dict(
    roi_head=dict(
        use_all_proposals_ood=True,
    )
)

load_from = './work_dirs/oln_mask/epoch_8.pth'