_base_ = "./oln_vos_box_db6_kmeans_5.py"

model = dict(
    roi_head=dict(
        k=100,
        repeat_ood_sampling=1,
    )
)