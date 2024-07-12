import matplotlib
from matplotlib import pyplot as plt

samples = [50, 100, 200, 300, 500, 1000]
ar10 = [16.6, 22.7, 11, 27.3, 18.6, 20]
ar100 = [23.8, 32.5, 13.5, 35.6, 26, 26.9]
plt.figure()
plt.plot(samples, ar10, 'Dr-', linewidth=1.5, label="AR@10")
plt.plot(samples, ar100, 'ob-', linewidth=1.5, label="AR@100")
font = {
    'weight': 'bold',
    'size': 15}
matplotlib.rc('font', **font)
plt.title("Effect of the outlier sampling size \n(OLN-FFS on the SIXRay10 Dataset)")
plt.legend()
plt.xlabel("Sampling size", fontsize=15)
plt.ylabel("Average Recall", fontsize=15)
plt.xticks(samples)
plt.xscale('log')
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xticks(samples)
plt.savefig("sixray10_ablations.pdf", bbox_inches='tight')




samples = [50, 100, 200, 300, 500, 1000]
ar10 = [1.2, 5.2, 10.9, 12.3, 10.9, 10.1]
ar100 = [1.4, 8.1, 13.5, 12.8, 10.4, 12.4]
plt.figure()
plt.plot(samples, ar10, 'Dr-', linewidth=1.5, label="AR@10")
plt.plot(samples, ar100, 'ob-', linewidth=1.5, label="AR@100")
font = {
    'weight': 'bold',
    'size': 15}
matplotlib.rc('font', **font)
plt.title("Effect of the outlier sampling size \n(OLN-FFS on the LTDImaging Dataset)")
plt.legend()
plt.xlabel("Sampling size", fontsize=15)
plt.ylabel("Average Recall", fontsize=15)
plt.xticks(samples)
plt.xscale('log')
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xticks(samples)
plt.savefig("ltdimaging_ablations.pdf", bbox_inches='tight')


samples = [50, 100, 200, 300, 500, 1000]
ar10 = [44.9, 27.7, 42.5, 35.9, 39.2, 38.6]
ar100 = [49.8, 35, 47, 46.3, 43.4, 44.8]
plt.figure()
plt.plot(samples, ar10, 'Dr-', linewidth=1.5, label="AR@10")
plt.plot(samples, ar100, 'ob-', linewidth=1.5, label="AR@100")
font = {
    'weight': 'bold',
    'size': 15}
matplotlib.rc('font', **font)
plt.title("Effect of the outlier sampling size \n(OLN-FFS on the DBF6 (Box) Dataset)")
plt.legend()
plt.xlabel("Sampling size", fontsize=15)
plt.ylabel("Average Recall", fontsize=15)
plt.xticks(samples)
plt.xscale('log')
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xticks(samples)
plt.savefig("db6box_ablations.pdf", bbox_inches='tight')

samples = [50, 100, 200, 300, 500, 1000]
ar10 = [53.4, 51.3, 46, 57.6, 50.7, 55.7]
ar100 = [59.6, 56.5, 52.4, 57.2, 56.8, 61.9]
plt.figure()
plt.plot(samples, ar10, 'Dr-', linewidth=1.5, label="AR@10")
plt.plot(samples, ar100, 'ob-', linewidth=1.5, label="AR@100")
font = {
    'weight': 'bold',
    'size': 15}
matplotlib.rc('font', **font)
plt.title("Effect of the outlier sampling size \n(OLN-FFS on the DBF6 (Mask) Dataset)")
plt.legend()
plt.xlabel("Sampling size", fontsize=15)
plt.ylabel("Average Recall", fontsize=15)
plt.xticks(samples)
plt.xscale('log')
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xticks(samples)
plt.savefig("db6mask_ablations.pdf", bbox_inches='tight')


samples = [50, 100, 200, 300, 500, 1000]
ar10_k100 = [9.6,7.6,9.4,11.2, 9.9, 9.2]
ar100_k100 = [15.1, 11.8, 16.3, 17.8, 16.4, 16]

ar10_k5 = [5.3, 5, 7.6, 6.4, 7.6, 4.3]
ar100_k5 = [10.9, 11.3, 13.8, 11.8, 14, 8.4]

ar10_k10 = [4.5, 9, 6.4, 5.6, 7.7, 5.2]
ar100_k10 = [7.8, 15.5, 10.4, 10.9, 13.6, 7.6]

plt.figure()
plt.plot(samples, ar10_k5, 'Dr-', linewidth=1.5, label="AR@10 k=5")
plt.plot(samples, ar100_k5, 'or--', linewidth=1.5, label="AR@100 k=5")

plt.plot(samples, ar10_k10, 'Db-', linewidth=1.5, label="AR@10 k=10")
plt.plot(samples, ar100_k10, 'ob--', linewidth=1.5, label="AR@100 k=10")

plt.plot(samples, ar10_k100, '^k-', linewidth=1.5, label="AR@10 k=100")
plt.plot(samples, ar100_k100, '^k--', linewidth=1.5, label="AR@100 k=100")

font = {
    'weight': 'bold',
    'size': 15}
matplotlib.rc('font', **font)
plt.title("Effect of the number of pseudo-classes \nin the VOC/COCO experiment")
font = {
    'weight': 'normal',
    'size': 10}
matplotlib.rc('font', **font)
plt.legend()
plt.xlabel("Sampling size", fontsize=15)
plt.ylabel("Average Recall", fontsize=15)
plt.xticks(samples)
plt.xscale('log')
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xticks(samples)
plt.savefig("voccoco_ablations.pdf", bbox_inches='tight')
#
# samples = [5, 10, 100]
# oln_vos_box_ar10 = [20.9, 12.6, 25.8]
# oln_vos_box_ar100 = [30.5, 18.4, 35.3]
#
# oln_ffs_box_ar10 = [13.5, 23.5, 27.3]
# oln_ffs_box_ar100 = [18.5, 33, 35.6]
# plt.figure()
# plt.plot(samples, oln_vos_box_ar10, 'Dr-', linewidth=1.5, label="OLN-VOS Box AR@10")
# plt.plot(samples, oln_vos_box_ar100, 'or--', linewidth=1.5, label="OLN-VOS Box AR@100")
#
# plt.plot(samples, oln_ffs_box_ar10, 'Db-', linewidth=1.5, label="OLN-FFS Box AR@10")
# plt.plot(samples, oln_ffs_box_ar100, 'ob--', linewidth=1.5, label="OLN-FFS Box AR@100")
#
# font = {
#     'weight': 'bold',
#     'size': 15}
# matplotlib.rc('font', **font)
# plt.title("Effect of the number of pseudo-classes \nin the SIXRay10 dataset")
# font = {
#     'weight': 'normal',
#     'size': 10}
# matplotlib.rc('font', **font)
# plt.legend()
# plt.xlabel("Pseudo Classes", fontsize=15)
# plt.ylabel("Average Recall", fontsize=15)
# plt.xticks(samples)
# plt.xscale('log')
# ax = plt.gca()
# ax.tick_params(axis='both', which='major', labelsize=12)
# ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# plt.xticks(samples)
# plt.show()
#
# samples = [5, 10, 100]
# oln_vos_box_ar10 = [10.4, 12.2, 5.2]
# oln_vos_box_ar100 = [15.7, 18.2, 6.6]
#
# oln_ffs_box_ar10 = [12.3, 5.2, 4.1]
# oln_ffs_box_ar100 = [12.8, 8.1, 10.4]
# plt.figure()
# plt.plot(samples, oln_vos_box_ar10, 'Dr-', linewidth=1.5, label="OLN-VOS Box AR@10")
# plt.plot(samples, oln_vos_box_ar100, 'or--', linewidth=1.5, label="OLN-VOS Box AR@100")
#
# plt.plot(samples, oln_ffs_box_ar10, 'Db-', linewidth=1.5, label="OLN-FFS Box AR@10")
# plt.plot(samples, oln_ffs_box_ar100, 'ob--', linewidth=1.5, label="OLN-FFS Box AR@100")
#
# font = {
#     'weight': 'bold',
#     'size': 15}
# matplotlib.rc('font', **font)
# plt.title("Effect of the number of pseudo-classes \nin the LTDImaging dataset")
# font = {
#     'weight': 'normal',
#     'size': 10}
# matplotlib.rc('font', **font)
# plt.legend()
# plt.xlabel("Pseudo Classes", fontsize=15)
# plt.ylabel("Average Recall", fontsize=15)
# plt.xticks(samples)
# plt.xscale('log')
# ax = plt.gca()
# ax.tick_params(axis='both', which='major', labelsize=12)
# ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# plt.xticks(samples)
# plt.show()
