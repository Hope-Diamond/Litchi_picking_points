import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from cfg_mAP import Cfg

cfg = Cfg

x = np.linspace(1, 10, 10)
ap_systhesis_valid = []
ap_manual = []
plt.figure(num=1, figsize=(8, 5), )

with open(os.path.join(cfg.manual_cachedir, 'cls_ap.pkl'), 'rb') as out_data:
    # 按保存变量的顺序加载变量
    manual_cls_ap = pickle.load(out_data)
    print(manual_cls_ap)  # dataList
    print(len(manual_cls_ap))  # dataList

for cls in cfg.names:
    if cls in manual_cls_ap.keys():
        ap_manual.append(manual_cls_ap[cls])
    else:
        ap_manual.append(0.0)
print('ap_manual: ', ap_manual)
manual_mAP = np.mean(ap_manual)
l2, = plt.plot(x, ap_manual, color='k', linewidth=1.0, linestyle='-.', label='manual_AP')
plt.scatter(x, ap_manual, s=10, color='k')
for x1, y1 in zip(x, ap_manual):
    plt.text(x1, y1, '%s' % str('{0:.3f}'.format(y1)), fontdict={'fontsize': 14}, verticalalignment="bottom",
             horizontalalignment="center")

plt.annotate(r'manual_mAP=%s' % str('{0:.3f}'.format(manual_mAP)), xy=(5, manual_mAP), xycoords='data',
             xytext=(0.0, 0.0),
             textcoords='offset points', fontsize=13, )

plt.xticks(np.linspace(1, 10, 10),
           [r'combustion_lining', r'fan', r'fan_support', r'hp_core_casing', r'hpc_spool',
            r'hpc_stage5', r'mixer', r'nozzle', r'nozzle_cone', r'stand'])

plt.legend(handles=[l2], loc='best')

plt.show()
