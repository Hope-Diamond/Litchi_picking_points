import matplotlib.pyplot as plt
import numpy as np

with open('../runs/train/exp/results.txt', 'r') as out_data:
    text = out_data.readlines()  # 结果为str类型
loss = []
for ss in text:
    ss = ss.strip()
    ss = ss.split()
    strr = ss[2:6] + ss[8:12]
    numbers = list(map(float, strr))
    loss.append(numbers)

# 0-GIoU, 1-obj, 2-cls, 3-total, 4-P, 5-R, 6-mAP@.5, 7-mAP@.5:.95
loss = np.array(loss)

epoch_n = len(loss)
x = np.linspace(1, epoch_n, epoch_n)

GIoU = loss[:, 0]
obj = loss[:, 1]
cls = loss[:, 2]
total = loss[:, 3]
P = loss[:, 4]
R = loss[:, 5]
mAP_5 = loss[:, 6]
mAP_5_95 = loss[:, 7]

plt.figure(num=1, figsize=(16, 10), )
plt.subplot(4, 2, 1)
plt.plot(x, GIoU, color='red', linewidth=1.0, linestyle='--', label='GIoU')
plt.legend(loc='upper right')

plt.subplot(4, 2, 2)
plt.plot(x, obj, color='red', linewidth=1.0, linestyle='--', label='obj')
plt.legend(loc='upper right')

plt.subplot(4, 2, 3)
plt.plot(x, cls, color='red', linewidth=1.0, linestyle='--', label='cls')
plt.legend(loc='upper right')

plt.subplot(4, 2, 4)
plt.plot(x, total, color='red', linewidth=1.0, linestyle='--', label='total')
plt.legend(loc='upper right')

plt.subplot(4, 2, 5)
plt.plot(x, P, color='red', linewidth=1.0, linestyle='--', label='P')
plt.legend(loc='upper right')

plt.subplot(4, 2, 6)
plt.plot(x, R, color='red', linewidth=1.0, linestyle='--', label='R')
plt.legend(loc='upper right')

plt.subplot(4, 2, 7)
plt.plot(x, mAP_5, color='red', linewidth=1.0, linestyle='--', label='mAP_5')
plt.legend(loc='upper right')

plt.subplot(4, 2, 8)
plt.plot(x, mAP_5_95, color='red', linewidth=1.0, linestyle='--', label='mAP_5_95')
plt.legend(loc='upper right')

plt.show()
