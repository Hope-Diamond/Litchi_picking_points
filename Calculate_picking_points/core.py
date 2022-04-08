<<<<<<< HEAD
# 文件名core.py
import numpy as np
def leastSquare(x, y):
    if len(x) == 2:
        # At this time, x is the natural sequence
        sx = 0.5 * (x[1] - x[0] + 1) * (x[1] + x[0])
        ex = sx / (x[1] - x[0] + 1)
        sx2 = ((x[1] * (x[1] + 1) * (2 * x[1] + 1))
               - (x[0] * (x[0] - 1) * (2 * x[0] - 1))) / 6
        x = np.array(range(x[0], x[1] + 1))
    else:
        sx = sum(x)
        ex = sx / len(x)
        sx2 = sum(x ** 2)

    sxy = sum(x * y)
    ey = np.mean(y)

    a = (sxy - ey * sx) / (sx2 - ex * sx)
    b = (ey * sx2 - sxy * ex) / (sx2 - ex * sx)
    return a, b

=======
# 文件名core.py
import numpy as np
def leastSquare(x, y):
    if len(x) == 2:
        # At this time, x is the natural sequence
        sx = 0.5 * (x[1] - x[0] + 1) * (x[1] + x[0])
        ex = sx / (x[1] - x[0] + 1)
        sx2 = ((x[1] * (x[1] + 1) * (2 * x[1] + 1))
               - (x[0] * (x[0] - 1) * (2 * x[0] - 1))) / 6
        x = np.array(range(x[0], x[1] + 1))
    else:
        sx = sum(x)
        ex = sx / len(x)
        sx2 = sum(x ** 2)

    sxy = sum(x * y)
    ey = np.mean(y)

    a = (sxy - ey * sx) / (sx2 - ex * sx)
    b = (ey * sx2 - sxy * ex) / (sx2 - ex * sx)
    return a, b

>>>>>>> b201b31cc22f62ca79d4167c14e19308f24f3ead
