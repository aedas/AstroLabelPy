import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import torch
from torchvision import transforms
import cv2
from torch.nn.functional import conv2d


def init_padded_matrix(a, ksize):
    assert ksize % 2 == 1
    p = int((ksize - 1) / 2)
    res = np.zeros((a.shape[0] + 2 * p, a.shape[1] + 2 * p))
    res[p:-p, p:-p] = a
    at = a.T
    res[:p,p:-p]=a[0,:]
    res[-p:,p:-p]=a[-1,:]
    rest = res.T
    rest[:p,p:-p]=at[0,:]
    rest[-p:,p:-p]=at[-1,:]
    return rest.T


def get_avg_var(a, ksize):
    assert ksize % 2 == 1
    p = int((ksize - 1) / 2)
    ap = init_padded_matrix(a, ksize)
    ap2 = ap ** 2
    eap = cv2.blur(ap, (ksize, ksize))
    eap2 = cv2.blur(ap2, (ksize, ksize))
    v = eap2 - eap ** 2
    m = eap
    #print(v[p:-p, p:-p].shape)
    return m[p:-p, p:-p], v[p:-p, p:-p]




def apply(a, ksize, minvar):
    assert ksize % 2 == 1
    m, v = get_avg_var(a, ksize)
    res = a.copy()

    res[v < minvar] = res[v < minvar] - m[v < minvar]
    res[v >= minvar] = (res[v >= minvar] - m[v >= minvar]) / (v[v >= minvar] ** 0.25)

    res = cv2.GaussianBlur(np.array(res), (5,5), 0)
    #plt.imshow(res)
    #plt.show()
    return res

def runKmeans(img):

    k = 2
    res2_ = None
    arr = [2]
    ready = False
    attempts = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    while True:
        ret, label, center = cv2.kmeans(img, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

        center = np.uint8(center)

        res_ = center[label.flatten()]
        res2_ = res_.reshape(thres.shape)
        res2_ = res2_ - np.min(res2_)
        #print(np.average(res2_))
        res2_ = res2_ / np.max(res2_)
        res2_ = res2_ ** 2
        #print(k)
        if ready:
            break
        arr.append(np.average(res2_))
        #print(arr[-1])
        if arr[-1] >= arr[-2]:
            k -= 1
            ready = True
        else:
            k += 1
    return res2_


bar = 80/255
img = Image.open("../img/Summer Triangle Advanced.png")
pixels = (transforms.ToTensor())(img)

thres = torch.sqrt(0.241 * pixels[0] ** 2 + 0.691 * pixels[1] ** 2 + 0.068 * pixels[2] ** 2)
mask = torch.clone(thres)
mask[mask > bar] = 1
mask[mask <= bar] = 0


print("Normalizing layer")
thres = cv2.GaussianBlur(np.array(thres), (5,5), 0)
thres = (thres - np.average(thres)) / np.sqrt(np.var(thres))
thres = torch.tensor(thres)

print("Applying kernel")
thres = apply(np.array(thres), 51, 0.5)
thres = apply(np.array(thres), 101, 0.15)
thres = apply(np.array(thres), 201, 0)
thres = torch.tensor(thres)
thres = thres - torch.min(thres)
thres = thres / torch.max(thres)

test = np.array((thres) * 255, dtype=np.uint8)


thres[thres > bar] = 1
thres[thres <= bar] = 0
plt.imshow(thres)
image = Image.fromarray(np.array(thres * 255, dtype=np.uint8))
image.save('test.png')



print("Applying kmeans")
test2 = test.reshape(-1)
test2 = np.float32(test2)

mid = runKmeans(test2)

print("Normalizing layer")
#mid = cv2.GaussianBlur(mid, (5, 5), 0)
mid = (mid - np.average(mid)) / np.var(mid)

print("Applying kernel")
mid = apply(np.array(mid), 51, 0.3)
mid = mid - np.min(mid)
mid = mid / np.max(mid)
mid = mid * 255

print("Applying kmeans")
mid2 = mid.reshape(-1)
mid2 = np.float32(mid2)
fin = runKmeans(mid2)
fin = fin * 255
fin = np.uint8(fin)

plt.imshow(fin)
plt.show()
cv2.imwrite('segmented.png', fin)

