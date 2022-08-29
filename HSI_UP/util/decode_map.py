import numpy as np
import spectral
import matplotlib.pyplot as plt


def map_result(data):
    # colors = np.array([[0, 0, 0],
    #                    [255, 0, 0],
    #                    [0, 255, 0],
    #                    [0, 0, 255],
    #                    [255, 255, 0],
    #                    [0, 255, 255],
    #                    [255, 0, 255],
    #                    [192, 192, 192],
    #                    [128, 128, 128],
    #                    [128, 0, 0],
    #                    [128, 128, 0],
    #                    [0, 128, 0],
    #                    [128, 0, 128],
    #                    [0, 128, 128],
    #                    [0, 0, 128],
    #                    [255, 165, 0],
    #                    [255, 215, 0]])
    colors = np.array([[255, 255, 255],
                       [255, 218, 185],
                       [150, 205, 205],
                       [0, 229, 238],
                       [0, 139, 139],
                       [0, 0, 205],
                       [0, 255, 0],
                       [255, 255, 0],
                       [255, 106, 106],
                       [255, 69, 0],
                       [255, 0, 0],
                       [205, 38, 38],
                       [205, 0, 205],
                       [139, 0, 139],
                       [105, 105, 105],
                       [79, 79, 79],
                       [54, 54, 54]])
    img = spectral.imshow(classes=data.astype(int), figsize=(19, 19), colors=colors)
    plt.show()
