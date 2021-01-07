import numpy as np

def generate_anchor_boxes(resolutions, aspect_ratios, steps = 16):
    anchors = []
    mask = []

    x = np.arange(start = 16, stop = 640, step = 16)
    y = np.arange(start = 16, stop = 480, step = 16)

    for j in y:
        for i in x:
            for r in resolutions:
                for a1, a2 in aspect_ratios:
                    dx =  r * a1 / (a1 + a2)
                    dy =  r * a2 / (a1 + a2)
                    dx = dx/2
                    dy = dy/2
                    anchors.append([i - dx, i + dx, j - dy, j + dy])

                    if i - dx <= 0 or i + dx >= 640 or j - dy <= 0 or j + dy >= 480:
                        mask.append(0)
                    else:
                        mask.append(1)
    return anchors, mask
