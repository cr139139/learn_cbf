import io
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_2d_circle_image(x_circle, r_circle, real_demonstration):
    # x_circle: n x 2
    # r_circle : n x 1
    figure, axes = plt.subplots()
    if real_demonstration:
        axes.set_xlim([-4.75, -4.])
        axes.set_ylim([-0.5, 1.])
    else:
        axes.set_xlim([0, 1])
        axes.set_ylim([0, 1])
    axes.set_aspect(1)
    axes.axis('off')

    for i in range(x_circle.shape[0]):
        axes.add_artist(plt.Circle(x_circle[i], r_circle[i], color='k'))

    buf = io.BytesIO()
    figure.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    image = cv2.imdecode(image, 1)
    plt.close(figure)

    return image


def get_evaluation_metric(x_source, r_source, x_target, r_target, viz=False, real_demonstration=False):
    # x_source: n x 2
    # r_source : n x 1
    # x_target: m x 2
    # r_target : m x 1

    image_source = get_2d_circle_image(x_source, r_source, real_demonstration)
    mask_source = image_source[:, :, 0] == 0

    image_target = get_2d_circle_image(x_target, r_target, real_demonstration)
    mask_target = image_target[:, :, 0] == 0

    intersection = np.logical_and(mask_source, mask_target)
    union = np.logical_or(mask_source, mask_target)

    iou = intersection.sum() / (union.sum() + 1e-6)
    sot = intersection.sum() / (mask_target.sum() + 1e-6)
    tos = intersection.sum() / (mask_source.sum() + 1e-6)

    if viz:
        cv2.imshow('source', mask_source * 1.)
        cv2.imshow('target', mask_target * 1.)
        cv2.imshow('intersection', intersection * 1.)
        cv2.imshow('union', union * 1.)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return iou, sot, tos


