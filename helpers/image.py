import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def overlay_segmentation(image: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor, num_classes: int):
    """
    Overlay multi-class segmentation masks (GT + Prediction) on top of an image.

    Args:
        image: Tensor [C, H, W] (normalized)
        mask: Tensor [H, W] (ground truth labels, int)
        pred: Tensor [H, W] (predicted labels, int)
        num_classes: number of classes
    """

    def _make_mask(mask_np: np.ndarray, cmap):
        mask_np = mask_np[..., 0]
        rgba_img = cmap(mask_np / (num_classes - 1))
        # Transparent background
        rgba_img[mask_np == 0, -1] = 0
        return rgba_img

    def _expand_mask(mask: np.ndarray):
        if mask.ndim == 2:
            mask = np.repeat(mask[np.newaxis], 3, axis=0)
        return mask

    image = image.detach().cpu().numpy()
    image = (255 * ((image - image.min()) / (image.max() - image.min() + 1e-8))).astype(np.uint8)
    mask = mask.detach().cpu().numpy().astype(int)
    pred = pred.detach().cpu().numpy().astype(int)

    mask, pred = [_expand_mask(x) for x in [mask, pred]]
    image, mask, pred = [x.transpose(1, 2, 0) for x in [image, mask, pred]]

    # Colormap for masks
    cmap = cm.get_cmap('tab10', num_classes)
    cmap_gt= cm.get_cmap('Greens', num_classes)
    cmap_pred = cm.get_cmap('Blues', num_classes)

    # Make figure
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(image)
    axes[0].imshow(_make_mask(mask, cmap), alpha=0.5)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(_make_mask(pred, cmap), alpha=0.5)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(_make_mask(mask, cmap_gt), alpha=0.5)
    axes[2].imshow(_make_mask(pred, cmap_pred), alpha=0.3)  # overlay both
    axes[2].set_title("GT + Pred")
    axes[2].axis("off")

    fig.tight_layout()

    fig.canvas.draw()
    overlay = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    overlay = overlay.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)


    return overlay
