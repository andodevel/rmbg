import os

import matplotlib
import matplotlib.pyplot as plt

import config as cfg

if os.name == "nt":
    matplotlib.use("TKAgg")


# event listener
def press(event):
    if event.key == 'escape' or event.key == 'enter':
        plt.close()


def display_single_image(image, title="Single image"):
    if not cfg.visual_debug_enabled:
        return

    f = plt.figure(title, figsize=(8, 8))
    f.canvas.mpl_connect('key_press_event', press)

    plt.axis("off")
    plt.imshow(image)

    plt.tight_layout()
    plt.show()


def display_two_images(image1, image2, title1="Image 1", title2="Image 2"):
    if not cfg.visual_debug_enabled:
        return

    f = plt.figure("Images", figsize=(12, 6))
    f.canvas.mpl_connect('key_press_event', press)

    ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
    ax1.axis("off")
    ax1.set_title(title1)
    ax1.imshow(image1)

    ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)
    ax2.axis("off")
    ax2.set_title(title2)
    ax2.imshow(image2)

    plt.tight_layout()
    plt.show()


def display_colors_summary(image_org, foreground_image, foreground_colors, fuzzy_edge_image, fuzzy_edge_colors):
    if not cfg.visual_debug_enabled:
        return

    f = plt.figure("Colors summary", figsize=(14, 12))
    f.canvas.mpl_connect('key_press_event', press)

    ax1 = plt.subplot2grid((3, 7), (0, 2), colspan=3, rowspan=1)
    ax1.axis("off")
    ax1.set_title("Original image")
    ax1.imshow(image_org)

    ax2 = plt.subplot2grid((3, 7), (1, 0), colspan=3, rowspan=1)
    ax2.axis("off")
    ax2.set_title("Foreground image")
    ax2.imshow(foreground_image)

    ax3 = plt.subplot2grid((3, 7), (1, 4), colspan=3, rowspan=1)
    ax3.axis("off")
    ax3.set_title("Foreground colors summary")
    ax3.imshow(foreground_colors)

    ax4 = plt.subplot2grid((3, 7), (2, 0), colspan=3, rowspan=1)
    ax4.axis("off")
    ax4.set_title("Fuzzy edge detection")
    ax4.imshow(fuzzy_edge_image)

    ax5 = plt.subplot2grid((3, 7), (2, 4), colspan=3, rowspan=1)
    ax5.axis("off")
    ax5.set_title("Fuzzy edge colors")
    ax5.imshow(fuzzy_edge_colors)

    plt.tight_layout()
    plt.show()
