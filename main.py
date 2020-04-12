import argparse

from bootstrap import do_bootstrap
from helpers import *
from libs.deeplabv3p.model import Deeplabv3
from debug import *
from config import get_logger
from PIL import Image

logger = get_logger()


def generate_renditions(image_org_):
    image_ = resize_image(image_org_, width=512)
    image_gray_ = to_gray_image(image_)
    image_lab_ = rgb_to_lab(image_)

    return image_, image_gray_, image_lab_


def iterate(file_path):
    if not is_image_file(file_path):
        logger.error(f"'{file_path}' is not a valid image file!")

    logger.info(f"Process file '{file_path}'")
    image_org = time_exec("load_image", load_image, file_path)
    if image_org is None:
        logger.error(f"Could not open or find the image: {file_path}")
        return

    _, file_name = os.path.split(file_path)
    image, image_gray, image_lab = generate_renditions(image_org)

    # canny_rate = assess_canny(image, image_gray)
    # logger.info(f"Canny rate: {canny_rate}")

    trained_image_width = 512
    mean_subtraction_value = 127.5

    # resize to max dimension of images from training dataset
    w, h, _ = image.shape
    ratio = float(trained_image_width) / np.max([w, h])
    resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))

    # apply normalization for trained dataset images
    resized_image = (resized_image / mean_subtraction_value) - 1.

    # pad array to square image to match training images
    pad_x = int(trained_image_width - resized_image.shape[0])
    pad_y = int(trained_image_width - resized_image.shape[1])
    resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

    # --- Deeplab
    # Generates labels using most basic setup.  Supports various image sizes.  Returns image labels in same format
    # as original image.  Normalization matches MobileNetV2
    # make prediction
    deeplab_model = Deeplabv3(backbone="xception")
    res = deeplab_model.predict(np.expand_dims(resized_image, 0))
    labels = np.argmax(res.squeeze(), -1)

    # remove padding and resize back to original image
    if pad_x > 0:
        labels = labels[:-pad_x]
    if pad_y > 0:
        labels = labels[:, :-pad_y]
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))

    # Remove noise and expand neighbors.
    # NOTE: Disable because it reduce details from main object, must find a better way
    # from scipy import ndimage
    # labels = ndimage.binary_erosion(labels, structure=np.ones((20, 20))).astype(labels.dtype)
    # labels = ndimage.binary_dilation(labels, structure=np.ones((20, 20))).astype(labels.dtype)

    labels = np.expand_dims(labels, -1)
    # Apply labels mask to the image
    image_splashed = apply_image_mask(labels, image, [0, 0, 0])

    # ----- section: canny -----
    # import canny
    # # Apply labels mask to canny filtered image
    # image_canny = canny.apply(image, image_gray)
    # image_canny_splashed = apply_image_mask(labels, image_canny, image_gray)
    # display_two_images(image_splashed, image_canny_splashed)
    # ----- end section: canny -----

    # ------ section: superpixel ------
    # from skimage.filters import sobel
    # from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
    # from skimage.segmentation import mark_boundaries
    # from skimage.util import img_as_ubyte

    # img = image
    # segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=30)
    # segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
    # segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    # gradient = sobel(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    # segments_watershed = watershed(gradient, markers=250, compactness=0.001)

    # print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
    # print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
    # print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")
    # print(f"Watershed number of segments: {len(np.unique(segments_watershed))}")

    # image_fz = img_as_ubyte(mark_boundaries(image_splashed, segments_fz))
    # image_fz = apply_image_mask(labels, image_fz, image_gray)
    # image_slic = img_as_ubyte(mark_boundaries(img, segments_slic))
    # image_slic = apply_image_mask(labels, image_slic, image_gray)
    # display_two_images(image_fz, image_slic, "Felzenszwalbs's method", "SLIC")

    # image_quick = img_as_ubyte(mark_boundaries(img, segments_quick))
    # image_quick = apply_image_mask(labels, image_quick, image_gray)
    # image_watershed = img_as_ubyte(mark_boundaries(img, segments_watershed))
    # image_watershed = apply_image_mask(labels, image_watershed, image_gray)
    # display_two_images(image_quick, image_watershed, 'Quickshift', 'Compact watershed')
    # display_single_image(image_fz)
    # ------ end section: superpixel ------

    from libs.crfasrnn.crfrnn_model import get_crfrnn_model_def
    import libs.crfasrnn.util as crfasrnn_utils
    model = get_crfrnn_model_def()
    model.load_weights('crfrnn_keras_model.h5')
    img_data, img_h, img_w, size = crfasrnn_utils.get_preprocessed_image(image)
    probs = model.predict(img_data, verbose=False)[0]
    probs = probs.argmax(axis=2).astype('uint8')[:img_h, :img_w]
    probs = np.array(Image.fromarray(probs.astype('uint8')).resize((h, w)))
    # Remove noise and expand neighbors. TODO: use opencvs
    probs = np.expand_dims(probs, -1)
    # image_crfasrnn = crfasrnn_utils.get_label_image(probs, img_h, img_w, size);
    image_crf_splashed = apply_image_mask(probs, image, [0, 0, 0])
    # display_two_images(image_splashed, image_crf_splashed, "deeplabv3", "crfasrnn")

    # ----- start section: simple filer to remove smooth background ----
    # == Parameters =======================================================================
    BLUR = 21
    CANNY_THRESH_1 = 5
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 5
    MASK_ERODE_ITER = 5
    MASK_COLOR = (0.0, 0.0, 0.0)  # In BGR format

    # == Processing =======================================================================

    # -- Read image -----------------------------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # -- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    display_single_image(edges, "canny")

    # -- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Thanks to notes from commenters, I've updated the code but left this note
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    # -- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    # -- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
    img = image.astype('float32') / 255.0  # for easy blending

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
    masked = (masked * 255).astype('uint8')  # Convert back to 8-bit
    # display_single_image(masked)
    # ----- end section: simple filer to remove smooth background ----

    display_four_images(image, masked, image_splashed, image_crf_splashed, "original", "simple", "deeplabv3", "crfasrcnn")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='rmbg v0.1')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--cpu', dest='cpu', action='store_true', help='Force to use CPU.')
    parser.add_argument('input', help='Path to input image or images directory.')
    args = parser.parse_args()
    if args.debug:
        cfg.enable_debug()
    if args.cpu:
        cfg.use_cpu()

    # Bootstrap requirements and configurations
    do_bootstrap()
    logger = get_logger()

    input_path = args.input
    logger.info("start time")
    logger.info(f"Run tool on '{input_path}'")
    if os.path.isfile(input_path):
        iterate(input_path)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                iterate(os.path.join(root, file))

    logger.info("end time")
