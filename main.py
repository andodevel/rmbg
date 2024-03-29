import argparse

from bootstrap import do_bootstrap
from helpers import *
from debug import *
from config import get_logger
from PIL import Image
from libs.crfasrnn.crfrnn_model import get_crfrnn_model_def
import libs.crfasrnn.util as crfasrnn_utils

logger = get_logger()


def generate_renditions(image_org_):
    image_ = resize_image(image_org_, width=512)
    image_gray_ = to_gray_image(image_)
    image_lab_ = rgb_to_lab(image_)

    return image_, image_gray_, image_lab_


def optimize_labels_with_segments(image, labels, segments):
    segmented_labels = np.copy(labels)

    # loop over the unique segment values
    for (i, segVal) in enumerate(np.unique(segments)):
        # construct a mask for the segment
        mask = np.zeros(image.shape[:2], dtype="uint8")
        current_segment = segments == segVal
        mask[current_segment] = 1
        sum_mask = mask.sum()
        # mask[segments == segVal] = 255
        # show the masked region
        # display_two_images(mask, cv2.bitwise_and(image, image, mask=mask))
        mask = mask & np.squeeze(segmented_labels)
        sum_labels = mask.sum()
        if sum_labels / sum_mask < 0.6:
            # means current segment does not belong to foreground.
            segmented_labels[current_segment] = 0

    return segmented_labels


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
    w, h, _ = image.shape

    # ------ start section: semantic segmentation apply ------
    img_data, img_h, img_w, size = crfasrnn_utils.get_preprocessed_image(image)
    labels = model.predict(img_data, verbose=False)[0]
    labels = labels.argmax(axis=2).astype('uint8')[:img_h, :img_w]
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
    image_crf_splashed = apply_image_mask(np.expand_dims(labels, -1), image, [0, 0, 0])
    # Remove small objects
    labels = cv2.erode(labels, None, iterations=3)
    labels = cv2.dilate(labels, None, iterations=4)
    # ------ end section: semantic segmentation apply ------

    # ------ start section: superpixel ------
    from skimage.segmentation import quickshift
    # from skimage.segmentation import slic
    # segments = slic(image, n_segments=200, compactness=10, sigma=1, convert2lab=True)
    segments = quickshift(image, kernel_size=2, max_dist=6, ratio=0.6, convert2lab=True)
    # image_quick = img_as_ubyte(mark_boundaries(image, segments_quick))
    # image_quick = img_as_ubyte(mark_boundaries(image_crf_splashed, segments_quick))
    # image_quick = apply_image_mask(labels, image_quick, [0, 0, 0])
    # ------ end section: superpixel ------

    # smooth mask
    labels = optimize_labels_with_segments(image, labels, segments)
    labels = np.expand_dims(labels, -1)
    image_quick_final = apply_image_mask(labels, image, [0, 0, 0])
    display_two_images(image_crf_splashed, image_quick_final, "AI", "AI + superpixel")


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

    # load model once
    model = get_crfrnn_model_def()
    model.load_weights('crfrnn_keras_model.h5')

    if os.path.isfile(input_path):
        iterate(input_path)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                iterate(os.path.join(root, file))

    logger.info("end time")
