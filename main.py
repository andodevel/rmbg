import argparse

from bootstrap import do_bootstrap
from helpers import *
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
    w, h, _ = image.shape

    # ------ start section: crfasrnn ------
    from libs.crfasrnn.crfrnn_model import get_crfrnn_model_def
    import libs.crfasrnn.util as crfasrnn_utils
    model = get_crfrnn_model_def()
    model.load_weights('crfrnn_keras_model.h5')
    img_data, img_h, img_w, size = crfasrnn_utils.get_preprocessed_image(image)
    probs = model.predict(img_data, verbose=False)[0]
    probs = probs.argmax(axis=2).astype('uint8')[:img_h, :img_w]
    probs = np.array(Image.fromarray(probs.astype('uint8')).resize((h, w)))
    # Remove noise and expand a bit to the 'background'
    probs = cv2.erode(probs, None, iterations=3)
    probs = cv2.dilate(probs, None, iterations=3)
    probs = np.expand_dims(probs, -1)
    # image_crfasrnn = crfasrnn_utils.get_label_image(probs, img_h, img_w, size);
    image_crf_splashed = apply_image_mask(probs, image, [0, 0, 0])
    # display_two_images(image_splashed, image_crf_splashed, "deeplabv3", "crfasrnn")
    # ------ end section: crfasrnn ------

    # ------ start section: superpixel ------
    from skimage.segmentation import felzenszwalb
    from skimage.segmentation import mark_boundaries
    from skimage.util import img_as_ubyte

    img = image
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=15)

    # loop over the unique segment values
    for (i, segVal) in enumerate(np.unique(segments_fz)):
        # construct a mask for the segment
        mask = np.zeros(image.shape[:2], dtype="uint8")
        current_segment = segments_fz == segVal
        mask[current_segment] = 1
        sum_mask = mask.sum()
        # mask[segments_fz == segVal] = 255
        # show the masked region
        # display_two_images(mask, cv2.bitwise_and(image, image, mask=mask))
        mask = mask & np.squeeze(probs)
        sum_probs = mask.sum()
        # if sum_mask > sum_probs:
        if sum_probs / sum_mask < 0.6:
            # means current segment does not belong to foreground.
            probs[current_segment] = 0

    # print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
    # print(f"segments_fz's shape {segments_fz.shape}")
    # image_fz = img_as_ubyte(mark_boundaries(image_crf_splashed, segments_fz))
    # ------ end section: superpixel ------

    # image_fz = apply_image_mask(probs, image_fz, [0, 0, 0])
    image_final = apply_image_mask(probs, image, [0, 0, 0])
    display_two_images(image_crf_splashed, image_final)


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
