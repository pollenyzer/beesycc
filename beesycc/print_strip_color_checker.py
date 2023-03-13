from typing import List, Union
import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from .custom_color_checker import CustomColorChecker
import tensorflow as tf
from tensorflow import keras
from os.path import join, dirname
from skimage.feature import peak_local_max


def _sort_points_clockwise(points: np.ndarray) -> np.ndarray:
    """Sort 4 points in a circular order.

    Args:
        points (np.ndarray): Points to sort.

    Returns:
        np.ndarray: Sorted points.
    """
    first = points[0]
    second = points[
        np.argsort([euclidean(first, v) for v in points])[1]
    ]
    third = points[
        np.argsort([euclidean(second, v) for v in points])[2]
    ]
    fourth = points[
        np.argsort([euclidean(third, v) for v in points])[1]
    ]
    return np.float32([first, second, third, fourth])


class PrintStripColorChecker(CustomColorChecker):
    """ Stores the printer test strip's reference values.

        These values are taken from the MediaStandard Print.
        TODO: Ursprung so genau genug?
    """

    # these values correspond to the packacking used for Kellogg's (TM)
    # corn flakes products and given in CIE LAB notation
    # (not CV2 notation), that is L* is scaled 0-100 and
    # A* & B* are scaled -128-127.
    _PRINTSPEC_LAB_COLOR_DEFINITION = [
        [
            [24.00, 22.00, -46.00],
            [85.85, -7.49, -14.13],
            [58.32, 54.99, -6.39],
            [75.62, -16.48, -26.70],
            [48.00, 74.00, -3.00],
            [55.00, -37.00, -50.00],
            [71.23, 33.64, -7.30],
            [64.68, -26.83, -39.61],
            [83.80, 15.34, -5.77],
            [39.47, -0.98, -1.13],
            [95.00, 0.00, -2.00],
            [23.00, 0.00, 0.00],
            [76.91, 0.56, -1.96],
            [47.00, 68.00, 48.00],
            [58.04, 0.28, -1.13],
            [93.32, -2.13, 17.61],
            [41.11, 0.00, -0.64],
            [91.55, -4.00, 41.00],
            [16.00, 0.00, 0.00],
            [89.00, -5.00, 93.00],
            [61.82, 0.00, -1.16],
            [90.02, -4.75, 67.99],
            [79.47, 0.00, -1.61],
            [50.00, -65.00, 27.00],
        ]
    ]

    printerstrip_lab_colors = np.array(
        _PRINTSPEC_LAB_COLOR_DEFINITION
    )
    printerstrip_lab_colors += np.array([0, 128, 128])
    printerstrip_lab_colors *= ([255/100, 1, 1])

    def detect_printspec_and_extract_reference_bgr_colors_from_bgr_img(
        standardized_img_bgr: np.ndarray,
        model: keras.models.Model = None
    ):
        """Returns a list of 24 printspec colors that were extracted from the
        input image.

        Args:
            standardized_img_bgr (np.ndarray): Input image that is scaled to
                2208x1568px and includes a printspec of approximately width of
                1261px (starting with the first color patch and ending with
                the last one). Input orientation can be either landscape or
                portrait.
            model (keras.models.Model): Provide a loaded model for warm
                start. Pass None for automatically loading the model.

        Returns:
            (np.ndarray): A np.ndarray of shape (1, 24, 3) containing 24 bgr
                colors in range 0-255 of type np.float.
        """
        assert np.min(
            standardized_img_bgr.shape[0:2]) == 1568, (
                "Input not properly normalized."
            )
        assert np.max(
            standardized_img_bgr.shape[0:2]) == 2208, (
                "Input not properly normalized."
            )
        assert standardized_img_bgr.shape[2] == 3, (
            "Wrong number of channels, expected BGR channels"
        )

        if model is None:
            model_path = join(
                dirname(__file__),
                "models",
                "printspec_detection",
                "run_2022-12-01T16-23-14"
            )
            model = tf.keras.models.load_model(model_path)

        # use model to predict PrintSpec location
        if standardized_img_bgr.shape[0] < standardized_img_bgr.shape[1]:
            # landscape
            # Apply some padding so that network input is
            # divisible by 2 at least 5 times. In theory, the network is fully
            # convolutional, so the input could be of any size, however, a
            # certain divisibility by two is needed for the pooling layers.
            standardized_img_bgr = np.concatenate(
                [standardized_img_bgr, np.zeros((1568, 144, 3), np.uint8)],
                axis=1
            )
            model_input = cv2.resize(standardized_img_bgr, (384, 256))
        else:
            # portrait
            standardized_img_bgr = np.concatenate(
                [standardized_img_bgr, np.zeros((144, 1568, 3), np.uint8)],
                axis=0
            )
            model_input = cv2.resize(standardized_img_bgr, (256, 384))

        pred = model.predict(model_input[None], verbose=False)[0]
        pred = pred * 255
        pred = pred.astype(np.uint8)
        pred = cv2.resize(
            pred,
            (standardized_img_bgr.shape[1], standardized_img_bgr.shape[0])
        )

        # a blob detection is necessary to find the corners of the print strip
        _, pred_thresh = cv2.threshold(pred, 50, 255, 0)
        contours, _ = cv2.findContours(
            pred_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # find the best fitting contour, starting with the largest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        success = False
        for contour in contours:
            center, shape, angle = cv2.minAreaRect(contour)
            width = max(shape)
            height = min(shape)
            if height == 0:
                continue
            # only accept contour if its aspect ratio is within some margin
            # around print strips aspect ratio
            if (width / height) > 15 and (width / height) < 20:
                rect = cv2.minAreaRect(contour)
                kpts = cv2.boxPoints(rect)
                success = True

        if not success:
            print("No printspec found")
            return None

        # reshape the detected printstrip to a fixed size. This is done by
        # warping
        height_dst = 70
        width_dst = 1200
        pts_src = kpts.astype(np.float32)
        pts_src = _sort_points_clockwise(pts_src)
        pts_dst = np.float32([
            [0, 0],
            [0, height_dst],
            [width_dst, height_dst],
            [width_dst, 0]
        ])
        warp_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        img_warp = cv2.warpPerspective(
            standardized_img_bgr, warp_matrix, (width_dst, height_dst)
        )

        # in the following the print strip is processed such that the color
        # patches center points can be estimated.
        img_warp_gray = cv2.cvtColor(img_warp, cv2.COLOR_BGR2GRAY)
        img_warp_blur = cv2.GaussianBlur(img_warp_gray, (3, 3), 0)
        _, img_warp_thresh = cv2.threshold(
            img_warp_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        dist = cv2.distanceTransform(img_warp_thresh, cv2.DIST_L2, 3)

        # find coords of first and last patch
        local_maxima = peak_local_max(
            dist,
            threshold_abs=0.5,
            min_distance=70,
            exclude_border=10,
        )

        # coords
        first_patch = local_maxima[local_maxima[:, 1].argmin()]
        last_patch = local_maxima[local_maxima[:, 1].argmax()]

        # contains the coordinates of all color patches
        patches = np.linspace(
            start=first_patch,
            stop=last_patch,
            num=24,
            endpoint=True
        ).astype(np.int32)

        patch_dim = 24  # pixels
        patch_colors = []
        for patch in patches:
            patch_color = img_warp[
                patch[0]-patch_dim//2: patch[0]+patch_dim // 2,
                patch[1]-patch_dim//2: patch[1]+patch_dim // 2
            ].mean(axis=(0, 1)).astype(np.uint8)
            patch_colors.append(patch_color)

        patch_colors = np.array(patch_colors)[None]

        # the strip could be left-right aligned. In this case a flip is
        # necessary
        if patch_colors[0, 11].mean() > patch_colors[0, 12].mean():
            patch_colors = np.flip(patch_colors, axis=1)

        return patch_colors

    def extract_reference_bgr_colors_from_bgr_img(
        img_bgr: np.ndarray,
        xy_corners: Union[List, np.ndarray],
    ) -> np.ndarray:
        """Extracts the printer test strip colors from an input image for
        cases where the printer strip's position is known.

        Args:
            img_bgr (np.ndarray): A input image that contains a print strip.
            xy_corners (Union[List, np.ndarray]): An (unordered) list
                (or array) of four xy-coordinates that describe the
                color strips corners within the image. The labels are expected
                to be on the outer corners of the left-/rightmost color
                patches (blue & green color patches).
        Returns:
            (np.ndarray): A np.ndarray of shape (1, 24) containing 24 bgr
                colors in range 0-255 of type np.float.
        """

        pts1 = _sort_points_clockwise(xy_corners)

        # scale all cropped cc images to the same width and height
        # height/ width of printer strip roughly equals 0.4 / 9.6
        new_height, new_width = 50, 1200
        pts2 = np.float32([
            [0, 0],
            [0, new_height],
            [new_width, new_height],
            [new_width, 0],
        ])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        normalized_patch_bgr = cv2.warpPerspective(
            img_bgr, matrix, (new_width, new_height)
        )

        extracted_colors_bgr = np.empty_like(
            PrintStripColorChecker.printerstrip_lab_colors
        )
        for x in range(extracted_colors_bgr.shape[1]):
            extracted_colors_bgr[0, x] = normalized_patch_bgr[
                10:40,
                24 + x*50 - 10:24 + x*50 + 10
            ].mean(axis=(0, 1))

        # flip necessary?
        if (
            extracted_colors_bgr[0, 11].mean() >
            extracted_colors_bgr[0, 12].mean()
        ):
            extracted_colors_bgr = np.flip(extracted_colors_bgr, axis=1)

        return extracted_colors_bgr

    def extract_reference_lab_colors_from_bgr_img(
        img_bgr: np.ndarray,
        xy_corners: Union[List, np.ndarray],
    ) -> np.ndarray:
        """Extracts the printer test strip colors from an input image.

        Args:
            img_bgr (np.ndarray): The input image containing the PrintSpec
                color strip.
            xy_corners (Union[List, np.ndarray]): An (unordered) list
                (or array) of four xy-coordinates that describe the
                color strips corners within the image. The labels are expected
                to be on the outer corners of the left-/rightmost color
                patches (blue & green color patches).
        Returns:
            (np.ndarray): A np.ndarray of shape (1, 24) containing 24 lab
                colors in range 0-255 of type np.float.
        """
        extracted_colors_bgr = (
            PrintStripColorChecker
            .extract_reference_bgr_colors_from_bgr_img(
                img_bgr,
                xy_corners
            )
        )
        return cv2.cvtColor(
            extracted_colors_bgr.astype(np.uint8),
            cv2.COLOR_BGR2LAB
        )
