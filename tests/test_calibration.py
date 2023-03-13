
from os.path import join, dirname
import cv2
import numpy as np
from beesycc.color_correction import ColorCorrectionCV2
from beesycc.calibr8_color_checker import Calibr8
from beesycc.print_strip_color_checker import PrintStripColorChecker
import matplotlib.pyplot as plt


def test_printspec_calibration():
    """This test loads a sample file from the test resource folder, extracts
    the print spec color strip, automatically detects its position, calculates
    the color correction matrix and applies the color transformation on the
    input image. The results are stored in ./test_output
    """

    img_bgr = cv2.imread(
        join(
            dirname(__file__),
            "test_resources",
            "test_cc_printspec_standardized_1.jpg"
        )
    )

    # img needs to be have shape 1568 × 2208 pixels
    extracted_bgr_colors = (
        PrintStripColorChecker
        .detect_printspec_and_extract_reference_bgr_colors_from_bgr_img(
            standardized_img_bgr=img_bgr
        )
    )

    cv2.imwrite(
        join(
            dirname(__file__), "test_output", "printspec_patch_extracted.png"
        ),
        cv2.resize(
            extracted_bgr_colors,
            interpolation=cv2.INTER_NEAREST,
            dsize=None,
            fx=20,
            fy=20
        )
    )

    cc = ColorCorrectionCV2(
        source_bgr_colors=extracted_bgr_colors.reshape(-1, 3),
        target_lab_colors=PrintStripColorChecker
        .printerstrip_lab_colors.reshape(-1, 3)
    )

    cv2.imwrite(
        join(
            dirname(__file__),
            "test_output",
            "printspec_patch_calibrated.png"
        ),
        cv2.resize(
            cc.calibrate_bgr_image(
                img_bgr=extracted_bgr_colors.astype(np.uint8)
            ),
            interpolation=cv2.INTER_NEAREST,
            dsize=None,
            fx=20,
            fy=20
        )
    )
    cv2.imwrite(
        join(
            dirname(__file__),
            "test_output", "printspec_img_calibrated.jpg"
        ),
        cc.calibrate_bgr_image(img_bgr=img_bgr)
    )


def test_calibr8_calibration():
    """This test loads a sample file from the test resource folder, extracts
    the calibr8 color checker (hard coded position), calculates the color
    correction matrix and applies the color transformation on the input
    image. The results are stored in ./test_output
    """
    img_bgr = cv2.imread(
        join(dirname(__file__), "test_resources", "test_cc_calibr8.jpg")
    )

    # hard coded corners in arbitrary order
    xy_corners = np.array([
        [1800, 2000],
        [1801, 2538],
        [2541, 1997],
        [2547, 2533],
    ])

    extracted_bgr_colors = Calibr8.extract_reference_bgr_colors_from_bgr_img(
        img_bgr=img_bgr,
        xy_corners=xy_corners
    )

    # save extracted color patches for visual inspection
    cv2.imwrite(
        join(
            dirname(__file__),
            "test_output",
            "calibr8_patch_extracted.png"
        ),
        cv2.resize(
            extracted_bgr_colors,
            interpolation=cv2.INTER_NEAREST,
            dsize=None,
            fx=20,
            fy=20
        )
    )

    cv2.imwrite(
        join(
            dirname(__file__),
            "test_output",
            "calibr8_patch_target_colors.png"
        ),
        cv2.resize(
            cv2.cvtColor(
                Calibr8.calibr8_lab_colors.astype(np.uint8),
                cv2.COLOR_Lab2BGR
            ),
            interpolation=cv2.INTER_NEAREST,
            dsize=None,
            fx=20,
            fy=20
        )
    )

    cc = ColorCorrectionCV2(
        source_bgr_colors=extracted_bgr_colors.reshape(-1, 3),
        target_lab_colors=Calibr8.calibr8_lab_colors.reshape(-1, 3)
    )

    cv2.imwrite(
        join(
            dirname(__file__),
            "test_output",
            "calibr8_patch_calibrated.png"
        ),
        cv2.resize(
            cc.calibrate_bgr_image(
                img_bgr=extracted_bgr_colors.astype(np.uint8)
            ),
            interpolation=cv2.INTER_NEAREST,
            dsize=None,
            fx=20,
            fy=20
        )
    )

    cv2.imwrite(
        join(dirname(__file__), "test_output", "calibr8_img_calibrated.jpg"),
        cc.calibrate_bgr_image(img_bgr=img_bgr)
    )
