from typing import List, Union
import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from beesycc.custom_color_checker import CustomColorChecker


class Calibr8(CustomColorChecker):
    """ Stores the calibr8 color checker's target values. """

    # these values are found in the color checker's handbook
    # and given in CIE LAB notation (not CV2 notation), that is
    # L* is scaled 0-100 and A* & B* are scaled -128-127.
    _CALIBR8_LAB_COLOR_DEFINITION = """
        G1=95.04 -0.36 -0.57
        G2=9.06 -0.86 -2.71
        G3=51.5 -0.23 -0.14
        G4=95.18 -0.44 -0.67
        G5=9.01 -0.87 -2.62
        G6=51.52 -0.08 -0.07
        G7=95.23 -0.41 -0.73
        G8=9.02 -0.86 -2.59
        G9=51.53 -0.18 -0.19
        G10=95.23 -0.45 -0.62
        G11=50.99 -0.02 0.31
        C12=33.42 52.1 -10.86
        C13=61.56 25.2 -19.95
        C14=28.38 49.4 -39.7
        C15=49.89 -16.82 -49.67
        C16=61.43 -32.51 -26.67
        C17=20.15 -28.84 -5.46
        C18=60.38 -43.48 -11.23
        C19=22.41 0.35 10.6
        G20=51.01 -0.03 0.1
        G21=8.81 -0.86 -2.55
        C22=20.83 17.73 -19.22
        C23=41.08 19.97 -38.83
        C24=20.05 -1.73 -37.18
        C25=60.73 -19.09 -32
        C26=22.52 -21.5 -22.03
        C27=62.37 -7.8 -33.36
        C28=51.1 -53.99 -9.33
        C29=61.41 -42.24 23.32
        G30=8.86 -0.97 -2.7
        G31=95.26 -0.43 -0.45
        C32=85.16 -2.29 -8.45
        C33=85.14 12.33 0.67
        C34=84.57 -17.95 0.1
        C35=85.31 10.52 6.93
        C36=85.73 -12.19 28.09
        C37=61.67 30.84 39.43
        C38=65.43 19 21.01
        C39=51.65 -55.5 16.87
        G40=95.34 -0.38 -0.57
        G41=50.65 0.23 0.34
        C42=31.17 17.72 25.2
        C43=61.32 36.83 72.8
        C44=19.7 22.15 -59.36
        C45=95.22 -0.45 -0.65
        C46=8.95 -0.88 -2.55
        C47=75.75 17.98 24.79
        C48=73.1 25.61 26.73
        C49=21.26 -28.22 5.97
        G50=50.65 0.03 0.35
        G51=8.82 -0.8 -2.56
        C52=64.39 18 20.75
        C53=34.67 11.71 -53.44
        C54=53.45 -47.2 40.96
        C55=80.31 -0.35 -0.48
        C56=31.4 0.58 1.02
        C57=64.13 11.9 27.8
        C58=65.52 12.85 17.93
        C59=60.74 -46.82 11.01
        G60=8.93 -0.92 -2.67
        G61=95.02 -0.43 -0.71
        C62=49.1 -6.99 -22.6
        C63=46.23 53.09 21.87
        C64=41.07 63.57 40.75
        C65=65.31 -0.05 -0.53
        C66=41.23 0.53 1.4
        C67=43.22 14.5 30.36
        C68=64.93 14.72 17.01
        C69=61.73 -29.72 44.23
        G70=95.13 -0.46 -0.62
        G71=50.8 -0.17 0.12
        C72=37.78 -16.7 32.02
        C73=20.73 31.92 -27.47
        C74=81.63 0.49 90.58
        C75=51.43 -0.02 -0.03
        C76=60.52 0.1 -0.61
        C77=68.09 12.12 17.25
        C78=65.68 14.67 19.94
        C79=53.2 -52.06 46.56
        G80=51.2 -0.17 -0.21
        G81=8.78 -0.8 -2.64
        C82=53.37 8.24 -25.95
        C83=72.04 -26.16 66.64
        C84=48.54 56.53 -16.46
        C85=35.6 0.46 1.13
        C86=75.69 -0.09 -0.24
        C87=45.12 24.11 42.54
        C88=35.07 15.52 28.71
        C89=62.43 -55.62 49.05
        G90=8.79 -0.92 -2.65
        G91=95.26 -0.44 -0.69
        C92=69.95 -35.03 1.16
        C93=71.87 16.21 80.57
        C94=48.21 -35.2 -30.65
        C95=15.94 -0.55 -0.61
        C96=91.14 -0.02 -0.56
        C97=64.6 23.49 28.4
        C98=66.28 20.8 29.08
        C99=62.37 14.91 52.85
        G100=95.3 -0.47 -0.67
        G101=51.54 -0.04 0.09
        C102=85.43 6.91 16.83
        C103=87.97 -15.01 5.71
        C104=85.28 4.21 -6.1
        C105=85.24 -14.35 -8.05
        C106=71.28 0.05 -0.32
        C107=46.17 0.02 1.06
        C108=20.94 0.05 0.48
        C109=62.04 -14.23 57.89
        G110=51.07 0.04 0.27
        G111=8.91 -0.84 -2.58
        C112=23.19 34.3 9.07
        C113=44.8 65.7 50.09
        C114=60.95 35.33 4.23
        C115=61.45 35.74 19.07
        C116=58.88 49.5 73.54
        C117=74.02 -10.22 88.57
        C118=64.03 1.2 60.14
        C119=74.33 -30.22 75.73
        G120=8.92 -0.74 -2.73
        G121=51.3 0.12 0.09
        C122=41.67 62.44 10.65
        C123=20.45 31.68 -6.73
        C124=41.9 64.95 36.92
        C125=48.52 66.46 50.9
        C126=76.68 16.79 89.11
        C127=82.62 2.54 95.63
        C128=73.55 -18.66 80.57
        C129=21.07 15.21 16.03
        G130=50.63 0.3 0.32
        G131=95.14 -0.48 -0.59
        G132=51.51 -0.23 -0.19
        G133=9.03 -0.72 -2.68
        G134=95.19 -0.45 -0.82
        G135=51.48 -0.19 -0.19
        G136=8.87 -0.69 -2.63
        G137=95.15 -0.47 -0.74
        G138=51.53 -0.16 -0.28
        G139=8.99 -0.76 -2.62
        G140=95.18 -0.46 -0.67
    """

    C = np.zeros((10, 14, 3), np.float)
    for line_id, line in enumerate(
        _CALIBR8_LAB_COLOR_DEFINITION.strip().split("\n")
    ):
        line = line.strip()
        C[line_id % 10, line_id // 10] = np.array(
            line.split("=")[-1].split(" "), np.float64
        )

    # reference colors in CV2 LAB notation
    # https://docs.opencv.org/2.4.8/modules/imgproc/
    # doc/miscellaneous_transformations.html#cvtcolor
    # L channel was [0.0, 100.0] -> [0, 255]
    # A channel was [-128.0, 127.0] -> [0, 255]
    # B channel was [-128.0, 127.0] -> [0, 255]
    C[:, :, 0] = C[:, :, 0] / 100 * 255
    C[:, :, 1] = (C[:, :, 1] + 128)
    C[:, :, 2] = (C[:, :, 2] + 128)

    calibr8_lab_colors = C

    def extract_reference_bgr_colors_from_bgr_img(
        img_bgr: np.ndarray,
        xy_corners: Union[List, np.ndarray]
    ) -> np.ndarray:
        """Extracts the color checker colors from an input image.

        Args:
            img_bgr (np.ndarray): The source image in bgr pixel format.
            xy_corners (Union[List, np.ndarray]): An (unordered) list
                (or array) of four xy-coordinates that describe the
                color checkers corners within the image.

        Returns:
            np.ndarray: A np.ndarray of shape (10, 14) containing 140 bgr
                colors in range 0-255 of type np.float.
        """
        # brings the xy-coordinates in (any) circular order
        first = xy_corners[0]
        second = xy_corners[
            np.argsort([euclidean(first, v) for v in xy_corners])[1]
        ]
        third = xy_corners[
            np.argsort([euclidean(second, v) for v in xy_corners])[2]
        ]
        fourth = xy_corners[
            np.argsort([euclidean(third, v) for v in xy_corners])[1]
        ]

        pts1 = np.float32([
            first, second, third, fourth
        ])
        # scale all cropped cc images to the same width and height
        new_height, new_width = 1000, 1400
        pts2 = np.float32([
            [0, 0],
            [0, new_height],
            [new_width, new_height],
            [new_width, 0],
        ])

        # Apply perspective transform algorithm to color card
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        normalized_img_bgr = cv2.warpPerspective(
            img_bgr, matrix, (new_width, new_height)
        )
        # NOTE: the up down check now compares the mean accross all channels
        # between the 4th tile from the left in the 2nd and 2nd last row.
        # This was more robust to distinct lighting changes.
        if (
            np.abs(
                normalized_img_bgr[140:160, 340:360][:, :, :].mean()
            ) < np.abs(
                normalized_img_bgr[-160:-140, 340:360][:, :, :].mean()
            )
        ):
            # up-down is reversed and needs flipping
            normalized_img_bgr = np.flipud(normalized_img_bgr)
        if (
            normalized_img_bgr[-160:-140, 40:60][:, :].mean() <
            normalized_img_bgr[-160:-140, -60:-40][:, :].mean()
        ):
            # left-right is reversed and needs flipping
            normalized_img_bgr = np.fliplr(normalized_img_bgr)

        extracted_colors_bgr = np.empty_like(Calibr8.calibr8_lab_colors)
        for y in range(extracted_colors_bgr.shape[0]):
            for x in range(extracted_colors_bgr.shape[1]):
                # colors are averaged within 20x20 pixel patches
                extracted_colors_bgr[y, x] = normalized_img_bgr[
                    50 + y*100 - 10:50 + y*100 + 10,
                    50 + x*100 - 20:50 + x*100 + 20
                ].mean(axis=(0, 1))

        return extracted_colors_bgr

    def extract_reference_lab_colors_from_bgr_img(
        img_bgr: np.ndarray,
        xy_corners: Union[List, np.ndarray]
    ) -> np.ndarray:
        """Extracts the color checker colors from an input image.

        Args:
            img_bgr (np.ndarray): The source image in bgr pixel format.
            xy_corners (Union[List, np.ndarray]): An (unordered) list
                (or array) of four xy-coordinates that describe the
                color checkers corners within the image.

        Returns:
            np.ndarray: A np.ndarray of shape (10, 14) containing 140 lab
                colors in range 0-255 of type np.float.
        """
        extracted_colors_bgr = (
            Calibr8.extract_reference_bgr_colors_from_bgr_img(
                img_bgr,
                xy_corners
            )
        )
        return cv2.cvtColor(
            extracted_colors_bgr.astype(np.uint8),
            cv2.COLOR_BGR2LAB
        )
