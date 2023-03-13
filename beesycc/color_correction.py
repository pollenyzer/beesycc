from typing import List, Union
import numpy as np
import cv2


class ColorCorrectionCV2:
    def __init__(
        self,
        source_bgr_colors: Union[List, np.ndarray],
        target_lab_colors: Union[List, np.ndarray],
    ) -> None:
        """Constructs a ColorCorrection class by source colors (found on
        images) and target colors (found in handbooks)

        NOTE: cv2.ccm.ColorCorrectionModel takes (target) LAB values in
            rang [0-100, -128-127, -128, 127]
        NOTE: cv2.ccm.ColorCorrectionModel takes (target) sRGB values in
            range [0-1, 0-1, 0-1]
        NOTE: cv2.ccm.ColorCorrectionModel takes (extracted) RGB values  in
            range [0-1, 0-1, 0-1]
        NOTE: More information can be found here
            https://docs.opencv.org/4.6.0/de/df4/
            group__color__correction.html#gafe5da1d946c69c97d42acddd476cc89b
        NOTE: More information can be found here
            https://github.com/riskiest/color_calibration/blob/master/doc/md/
            English/Algorithm/Algorithm.md
        NOTE: More inforamtion can be found here
            https://www.imatest.com/docs/colormatrix/

        Args:
            source_bgr_colors (Union[List, np.ndarray]): The real-world
                colors that should be enhanced (ranging from 0 to 255) with
                type being either `np.uint8` or `np.float`. OpenCV uses
                0-255 values for all BGR channels. The shape should be (-1, 3)
            target_lab_colors (Union[List, np.ndarray]): The lab reference
                colors provided in the same format as the
                `source_bgr_colors`, that is ranging from 0 to 255
                (=CVAT LAB). LAB colors are expected to come from a D50 2 deg
                observer. The shape should be (-1, 3).
        """
        source_bgr_colors = np.array(source_bgr_colors, np.float)
        target_lab_colors = np.array(target_lab_colors, np.float)

        assert len(source_bgr_colors.shape) == 2, \
            "Source array should be list-like"
        assert len(target_lab_colors.shape) == 2, \
            "Target array should be list-like"
        assert source_bgr_colors.shape[-1] == 3, \
            "Wrong number of channels."
        assert target_lab_colors.shape[-1] == 3, \
            "Wrong number of channels."

        # the best results were achieved by the following calibration settings
        # 1. 4x3, linearization: polyfit, CIE2000

        # model requires RGB colors as source scaled [0, 1] and empty first
        # dimension. Shape must be (1, -1, 3)
        # model requires as target either sRGB colors scaled [0, 1] or LAB
        # colors scaled [0-100, -128-127, -128-127]. Shape must be (1, -1, 3)
        c_model = cv2.ccm.ColorCorrectionModel(
            (source_bgr_colors[..., ::-1] / 255.0)[None],
            (
                (target_lab_colors - np.array([0, 128, 128])) *
                np.array([1/255*100, 1, 1])
            )[None],
            cv2.ccm.COLOR_SPACE_Lab_D50_2
        )

        c_model.setCCM_TYPE(cv2.ccm.CCM_4x3)

        # linearization by fitting a third order polygon is legit for
        # unknown image sources and leads to intense colors & much improved
        # black / white contrast. Polyfit works for both, 3x3 and 4x3 while
        # gamma linearization works only well with 4x3 cc matrices.
        # c_model.setLinear(cv2.ccm.LINEARIZATION_COLORPOLYFIT)
        # c_model.setLinearDegree(3)
        c_model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
        c_model.setLinearGamma(2.2)        

        c_model.setMaxCount(5000)  # increased values dont have any effect

        # this removes black, white and saturated colors from the calculations.
        # The model becomes more robust. As a result, black and white patches
        # show less contrast, but colors are more precise now.
        c_model.setSaturatedThreshold(0.05, 0.95)

        # This should give more emphasis on L* channel (L* loss is squared).
        # However, this seems only to work if white and black patches are
        # included (which are filtered out at the moment)
        # c_model.setWeightCoeff(2)

        # this is most appropriate
        c_model.setDistance(cv2.ccm.DISTANCE_CIE2000)
        # c_model.setDistance(cv2.ccm.DISTANCE_RGBL) # also good results

        c_model.run()
        mask = c_model.getMask()

        print(
            "color calibration:",
            f"{mask.shape[0]*mask.shape[1] - mask.sum()} "
            "cc values skipped due to saturation issues "
            "loss =", c_model.getLoss()
        )
        self.c_model = c_model

    def calibrate_bgr_image(self, img_bgr) -> np.ndarray:
        """ Applies the color correction to a BGR image.

        Args:
            img (np.ndarray): The BGR uint8 image to calibrate.

        Returns:
            np.ndarray: The calibrated BGR image.
        """
        assert img_bgr.dtype == np.uint8, "Type `np.uint8` expected."
        img_bgr = img_bgr.copy()
        img_bgr = img_bgr / 255.0  # [0, 255] -> [0, 1] (float)
        img_rgb = img_bgr[:, :, ::-1]  # BGR->RGB
        img_rgb = self.c_model.infer(img_rgb)

        # apply clipping
        img_rgb = np.minimum(np.maximum(np.round(img_rgb*255), 0), 255)
        img_rgb = img_rgb.astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr

    def calibrate_bgr_colors(self, colors):
        """ Applies the color correction to a list of BGR colors.

        Args:
            colors (np.ndarray): The BGR uint8 colors to calibrate.

        Returns:
            np.ndarray: The calibrated BGR colors.
        """
        colors = colors[None, ...]  # additional image dimension
        return self.calibrate_bgr_image(colors)[0]
