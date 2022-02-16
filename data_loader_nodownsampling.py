import os
from abc import ABC
from typing import List, Tuple, Union, Literal

import SimpleITK as sitk
import cv2
import numpy as np
import torch
from skimage.filters import threshold_multiotsu


class BaseDataProcessor(ABC):

    def __init__(self):
        """initializes a chain of preprocessor steps which is then executed for each elem"""
        self._chain = []

    def _chain_exe(self, image: np.ndarray) -> np.ndarray:
        """applies the preprocessor steps to an image, based on the order in the chain
            arg: image: nd array
            return: image as nd array"""
        for processor_step, kwargs in self._chain:
            if kwargs:
                image = processor_step(image, **kwargs)
            else:
                image = processor_step(image)
        return image.astype(np.float32)

    def add_processor_step(self, func_name: str, kwargs: dict):
        """appends preprocessor step from name and kwargs to the chain
        args:
            func_name: name of the preprocessor function
            kwargs: keyword arguments applied to the func_name together with nd image/array"""
        func = getattr(self, func_name)
        self._chain.append((func, kwargs))

    @classmethod
    def to_chain(cls, chain_elems: dict):
        """adds preprocesser functions to the chain with corresponding elements.
        chain_elems contains the functain name as a key and kwargs as values
        args:
            chain_elems: is a dictionary with the preprocessor function name as key and the coressponding kwargs as
            items. Items are a dict itself."""
        inst = cls()
        for chain_elem, elem_kwargs in chain_elems.items():
            inst.add_processor_step(chain_elem, elem_kwargs)
        return inst

    def __call__(self, image: np.ndarray, **kwargs):
        """executes chain"""
        return self._chain_exe(image)


class BasePreProcessor(BaseDataProcessor):
    """Implementation of Abstract Class BaseDataProcessor. Basic steps are implemented for preprocessing."""

    def __init__(self):
        super().__init__()

    """all functions take an nd.array as input and return a n.array"""

    @staticmethod
    def temp_normalize_image(image: np.ndarray) -> np.ndarray:
        """normalizes images by substracting the mean along temporal axis"""
        return image - np.mean(image, axis=0)

    @staticmethod
    def normalize_image_range(image: np.ndarray) -> np.ndarray:
        """scales image values to an intervall of -1 to 1"""
        return 2 * (image - image.min()) / (image.max() - image.min()) - 1


class PreProcessor(BasePreProcessor):
    """implements more specicfic PreProcessor steps, that take additional arguments"""

    @staticmethod
    def clip_image(image: np.ndarray, boundaries: Tuple[float, float] = None,
                   mode: Literal['multiotsu', 'median'] = None) -> np.ndarray:
        """clips image values to an intervall based on image intensieties.
        Availabe are multiotsu thresholding, and above median
        args:
            boundaries: image is clipped by those fixed values, must be in the format(min,max). If mode is given,
                        boundaries are ignored.
            mode: must be either mutlitotsu or median. If multiotsu boundraries are based on two thresholds. If median
                  the upper boundary is the median value and lower the min value of the image """

        if mode == 'multiotsu':
            boundaries = threshold_multiotsu(image, classes=3)
        elif mode == 'median':
            boundaries = [image.min(), None]
            boundaries[1] = np.median(image)
        elif not boundaries and not mode:
            boundaries = [image.min(), image.max()]
        return np.clip(image, boundaries[0], boundaries[-1])

    def resample_image(image: np.ndarray, frame_shape=(380, 380)) -> np.ndarray:
        """resamples image to specified pixel size using nearest neighbours
        args:
            frame_shape: crops the last two dimensions to the given value"""
        num_frames = len(image)
        resized_image = np.zeros((num_frames,) + frame_shape)
        for idx, frame in enumerate(image):
            resized_image[idx] = cv2.resize(frame, frame_shape, interpolation=cv2.INTER_NEAREST)
        return resized_image

    @staticmethod
    def crop_image(image: np.ndarray, tol=0.01) -> np.ndarray:
        """removes bars (with noise) on image edges. tol is the amount of noise allowed.
            args:
                tol: tolerance value above which axis are accepted to contain image information. If a whole row
                     contains no information and is at the image edge it is assumed to contain no value for
                     classifiation"""
        mask = np.std(image, axis=0) > tol
        idx = np.ix_(mask.any(1), mask.any(0))
        return image[:, idx[0], idx[1]]


class DataLoader:
    def __init__(self, preprocessor_steps: dict):
        """class to load dicom images from path, gets preprocessor arguments to build the preprocessor
        args:
            preprocessor_steps: dict containing names and kwargs of the preprocessing steps"""
        self.preprocessor = PreProcessor.to_chain(preprocessor_steps)

    def __call__(self, series_paths: List[Union[str, os.PathLike]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """loads images to the RAM and applies the preprocessor steps
        args:
            series_paths: List of paths. First entry must point to AP and second to the lateral view. Views must be the
                          same the length.
            return: A tuple containing, two tensors 1.) both views shape: 1 (dummy batch) x time x 2 (views) x height x
                    width 2.) length of the series """

        # loads both dicom views
        series = [self._load_image(path) for path in series_paths]
        series_lengths = [len(view) for view in series]
        assert len(series_lengths) == 2, f'expected 2 views, got {len(series_lengths)} views'
        assert series_lengths[0] == series_lengths[1], f'Length of DSA views are not equal'

        # # converts views to array and make a 3Ch image. The 3Ch are filled with consecutive frames
        # min_size = round(min(series[0].shape[-1], series[1].shape[-1]),-1)
        # if series[1].shape[-1] != min_size or series[1].shape[-2] != min_size:
        #     series[1] = PreProcessor.resample_image(np.float32(series[1]), frame_shape=(min_size, min_size))
        # if series[0].shape[-1] != min_size or series[0].shape[-2] != min_size:
        #     series[0] = PreProcessor.resample_image(np.float32(series[0]), frame_shape=(min_size, min_size))

        series = np.array(series)
        series = self._make_3_ch_img(series)
        series = torch.tensor(series, dtype=torch.float32).unsqueeze(dim=0)
        return series, torch.tensor(series_lengths[0]).unsqueeze(dim=0)

    def _load_image(self, image_path: Union[str, os.PathLike]) -> np.ndarray:
        """loads dicom images to RAM and preprocesses image from path
            args:
                image_paths: list containing path to two views order: [ap, lat]
            return: preprocessed dicom image as np.array"""
        image = self._load_img_to_ram(image_path)
        image = self.preprocessor(image)
        return image.astype(np.half)

    @staticmethod
    def _make_3_ch_img(img: np.ndarray) -> np.ndarray:
        """creates a 3Ch image from a 1Ch image. The 3Ch consists of 3 consecutive frames, centered around the original
        Frame
        arg:
            img: 1CH Image
        return: same image with 3CH. Each slice from the orginal image is converted to the center channel sourounded by
                the previous and nect slice."""
        new_img = np.zeros((*img.shape[0:2], 3, *img.shape[-2:])).swapaxes(1, 0)
        for i_frame in range(len(new_img)):
            img_temp = np.zeros(new_img.shape[1:])
            for i_channel in range(img_temp.shape[1]):
                if i_frame == 0 and i_channel == 0:
                    pass
                elif i_frame == len(new_img) - 1 and i_channel == 2:
                    pass
                else:
                    img_temp[:, i_channel] = img[:, i_frame - 1 + i_channel]
            new_img[i_frame] = img_temp
        return new_img

    @staticmethod
    def _load_img_to_ram(file_path: Union[str, os.PathLike]) -> np.ndarray:
        """loads dicom image to RAM
            args:
                image_path: path pointing to an dicom image
            return: dicom image as np.array"""
        assert file_path.endswith('.dcm'), 'only Dicom file format is supported'
        image = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        return image
