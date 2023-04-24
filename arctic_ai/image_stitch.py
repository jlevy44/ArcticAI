"""Image Writing 
==========
Contains functions for stitching images together."""
import pandas as pd, numpy as np
from scipy.ndimage import label as scilabel
from skimage.measure import regionprops_table
import cv2, os, subprocess
from deepzoom import *
from deepzoom import _get_or_create_path,_get_files_path
from PIL import Image
import tqdm
import dask
from dask.diagnostics import ProgressBar
from scipy.special import softmax
import torch
from sauth import SimpleHTTPAuthHandler, serve_http
from skimage.draw import circle
from pathpretrain.utils import load_image
Image.MAX_IMAGE_PIXELS = None

class Numpy2DZI(ImageCreator):
    def __init__(
        self,
        tile_size=254,
        tile_overlap=1,
        tile_format="jpg",
        image_quality=0.8,
        resize_filter=None,
        copy_metadata=False,
        compression=1.
    ):
        super().__init__(tile_size,tile_overlap,tile_format,image_quality,resize_filter,copy_metadata)
        self.compression=compression

    def create(self, source_arr, destination):
        """
        Create a deep zoom image from the given source array.

        Parameters
        ----------
        source_arr : ndarray
            The source image as a NumPy array.
        destination : str
            The destination folder to save the tiles to.

        Returns
        -------
        str
            The destination folder where the tiles were saved.
        """
        # potentially have an option where dynamically softlink once deeper layer is made so slide is readily available, push to background process and write metadata for dash app to read
        # speed up image saving with dask https://stackoverflow.com/questions/54615625/how-to-save-dask-array-as-png-files-slice-by-slice https://github.com/dask/dask-image/issues/110
        self.image = PIL.Image.fromarray(source_arr if self.compression==1 else cv2.resize(source_arr,None,fx=1/self.compression,fy=1/self.compression,interpolation=cv2.INTER_CUBIC))
        width, height = self.image.size
        self.descriptor = DeepZoomImageDescriptor(
            width=width,
            height=height,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            tile_format=self.tile_format,
        )
        image_files = _get_or_create_path(_get_files_path(destination))
        for level in tqdm.trange(self.descriptor.num_levels, desc='level'):
            level_dir = _get_or_create_path(os.path.join(image_files, str(level)))
            level_image = self.get_image(level)
            for (column, row) in tqdm.tqdm(self.tiles(level), desc='tiles'):
                bounds = self.descriptor.get_tile_bounds(level, column, row)
                tile = level_image.crop(bounds)
                format = self.descriptor.tile_format
                tile_path = os.path.join(level_dir, "%s_%s.%s" % (column, row, format))
                tile_file = open(tile_path, "wb")
                if self.descriptor.tile_format == "jpg":
                    jpeg_quality = int(self.image_quality * 100)
                    tile.save(tile_file, "JPEG", quality=jpeg_quality)
                else:
                    tile.save(tile_file)
        self.descriptor.save(destination)
        return destination

def stitch_slide(arr,
                 dzi_out,
                 compression=4,
                 mask=None):
    if mask is not None: arr[mask]=255
    Numpy2DZI(compression=compression).create(arr,
                                              dzi_out)

def npy2dzi(npy_file='',
            dzi_out='',
            compression=1.):
    stitch_slide(load_image(npy_file),compression=compression,dzi_out=dzi_out)


def stitch_slides(basename="",
                 compression=4,
                 dirname=".",
                 ext=".tif",
                 pull_mask=False,
                 mask_compresssion=8):
    image=os.path.join(dirname,"inputs",f"{basename}{ext}")
    basename=os.path.basename(image).replace(ext,'')
    image=load_image(image)#np.load(image)
    xy_bounds=pd.read_pickle(os.path.join(dirname,"masks",f"{basename}.pkl"))
    write_files=[]
    for ID in xy_bounds:
        (xmin,ymin),(xmax,ymax)=xy_bounds[ID]
        dzi_out=os.path.join(dirname,"dzi_files",f"{basename}_{ID}_img.dzi")
        arr=image[xmin:xmax,ymin:ymax]
        mask=None
        if pull_mask:
            mask=dask.delayed(lambda x: cv2.resize(np.load(x).astype(int),arr.shape[:2],interpolation=cv2.INTER_NEAREST).astype(bool))(os.path.join(dirname,"masks",f"{basename}_{ID}.npy"))
        write_files.append(dask.delayed(stitch_slide)(arr,dzi_out,compression,mask))
    with ProgressBar():
        dask.compute(write_files,scheduler='threading')
