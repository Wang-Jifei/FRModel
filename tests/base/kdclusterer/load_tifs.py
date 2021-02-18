import numpy as np
from PIL import Image
import sys
import gdal

bands_str = ["red", "green", "blue"]
bands = []

for band_str in bands_str:
    band_ds: gdal.Dataset = gdal.Open(sys.path[0] + f"/testimgs/{band_str}.tif")
    band: gdal.Band = band_ds.GetRasterBand(1)
    bands.append(band.ReadAsArray())

bands_ar = np.moveaxis(np.stack(bands), 0, -1)
print(f"bands_ar.shape: {bands_ar.shape}")