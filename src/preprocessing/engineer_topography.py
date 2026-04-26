import os
import rasterio
import numpy as np
from scipy import ndimage

def calculate_topography():
    print("--- Refinement 3: Topological Feature Engineering ---")
    
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    standard_dir = os.path.join(base_dir, "data", "processed", "standardized")
    
    dem_path = os.path.join(standard_dir, "dem.tif")
    if not os.path.exists(dem_path):
        print("Error: dem.tif not found in standardized folder.")
        return

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        meta = src.meta.copy()
        nodata = src.nodata
        
        # Mask nodata for calculations
        mask = (dem == nodata)
        dem_masked = np.where(mask, np.nan, dem)
        
        # 1. TPI (Topographic Position Index)
        # Difference between pixel and mean of 3x3 neighborhood
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0 # Exclude the center pixel
        mean_neighbors = ndimage.generic_filter(dem_masked, np.nanmean, footprint=kernel)
        tpi = dem_masked - mean_neighbors
        
        # 2. TRI (Terrain Ruggedness Index)
        # Mean of absolute differences between pixel and neighbors
        def tri_calc(values):
            center = values[len(values)//2]
            return np.nanmean(np.abs(values - center))
            
        tri = ndimage.generic_filter(dem_masked, tri_calc, size=3)
        
        # Clean up NaNs back to nodata
        tpi = np.where(np.isnan(tpi), nodata, tpi)
        tri = np.where(np.isnan(tri), nodata, tri)
        
        # Save TPI
        meta.update(dtype=rasterio.float32)
        with rasterio.open(os.path.join(standard_dir, "tpi.tif"), 'w', **meta) as dst:
            dst.write(tpi.astype(np.float32), 1)
        
        # Save TRI
        with rasterio.open(os.path.join(standard_dir, "tri.tif"), 'w', **meta) as dst:
            dst.write(tri.astype(np.float32), 1)

    print("Successfully created tpi.tif and tri.tif.")

if __name__ == "__main__":
    calculate_topography()
