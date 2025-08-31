# app.py
import streamlit as st
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import traceback

# Try scikit-image resize; show clear message if missing
try:
    from skimage.transform import resize
except Exception:
    resize = None

st.set_page_config(page_title="NDVI Change Detection", layout="wide")

def read_bytes(uploaded_file):
    """Read uploaded file into bytes once."""
    if uploaded_file is None:
        return None
    uploaded_file.seek(0)
    return uploaded_file.read()

def open_memfile_from_bytes(b):
    """Return (memfile, dataset) from bytes."""
    mem = MemoryFile(b)
    ds = mem.open()
    return mem, ds

def compute_ndvi_from_ds(ds, red_idx, nir_idx):
    """Compute NDVI given a rasterio dataset and 1-based band indices."""
    # read and convert to float32
    red = ds.read(int(red_idx)).astype('float32')
    nir = ds.read(int(nir_idx)).astype('float32')

    # mask nodata
    mask = None
    if getattr(ds, "nodata", None) is not None:
        nod = ds.nodata
        mask = (red == nod) | (nir == nod)

    # avoid division by zero
    ndvi = (nir - red) / (nir + red + 1e-10)

    if mask is not None:
        ndvi = np.where(mask, np.nan, ndvi)
    return ndvi

def reproject_to_target(src_array, src_transform, src_crs,
                        dst_shape, dst_transform, dst_crs,
                        resampling=Resampling.bilinear):
    """Wrap rasterio.reproject to create destination array and return it."""
    dst = np.empty(dst_shape, dtype=src_array.dtype)
    reproject(
        source=src_array,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling
    )
    return dst

def safe_resize(arr, target_shape):
    """Resize with skimage if available, else simple numpy fallback (nearest)."""
    if resize is not None:
        out = resize(arr, target_shape, order=1, mode="reflect", preserve_range=True, anti_aliasing=False)
        return out.astype(arr.dtype)
    else:
        # fallback: simple nearest-neighbor via indexing (may distort). Notify user:
        st.warning("scikit-image not installed — using crude nearest-neighbor resize. Install scikit-image for better resampling: pip install scikit-image")
        y_scale = arr.shape[0] / target_shape[0]
        x_scale = arr.shape[1] / target_shape[1]
        yy = (np.arange(target_shape[0]) * y_scale).astype(int)
        xx = (np.arange(target_shape[1]) * x_scale).astype(int)
        return arr[np.clip(yy, 0, arr.shape[0]-1)[:, None], np.clip(xx, 0, arr.shape[1]-1)][...,].astype(arr.dtype)

def plot_array(arr, title="NDVI", vmin=-1, vmax=1, cmap='RdYlGn'):
    fig, ax = plt.subplots(figsize=(6,5))
    # handle all-nan arrays gracefully
    if np.all(np.isnan(arr)):
        ax.text(0.5, 0.5, "All values are NaN", ha='center', va='center')
        im = None
    else:
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    if im is not None:
        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label('NDVI' if "ndvi" in title.lower() else "Value")
    st.pyplot(fig)
    plt.close(fig)

# ---------- Sidebar UI ----------
st.sidebar.title("Controls & Uploads")
st.sidebar.markdown("Upload two GeoTIFF files and pick Red/NIR band indices (1-based). Click Run when ready.")

uploaded1 = st.sidebar.file_uploader("Earlier image (GeoTIFF)", type=["tif","tiff"])
uploaded2 = st.sidebar.file_uploader("Later image (GeoTIFF)", type=["tif","tiff"])

# read bytes once (prevents stream exhaustion)
b1 = read_bytes(uploaded1) if uploaded1 else None
b2 = read_bytes(uploaded2) if uploaded2 else None

# derive sensible defaults for band selectors by peeking at headers (if available)
max_b1 = 12
max_b2 = 12
if b1:
    try:
        mem_t, ds_t = open_memfile_from_bytes(b1)
        max_b1 = ds_t.count
        ds_t.close(); mem_t.close()
    except Exception:
        max_b1 = 12
if b2:
    try:
        mem_t, ds_t = open_memfile_from_bytes(b2)
        max_b2 = ds_t.count
        ds_t.close(); mem_t.close()
    except Exception:
        max_b2 = 12

st.sidebar.markdown("### Band selection (1-based indices)")
col1, col2 = st.sidebar.columns(2)
with col1:
    red1 = st.number_input("Earlier: Red", min_value=1, max_value=max_b1, value=min(3, max_b1), step=1)
    nir1 = st.number_input("Earlier: NIR", min_value=1, max_value=max_b1, value=min(4, max_b1), step=1)
with col2:
    red2 = st.number_input("Later: Red", min_value=1, max_value=max_b2, value=min(3, max_b2), step=1)
    nir2 = st.number_input("Later: NIR", min_value=1, max_value=max_b2, value=min(4, max_b2), step=1)

auto_align = st.sidebar.checkbox("Auto-align (reproject later → earlier)", value=True)
run = st.sidebar.button("▶ Run Change Detection")

# ---------- Main canvas ----------
st.title("🛰 NDVI Change Detection")
st.markdown("Main view: NDVI maps and difference. Use the sidebar to upload files and set parameters.")

if run:
    if not (b1 and b2):
        st.sidebar.error("Please upload both earlier and later GeoTIFF files before running.")
    else:
        # wrap entire processing with try/except to show tracebacks in the app
        try:
            mem1, src1 = open_memfile_from_bytes(b1)
            mem2, src2 = open_memfile_from_bytes(b2)
            st.sidebar.markdown("#### File info (from run)")
            st.sidebar.write("Earlier:", f"bands={src1.count}, crs={src1.crs}, size={src1.width}×{src1.height}")
            st.sidebar.write("Later:  ", f"bands={src2.count}, crs={src2.crs}, size={src2.width}×{src2.height}")

            ndvi1 = compute_ndvi_from_ds(src1, red1, nir1)
            ndvi2 = compute_ndvi_from_ds(src2, red2, nir2)

            # alignment
            need_align = (src1.crs != src2.crs) or (src1.transform != src2.transform) or (ndvi1.shape != ndvi2.shape)
            if need_align and auto_align:
                if src1.crs is None or src2.crs is None:
                    st.warning("One or both images have no CRS. Using array resize to match shapes.")
                    ndvi2_aligned = safe_resize(ndvi2, ndvi1.shape)
                else:
                    st.info("Reprojecting later image onto earlier image grid (this may take a little time).")
                    ndvi2_aligned = reproject_to_target(
                        ndvi2, src2.transform, src2.crs,
                        ndvi1.shape, src1.transform, src1.crs,
                        resampling=Resampling.bilinear
                    )
            else:
                ndvi2_aligned = ndvi2

            # plot NDVI
            st.subheader("NDVI maps")
            cA, cB = st.columns(2)
            with cA:
                plot_array(ndvi1, "NDVI - Earlier", vmin=-1, vmax=1)
            with cB:
                title_l = "NDVI - Later (aligned)" if need_align else "NDVI - Later"
                plot_array(ndvi2_aligned, title_l, vmin=-1, vmax=1)

            # compute and plot diff
            st.subheader("NDVI Difference (Later - Earlier)")
            change = ndvi2_aligned - ndvi1
            finite = change[np.isfinite(change)]
            if finite.size > 0:
                m = np.max(np.abs([finite.min(), finite.max()]))
                vmin, vmax = -m, m
            else:
                vmin, vmax = -1, 1
            plot_array(change, "NDVI Difference (Later - Earlier)", vmin=vmin, vmax=vmax, cmap='RdYlGn')
            st.markdown("🔴 Negative = vegetation loss, 🟢 Positive = vegetation gain")

        except Exception as e:
            tb = traceback.format_exc()
            st.error("An error occurred during processing. See details below.")
            st.code(tb)
        finally:
            # attempt to safely close
            try:
                src1.close(); mem1.close()
            except Exception:
                pass
            try:
                src2.close(); mem2.close()
            except Exception:
                pass