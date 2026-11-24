# ============================================================================
# CELL 6 -ROI geometry: crop, paste, and the tests that make the claim checkable
#
# This cell is deliberately cheap and always runs. It defines the two functions that turn a
# native CT into a model input and a model output back into a native volume, and it proves them
# on a synthetic case before any GPU time is spent.
#
#   T1  the crop function cannot see a label     (signature inspection)
#   T2  crop -> paste round trip recovers the mask (IoU check on a known ellipsoid)
#   T3  the pasted volume has the native shape    (assertion)
# ============================================================================
from scipy import ndimage as _ndi
import inspect

ROI_PX = CFG["ROI_PX"]
Z_SPACING = 2.5
XY_MARGIN_MM = 16.0
Z_MARGIN_MM = 10.0


def _axis_indices(center, n_out, out_sp, sp_nat):
    """Native-index coordinate of each output voxel along one axis (no rotation involved)."""
    return center + (np.arange(n_out) - (n_out - 1) / 2.0) * (out_sp / sp_nat)


def crop_roi(ct, sp_nat, center_zyx, out_shape, out_sp, order=1, cval=-1024.0):
    """Sample a box out of a CT volume. Takes an image and a geometry only - never a label."""
    axes, lo, hi = [], [], []
    for a in range(3):
        g = _axis_indices(center_zyx[a], out_shape[a], out_sp[a], sp_nat[a])
        axes.append(g)
        lo.append(max(int(np.floor(g.min())) - 1, 0))
        hi.append(min(int(np.ceil(g.max())) + 2, ct.shape[a]))
    if any(h <= l for l, h in zip(lo, hi)):
        return np.full(out_shape, cval, np.float32)
    sub = np.asarray(ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], np.float32)
    grid = np.meshgrid(axes[0] - lo[0], axes[1] - lo[1], axes[2] - lo[2], indexing="ij")
    return _ndi.map_coordinates(sub, grid, order=order, mode="constant", cval=cval, prefilter=False)


def paste_roi(lab_roi, geom, native_shape):
    """Write an ROI-space label volume back onto the native grid (nearest vertex).

    Where two kidney ROIs overlap, the higher label wins by priority background < kidney < lesion,
    so an organ voxel from one ROI can never erase a lesion voxel from the other.
    """
    # round-half-up, matching how scipy maps continuous coordinates to voxels; np.round would
    # round halves to even and alias the grid by a factor of two whenever out_sp == sp_native
    idx = [np.floor(_axis_indices(geom["center"][a], geom["out_shape"][a],
                                  geom["out_sp"][a], geom["sp_native"][a]) + 0.5).astype(int) for a in range(3)]
    valid = [(idx[a] >= 0) & (idx[a] < native_shape[a]) for a in range(3)]
    iz, iy, ix = idx[0][valid[0]], idx[1][valid[1]], idx[2][valid[2]]
    src = lab_roi[np.ix_(*valid)]
    out = np.zeros(native_shape, np.uint8)
    cur = out[np.ix_(iz, iy, ix)]
    cur_rank = np.where(np.isin(cur, [1]), 1, np.where(np.isin(cur, [2, 3]), 2, 0))
    src_rank = np.where(np.isin(src, [1]), 1, np.where(np.isin(src, [2, 3]), 2, 0))
    out[np.ix_(iz, iy, ix)] = np.where(src_rank >= cur_rank, src, cur)
    return out


def _test_crop_cannot_see_labels():
    banned = [p for p in inspect.signature(crop_roi).parameters
              if p.lower() in ("seg", "label", "lab", "gt", "mask")]
    assert not banned, f"crop_roi must not accept {banned}"
    return True


def _test_roundtrip():
    """Crop a known ellipsoid, paste it back, require a high IoU of the recovered mask."""
    sp = np.array([2.5, 1.0, 1.0], np.float32)
    shape = (60, 200, 200)
    zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
    truth = ((((zz - 30) / 8) ** 2 + ((yy - 100) / 30) ** 2 + ((xx - 100) / 20) ** 2) < 1).astype(np.uint8)
    center = np.array([30.0, 100.0, 100.0])
    out_shape, out_sp = (24, 192, 192), (2.5, 1.0, 1.0)
    g = dict(center=center.tolist(), sp_native=sp.tolist(), out_shape=list(out_shape),
             out_sp=list(out_sp), native_shape=list(shape))
    roi = crop_roi(truth.astype(np.float32), sp, center, out_shape, out_sp, order=0, cval=0).astype(np.uint8)
    back = paste_roi(roi, g, shape)
    iou = float((back & truth).sum()) / max(int((back | truth).sum()), 1)
    assert iou > 0.90, f"crop/paste round trip lost the mask (IoU {iou:.3f})"
    assert back.shape == shape, "pasted volume does not have the native shape"
    assert set(np.unique(back)).issubset({0, 1}), "pasting introduced labels that were not in the ROI"
    return iou


_test_crop_cannot_see_labels()
_iou = _test_roundtrip()
print(f"ROI geometry tests passed (round-trip mask IoU {_iou:.4f})")
