"""
Sample code to convert volume to PlenOctree
"""
from typing import Dict
import numpy as np

def vol2plenoctree(
            density : np.ndarray,
            colors : np.ndarray,
            radius: float = 1.0,
            density_threshold: float = 1.0,
            data_format : str = "RGBA") -> Dict[str, np.ndarray]:
    """
    Convert arbirary volume to PlenOctree

    :param density: (Dx, Dy, Dz); dimensions need not be powers
                                  of 2 nor equal
    :param colors: (Dx, Dy, Dz, (3, channel_size));
                                  color data,
                                  last dim is size 3 * channel_size
    :param radius: 1/2 side length of volume
    :param density_threshold: threshold below which
                              density is ignored
    :param data_format: standard PlenOctree data format string,
                        one of :code:`RGBA | SH1 | SH4 | SH9 | SH16`.
                        The channel_size should be respectively
                        :code:`1 | 1 | 4 | 9 | 16```.

    :return: Dict, generated PlenOctree data with all keys required
                   by the renderer. You can save this with
                   :code:`np.savez_compressed("name.npz", **returned_data)`
                   or
                   :code:`np.savez("name.npz", **returned_data)`
                   and open it with volrend.
    """
    # Check dimensions
    assert density.ndim == 3
    assert colors.ndim == 4 and tuple(colors.shape[:3]) == tuple(density.shape), \
            f"{density.shape} != {colors.shape[:3]}"

    dims = list(density.shape)
    maxdim = max(dims)
    assert maxdim <= 1024, "Voxel grid too large"
    n_levels = (maxdim - 1).bit_length()

    # Check data formats
    valid_data_formats = {
        "RGBA" : 4,
        "SH1" : 4,
        "SH4" : 13,
        "SH9" : 28,
        "SH16" : 49,
    }
    assert data_format in valid_data_formats, f"Invalid ddata format {data_format}"
    data_dim = valid_data_formats[data_format]

    # Check if given shape matches promised format
    assert colors.shape[-1] + 1 == data_dim

    result = {}
    result['data_dim'] = np.int64(data_dim)
    result['data_format'] = data_format
    result['invradius3'] = np.array([
             0.5 / radius,
             0.5 / radius,
             0.5 / radius], dtype=np.float32)

    require_pad = dims != [2 ** n_levels] * 3
    center = np.array([radius * (1.0 - dim / 2 ** n_levels) for dim in dims],
                      dtype=np.float32)

    result['offset'] = (0.5 * (1.0 - center / radius)).astype(
            np.float32)

    # Construct mask hierarchy
    hierarchy = []
    pow2 = 2 ** n_levels
    mask = np.zeros((pow2, pow2, pow2), dtype=bool)
    density_mask = density > density_threshold
    mask[:dims[0], :dims[1], :dims[2]] = density_mask

    hierarchy.append(mask)
    while pow2 > 1:
        mask = mask.reshape((pow2 // 2, 2, pow2 // 2, 2, pow2 // 2, 2))
        mask = mask.any(axis=(1, 3, 5))
        pow2 //= 2
        hierarchy.append(mask)

    hierarchy = hierarchy[::-1]

    # PlenOctree standard format data arrays
    all_child = []
    all_data = []
    for i, (mask, next_mask) in enumerate(zip(hierarchy[:-1], hierarchy[1:])):
        nnodes = mask.sum()
        pow2 = mask.shape[0]

        if i == len(hierarchy) - 2:
            # Construct the last tree level
            child = np.zeros((nnodes, 2, 2, 2), dtype=np.uint32);
            if require_pad:
                # Data is not power of 2, pad it (conceptually)
                voxel_indices = np.zeros(next_mask.shape, dtype=np.uint32)
                voxel_indices[:dims[0], :dims[1], :dims[2]] = np.arange(
                        dims[0] * dims[1] * dims[2], dtype=np.uint32).reshape(dims)
                voxel_indices = voxel_indices.reshape(
                        (pow2, 2, pow2, 2, pow2, 2)).transpose(
                                0, 2, 4, 1, 3, 5)[mask].reshape(-1, 2, 2, 2)
                density_i = density.reshape(-1, 1)[voxel_indices]
                colors_i = colors.reshape(-1, data_dim - 1)[voxel_indices]
                bad_mask = ~(next_mask.reshape(pow2, 2, pow2, 2, pow2, 2).transpose(
                        0, 2, 4, 1, 3, 5)[mask])
                density_i[bad_mask] = 0
                colors_i[bad_mask] = 0
            else:
                density_i = density.reshape((pow2, 2, pow2, 2, pow2, 2)).transpose(
                        0, 2, 4, 1, 3, 5)[mask].reshape(-1, 2, 2, 2, 1)
                colors_i = colors.reshape(
                        (pow2, 2, pow2, 2, pow2, 2, data_dim - 1)).transpose(
                        0, 2, 4, 1, 3, 5, 6)[mask].reshape(-1, 2, 2, 2, data_dim - 1)
            data = np.concatenate([colors_i.astype(np.float16),
                                   density_i.astype(np.float16)], -1)
        else:
            # Construct an internal level
            curr_indices = np.cumsum(mask.reshape(-1), dtype=np.uint32)
            next_indices = np.cumsum(next_mask.reshape(-1), dtype=np.uint32) + nnodes

            child = (next_indices.reshape(pow2, 2, pow2, 2, pow2, 2) -
                     curr_indices.reshape(pow2, 1, pow2, 1, pow2, 1))
            child = child.reshape(pow2 * 2, pow2 * 2, pow2 * 2)
            child[~next_mask] = 0
            child = child.reshape(pow2, 2, pow2, 2, pow2, 2)
            child = child.transpose(0, 2, 4, 1, 3, 5).reshape(pow2 ** 3, 2, 2, 2)
            child = child[mask.reshape(-1)]

            # For now, all interior nodes will be empty
            data = np.zeros((nnodes, 2, 2, 2, data_dim), dtype=np.float16);

        all_child.append(child)
        all_data.append(data)

    child = np.concatenate(all_child, 0)
    data = np.concatenate(all_data, 0)
    result['child'] = child.view(np.int32)
    result['data'] = data
    return result

if __name__ == '__main__':
    # Load some 256x256x256 1-channel uint8 volume data
    demo_density = np.load('vol.npy').astype(dtype=np.float32) / 255
    assert demo_density.ndim == 3

    demo_density *= 2
    demo_colors = np.zeros((256, 256, 256, 3), dtype=np.float32)
    results = vol2plenoctree(demo_density, demo_colors)
    #  new_results = {}
    #  for k in results:
    #      new_results['myvolume/1'] = 'volume'
    #      new_results['myvolume/1__' + k] = results[k]
    #      new_results['myvolume/2'] = 'volume'
    #      new_results['myvolume/2__' + k] = results[k]
    #
    #  new_results['myvolume/2__translation'] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    np.savez_compressed('mytree.npz', **results)
