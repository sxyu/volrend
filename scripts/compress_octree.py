"""
Octree compression code: quantization using median cut algorithm (in SVOX) + savez_compressed.
Tree can still be opened by volrend after compression.
Require svox: pip install svox, and a CUDA-capable GPU.

Basic usage:
python compress_octree.py x.npz [y.npz ...]
"""
import sys
import numpy as np
import os.path as osp
import torch
from svox.helpers import _get_c_extension
from tqdm import tqdm
import os
import argparse

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, nargs='+', default=None, help='Input npz(s)')
    parser.add_argument('--noquant', action='store_true',
            help='Disable quantization')
    parser.add_argument('--bits', type=int, default=16,
            help='Quantization bits (order)')
    parser.add_argument('--out_dir', type=str, default='min_alt',
            help='Where to write compressed npz')
    parser.add_argument('--overwrite', action='store_true',
            help='Overwrite existing compressed npz')
    parser.add_argument('--weighted', action='store_true',
            help='Use weighted median cut (seems quite useless)')
    parser.add_argument('--sigma_thresh', type=float, default=2.0,
            help='Kill voxels under this sigma')
    parser.add_argument('--retain', type=int, default=1,
            help='Do not compress first x SH coeffs, ' +
                 'needed for some scenes to keep good quality. For lego use --retain 4')

    args = parser.parse_args()

    _C = _get_c_extension()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.noquant:
        print('Quantization disabled, only applying deflate')
    else:
        print('Quantization enabled')

    for fname in args.input:
        fname_c = osp.join(args.out_dir, osp.basename(fname))
        print('Compressing', fname, 'to', fname_c)
        if not args.overwrite and osp.exists(fname_c):
            print(' > skip')
            continue

        z = np.load(fname)

        if not args.noquant:
            if 'quant_colors' in z.files:
                print(' > skip since source already compressed')
                continue
        z = dict(z)
        del z['parent_depth']
        del z['geom_resize_fact']
        del z['n_free']
        del z['n_internal']
        del z['depth_limit']

        if not args.noquant:
            data = torch.from_numpy(z['data'])
            sigma = data[..., -1].reshape(-1)
            snz = sigma > args.sigma_thresh
            sigma[~snz] = 0.0

            data = data[..., :-1]
            N = data.size(1)
            basis_dim = data.size(-1) // 3

            data = data.reshape(-1, 3, basis_dim).float()[snz].unbind(-1)
            if args.retain:
                retained = data[:args.retain]
                data = data[args.retain:]
            else:
                retained = None

            all_quant_colors = []
            all_quant_maps = []
            
            if args.weighted:
                weights = 1.0 - np.exp(-0.01 * sigma.float(float32))
            else:
                weights = torch.empty((0,))

            for i, d in tqdm(enumerate(data), total=len(data)):
                colors, color_id_map = _C.quantize_median_cut(d.contiguous(),
                                                              weights,
                                                              args.bits)
                color_id_map_full = np.zeros((snz.shape[0],), dtype=np.uint16)
                color_id_map_full[snz] = color_id_map

                all_quant_colors.append(colors.numpy().astype(np.float16))
                all_quant_maps.append(color_id_map_full.reshape(-1, N, N, N).astype(np.uint16))
            quant_map = np.stack(all_quant_maps, axis=0)
            quant_colors = np.stack(all_quant_colors, axis=0)
            del all_quant_maps
            del all_quant_colors
            z['quant_colors'] = quant_colors
            z['quant_map'] = quant_map
            z['sigma'] = sigma.reshape(-1, N, N, N)
            if args.retain:
                all_retained = []
                for i in range(args.retain):
                    retained_wz = np.zeros((snz.shape[0], 3), dtype=np.float16)
                    retained_wz[snz] = retained[i]
                    all_retained.append(retained_wz.reshape(-1, N, N, N, 3))
                all_retained = np.stack(all_retained, axis=0)
                del retained
                z['data_retained'] = all_retained
            del z['data']
        np.savez_compressed(fname_c, **z)
        print(' > Size', osp.getsize(fname) // (1024 * 1024), 'MB ->',
                osp.getsize(fname_c)  // (1024 * 1024), 'MB')


if __name__ == '__main__':
    main()
