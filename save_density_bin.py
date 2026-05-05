#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from density_field import sample_grid
from density_field import schwarz_p, gyroid, schwarz_d, neovius, lidinoid, schoen_iwp

DENSITY_FIELDS = {
    'schwarz_p': schwarz_p,
    'gyroid': gyroid,
    'schwarz_d': schwarz_d,
    'neovius': neovius,
    'lidinoid': lidinoid,
    'schoen_iwp': schoen_iwp
}


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        Command-line interface.

    """
    parser = argparse.ArgumentParser(description='Sample a density field on an n x n x n grid and save it in homo3d raw binary format.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--res', type=int, required=True, help='Sampling grid resolution n.')
    parser.add_argument('--field', choices=sorted(DENSITY_FIELDS), default='schwarz_p', help='Density field to use.')
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent, help='Output directory for saved rho.bin.')
    parser.add_argument('--rho-min', type=float, default=1e-10, help='Lower clamp for densities.')
    parser.add_argument('--rho-max', type=float, default=1.0, help='Upper clamp for densities.')
    parser.add_argument('--save-npy', action='store_true', help='Also save the sampled density as .npy.')

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.res <= 0:
        raise SystemExit('--res must be positive')
    if args.rho_min < 0 or args.rho_max > 1 or args.rho_min > args.rho_max:
        raise SystemExit('Expected 0 <= rho-min <= rho-max <= 1')

    points = sample_grid(n=args.res ** 3, dim=3)
    rho = DENSITY_FIELDS[args.field](points=points)
    rho = np.clip(rho, args.rho_min, args.rho_max)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    path = args.out_dir.resolve() / 'rho.bin'

    with open(file=path, mode='wb') as f:
        # Write header containing field dimensions
        header = np.array([args.res, args.res, args.res], dtype=np.int32)
        header.tofile(file=f)

        # Write field data
        body = rho.astype(np.float32)
        body.tofile(file=f)

    if args.save_npy:
        np.save(path.with_suffix('.npy'), rho)

    print(f'Wrote binary density: {path}')
    print(f'Resolution: {args.res}x{args.res}x{args.res}')
    print(f'Min density: {rho.min():.6f}')
    print(f'Max density: {rho.max():.6f}')
    print(f'Average density: {rho.mean():.6f}')
    print('')
    print('Next step: Convert to VDB with the C++ tool:')
    print(f"./save_density_vdb {path} {path.with_suffix('.vdb')}")


if __name__ == "__main__":
    main()
