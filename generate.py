#!/usr/bin/env python3

import pyphantom
import sys

if __name__ == "__main__":
    cmdparser = pyphantom.cmdline.parser(sys.argv[0])
    args = cmdparser.parse_args()
    phantom_out = pyphantom.phantoms.generator.get_phantom(
        args.ptype, args.shape, args.position, args.radius
    )
    pyphantom.fileio.save_phantom_all(phantom_out, args.outdir)
