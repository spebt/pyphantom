import argparse


def parser(program: str) -> argparse.ArgumentParser:
    myparser = argparse.ArgumentParser(
        prog=program,
        description="Python package for digital phantom generation",
        epilog="For more information, visit https:\\\\spebt.github.io\\pyphantom",
        add_help=True,
    )
    type_choises = ["derenzo", "contrast", "dot", "disk"]
    myparser.add_argument(
        "-t",
        "--type",
        type=str,
        dest="ptype",
        help="Phantom type\n available phantom types",
        metavar="TYPE",
        choices=type_choises,
        required=True,
    )
    myparser.add_argument(
        "-s",
        "--shape",
        type=int,
        nargs=2,
        help="Phantom shape",
        metavar="(WIDTH, HEIGHT)",
        default=(100, 100),
    )
    myparser.add_argument(
        "-p",
        "--position",
        type=int,
        nargs=2,
        help="disk center position or dot position",
        metavar=("POSX", "POSY"),
        default=(50, 50),
    )
    myparser.add_argument(
        "-r", "--radius", type=int, help="Disk phantom radius", default=10
    )
    myparser.add_argument(
        "-o", "--outdir", type=str, help="Output directory", default="output"
    )

    return myparser
