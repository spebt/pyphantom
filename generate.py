if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="generate.py",
        description="Python package for digital phantom generation",
        epilog="For more information, visit https:\\\\spebt.github.io\\pyphantom",
        add_help=True,
    )
    type_choises = ["hotrod", "Derenzo", "derenzo", "contrast", "dot", "disk"]
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        help="Phantom type\n available phantom types",
        choices=type_choises,
        required=True,
    )
    parser.add_argument(
        "-o", "--outdir", type=str, help="Output directory", default="output"
    )

    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        parser.print_help()
        raise SyntaxError("Invalid Arguments")
