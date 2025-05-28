from typing import Dict


def print_dict(data: Dict, level: int = 0):
    """
    Print the contents of a dictionary.
    """
    for key in data.keys():
        if isinstance(data[key], dict):
            sub_level = level + 1
            print("  " * sub_level + f"{key:32s}:")
            print_dict(data[key], sub_level + 1)
        elif isinstance(data[key], Tensor):
            print("  " * level + f"{key:32s}: ", end="")
            print(f" Tensor({tuple(data[key].shape)},dtype={data[key].dtype})")
        else:
            print("  " * level + f"{key:32s}: ", end="")
            print("  " * level, data[key])


if __name__ == "__main__":
    import os, sys
    from torch import load as torch_load, Tensor

    in_filename = sys.argv[1]

    # Check if the file exists
    if not os.path.isfile(in_filename):
        print(f"File {in_filename} does not exist.")
        sys.exit(1)

    # Read the file
    try:
        data = torch_load(in_filename, map_location="cpu")
        print(f"Content of {in_filename}:")
        print_dict(data)
    except Exception as e:
        print(f"Error reading file {in_filename}: {e}")
