import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser("adrtlib")
    parser.add_argument(
        "--include-dir",
        action="store_true",
        help="Print the path to the adrtlib C++ header directory.",
    )
    parser.add_argument(
        "--cmake-dir",
        action="store_true",
        help="Print the path to the adrtlib CMake module directory.",
    )
    args = parser.parse_args()
    if len(sys.argv) <= 1:
        parser.print_help()
    if args.include_dir:
        print(str(Path(__file__).parent / "include"))
    if args.cmake_dir:
        print(str(Path(__file__).parent))


main()
