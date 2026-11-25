"""Classify one local photo and write the versioned result as JSON."""

import argparse
import json

from server import util


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to one JPEG, PNG, or WebP image")
    args = parser.parse_args()
    try:
        result = util.classify_faces(util.read_image_file(args.image))
    except (OSError, util.ImageInputError, util.ClassifierUnavailable) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
