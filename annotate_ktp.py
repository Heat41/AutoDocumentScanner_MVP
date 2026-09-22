import sys

from autodocscanner.tools.ktp_annotator import main


if __name__ == "__main__":
    initial_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )
    main(
        initial_path=initial_path
    )
