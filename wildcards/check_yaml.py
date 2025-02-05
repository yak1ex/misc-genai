import os
import sys
import yaml
from pathlib import Path


def proc_file(target: Path):
    with target.open(encoding='utf8') as input_stream:
        print(target)
        yaml.safe_load(input_stream)


def proc(targets):
    for target in targets:
        target_path = Path(target)
        if target_path.is_dir():
            for root, dirs, files in os.walk(target):
                for file in filter(lambda x: x.endswith('.yaml'), files):
                    proc_file(Path(root, file))
        else:
            proc_file(target_path)


if __name__ == '__main__':
    proc(sys.argv[1:])
