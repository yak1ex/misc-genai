"""A module/script to handle PNG Info for generative AI for images.

Current supported PNG Info style:

    * Stable Diffusion web UI(a1111)
      * PNG
      * EXIF
    * Fooocus
      * PNG, both scheme of fooocus and a1111
      * EXIF, both scheme of fooocus and a1111
      
Typical usage example as a module:

    from PIL import Image
    import pnginfo

    pnginfo_dict = pnginfo.load('some_generated.png')
    im = Image.open('some.png')
    pnginfo.save(im, pnginfo_dict, 'some_transplanted.png')
    
Typical usage example as a script:

    # dump PNG Info as JSON
    python pnginfo.py --load some_generated.png
    # show help
    python pnginfo.py --help    
"""
import argparse
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from piexif.helper import UserComment
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
import pillow_avif

def to_pnginfo(pnginfo_dict: dict[str,str]|None) -> PngInfo|None:
    """Convert pnginfo dictionary to PngInfo

    Args:
        pnginfo_dict (dict[str,str] | None): dictionary object to be converted to PngInfo

    Returns:
        PngInfo|None: If None is given or no parameters are given, None is returned.
        Otherwise, converted PngInfo is returned.
    """
    if pnginfo_dict is None or 'parameters' not in pnginfo_dict:
        return None
    pnginfo = PngInfo()
    pnginfo.add_text('parameters', pnginfo_dict['parameters'])
    if 'fooocus_scheme' in pnginfo_dict:
        pnginfo.add_text('fooocus_scheme', pnginfo_dict['fooocus_scheme'])
    return pnginfo

def from_pnginfo(pnginfo: PngInfo) -> dict[str, str]:
    pnginfo_dict = {}
    for k,v in pnginfo.chunks:
        if k != b'tEXt':
            raise ValueError('chunk key is not an expected value')
        pnginfo_dict |= dict([map(bytes.decode, v.split(b'\x00'))])
    return pnginfo_dict

def load(in_filename, image_format: str|None=None) -> dict[str,str]:
    if image_format is None:
        extension = os.path.splitext(in_filename)[1]
        if extension in Image.registered_extensions():
            image_format = Image.registered_extensions()[extension]
        else:
            image_format = extension.upper()

    pnginfo_dict = {}
    if image_format.upper() == 'PNG':
        image = Image.open(in_filename)
        if 'parameters' in image.text:
            pnginfo_dict['parameters'] = image.text['parameters']
        if 'fooocus_scheme' in image.text:
            pnginfo_dict['fooocus_scheme'] = image.text['fooocus_scheme']
    elif image_format.upper() == 'GIF':
        image = Image.open(in_filename)
        if 'comment' in image.info:
            pnginfo_dict['parameters'] = image.info['comment']
    else:
        image = Image.open(in_filename)
        exif = image.getexif()
        # Fooocus uses IFD0, while SD webui uses IFD.Exif
        if ExifTags.Base.Software in exif:  # TODO: check Fooocus v\d\.\d\.\d
            pnginfo_dict['parameters'] = exif[ExifTags.Base.UserComment]
            pnginfo_dict['fooocus_scheme'] = exif[ExifTags.Base.MakerNote]
        else:
            ifd = exif.get_ifd(ExifTags.IFD.Exif)
            if ExifTags.Base.UserComment in ifd:
                pnginfo_dict['parameters'] = UserComment.load(ifd[ExifTags.Base.UserComment])
    return pnginfo_dict

def save(image: Image, pnginfo: PngInfo|dict[str,str]|None, out_filename, image_format=None, **kw) -> None:
    if image_format is None:
        extension = os.path.splitext(out_filename)[1]
        if extension in Image.registered_extensions():
            image_format = Image.registered_extensions()[extension]
        else:
            image_format = extension.upper()

    if image_format.upper() == 'PNG':
        if not isinstance(pnginfo, PngInfo):
            pnginfo = to_pnginfo(pnginfo)
        image.save(out_filename, format=image_format, pnginfo=pnginfo, **kw)
    else:
        if isinstance(pnginfo, PngInfo):
            pnginfo = from_pnginfo(pnginfo)
        exif = Image.Exif()
        if pnginfo is None:
            pass
        elif 'fooocus_scheme' in pnginfo:
            if pnginfo['fooocus_scheme'] == 'fooocus':
                exif[ExifTags.Base.Software] = pnginfo['parameters']['Version']
            else: # fooocus_scheme: a1111
                if m := re.match(r'Version: ([a-zA-Z 0-9.])', pnginfo['parameters']):
                    exif[ExifTags.Base.Software] = m.group(1)
                else:
                    exif[ExifTags.Base.Software] = 'Fooocus v0.0.0' # Unknown
            exif[ExifTags.Base.MakerNote] = pnginfo['fooocus_scheme']
            exif[ExifTags.Base.UserComment] = pnginfo['parameters']
        else: # SD webui
            ifd = exif.get_ifd(ExifTags.IFD.Exif)
            ifd[ExifTags.Base.UserComment] = UserComment.dump(pnginfo['parameters'], encoding="unicode")
        image.save(out_filename, format=image_format, exif=exif.tobytes() if pnginfo is not None else None, **kw)

def remove(in_filename, out_filename, image_format=None, **kw) -> None:
    image = Image.open(in_filename)
    save(image, None, out_filename, image_format=image_format, **kw)

def transplant(in_filename, out_filename, keep_mtime=False):
    pass


if __name__ == '__main__':
    @contextmanager
    def keep_mtime(flag, input, output):
        stat_in = os.stat(input)
        yield None
        if os.path.exists(output):
            os.utime(output, ns=(stat_in.st_atime_ns, stat_in.st_mtime_ns))

    def check_args(args):
        if args.load:
            if args.keep_mtime:
                raise argparse.ArgumentError(load_action, '--keep-mtime has no effect')
            if args.inplace:
                raise argparse.ArgumentError(load_action, '--inplace has no effect')
            if len(args.filename) != 1:
                raise argparse.ArgumentError(load_action, 'requires exactly 1 filename argument')
        else: # update operations
            if args.inplace and len(args.filename) != 1:
                raise argparse.ArgumentError(None, 'Update operation with --inplace requires exactly 1 filename argument')
            elif not args.inplace and len(args.filename) != 2:
                raise argparse.ArgumentError(None, 'Update operation requires exactly 2 filename arguments')

    parser = argparse.ArgumentParser(
        prog='pnginfo.py',
        description='A script to handle PNG Info for generative AI for images',
    )
    operation = parser.add_mutually_exclusive_group(required=True)
    load_action = operation.add_argument('--load', '-l', action='store_true')
    remove_action = operation.add_argument('--remove', '-r', action='store_true')
    trans_action = operation.add_argument('--transplant', '-t', action='store_true')
    savefile_action = operation.add_argument('--savefile', '-f', metavar='INFO_JSON_FILE')
    savetext_action = operation.add_argument('--savetext', '-T', metavar='INFO_JSON_TEXT')
    inplace_action = parser.add_argument('--inplace', '-i', action='store_true', help='output file is the same as input file')
    keeptime_action = parser.add_argument('--keep-mtime', '-k', action='store_true', help='keep mtime for update operations')
    parser.add_argument('filename', nargs='+')
    args = parser.parse_args()
    check_args(args)
    if args.load:
        print(json.dumps(load(args.filename[0])))
    else:
        in_filename, out_filename = args.filename[0], args.filename[0 if args.inplace else 1]
        with keep_mtime(args.keep_mtime, in_filename, out_filename):
            if args.savefile:
                with open(args.savefile) as info:
                    im = Image.open(in_filename)
                    save(im, json.load(info), out_filename)
            elif args.savetext:
                im = Image.open(in_filename)
                save(im, json.loads(args.savetext), out_filename)
            elif args.remove:
                remove(in_filename, out_filename)
            elif args.transplant:
                pnginfo_dict = load(in_filename)
                im = Image.open(out_filename)
                save(im, pnginfo_dict, out_filename)
            else:
                pass
