#!/usr/bin/env python

import logging
import os
import argparse
import sys
import glob
from sgdm.osdba import build_dbas_data_frame


def main(args):
    """Return the time-sorted list of dbas in the current working directory"""
    # Set up logger
    log_level = getattr(logging, args.loglevel.upper())
    log_format = '%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
    logging.basicConfig(format=log_format, level=log_level)

    if not os.path.isdir(args.dba_path):
        logging.error('Invalid search path specified: {:}'.format(args.dba_path))
        return 1

    file_listing = glob.glob(os.path.join(args.dba_path, '*.dat'))
    if not file_listing:
        logging.warning('No files found: {:}'.format(args.dba_path))
        return 1

    dbas_df = build_dbas_data_frame(file_listing).reset_index()
    if dbas_df.empty:
        return 1

    dbas_df.reset_index()

    max_file_length = 0
    max_bytes = 0

    sys.stdout.write('{:<32} {:<8} {:}\n'.format('filename', 'Kb', 'start_time'))

    for r in dbas_df.iterrows():
        file_length = len(r[1].file)
        if file_length > max_file_length:
            max_file_length = file_length

        if r[1].bytes > max_bytes:
            max_bytes = r[1].bytes

        sys.stdout.write('{:<32} {:>5}Kb {:}\n'.format(r[1].file, '{:0.1f}'.format(r[1].bytes/1000), r[1].created_time))

    return 0


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('dba_path',
                            nargs='?',
                            default=os.curdir,
                            help='Path to the dba files. Default is pwd')

    arg_parser.add_argument('-e', '--ext',
                            default='.dat',
                            help='Specify an alternate dba file extension.')

    arg_parser.add_argument('-l', '--loglevel',
                            help='Verbosity level',
                            type=str,
                            choices=['debug', 'info', 'warning', 'error', 'critical'],
                            default='info')

    parsed_args = arg_parser.parse_args()

    # print(parsed_args)
    # sys.exit(1)

    sys.exit(main(parsed_args))