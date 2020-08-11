#!/usr/bin/env python

import argparse
import os
import sys
import logging
import pandas as pd
import shutil
from sgdm.qc.dev import load_limits_config, check_file_bounds_by_depth


def main(args):
    """Check one or more NetCDF files for local bounds as set by the specified rules file"""

    # Set up logger
    log_level = getattr(logging, args.loglevel.upper())
    log_format = '%(asctime)s:%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
    logging.basicConfig(format=log_format, level=log_level)

    nc_files = args.nc_files
    depth_sensor = args.depth_sensor
    max_outliers = args.max_outliers

    qc_rules = load_limits_config(args.rules_file)
    if not qc_rules:
        return 1

    all_results = pd.DataFrame()
    for nc_file in nc_files:
        file_results = check_file_bounds_by_depth(nc_file, qc_rules, depth_sensor=depth_sensor)
        total_outliers = file_results.total_outliers.sum()

        if total_outliers > max_outliers:
            logging.warning('Failed bounds check(s): {:}'.format(nc_file))

            if args.ext:
                logging.info('Flagging file: {:}'.format(nc_file))
                flagged_file = '{:}.{:}'.format(nc_file, args.ext)
                shutil.move(nc_file, flagged_file)

        all_results = pd.concat([all_results, file_results])
        all_results.reset_index(drop=True, inplace=True)
        all_results.index.rename('test_count', inplace=True)

    sys.stdout.write('{:}\n'.format(all_results.to_csv()))

    return 0


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('nc_files',
                            nargs='+',
                            help='One or more NetCDF files to check')

    arg_parser.add_argument('rules_file',
                            help='YAML file containing the rules to enforce')

    arg_parser.add_argument('-n', '--max_outliers',
                            type=int,
                            help='Maximum number of allowable outliers',
                            default=0)

    arg_parser.add_argument('-z', '--depth_sensor',
                            type=str,
                            help='Name of depth/pressure variable',
                            default='depth')

    arg_parser.add_argument('-e', '--ext',
                            type=str,
                            help='If specified, string will be appended to each file that violates one or more rules')

    arg_parser.add_argument('-x', '--debug',
                            help='Check configuration and create NetCDF file writer, but does not process any files',
                            action='store_true')

    arg_parser.add_argument('-l', '--loglevel',
                            help='Verbosity level',
                            type=str,
                            choices=['debug', 'info', 'warning', 'error', 'critical'],
                            default='info')

    parsed_args = arg_parser.parse_args()

    sys.exit(main(parsed_args))
