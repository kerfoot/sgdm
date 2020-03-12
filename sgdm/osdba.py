"""Function to search for and return Teledyne Webb Research Slocum glider Dinkum Binary ASCII (dba) files"""

import os
import logging
import pandas as pd
import glob
from operator import itemgetter
import pytz
from dateutil import parser
from sgdm.constants import dba_data_types

logger = logging.getLogger(os.path.basename(__name__))


def get_dbas(dba_dir, dt0=None, dt1=None):
    """Search for all dba files in dba_dir, optionally filtered by fileopen time.

    Parameters:
        dba_dir: Path to search
        dt0: starting datetime
        dt1: ending datetime

    Returns a pandas data frame indexed by fileopen_time
    """

    dbas = ls_dbas(dba_dir, dt0=dt0, dt1=dt1)

    return build_dbas_data_frame(dbas)


def build_dbas_data_frame(dba_files):
    dba_records = []
    for dba_file in dba_files:

        header = parse_dba_header(dba_file)

        date_pieces = header['fileopen_time'].split('_')

        header['created_time'] = parser.parse(
            '{:} {:}, {:} {:}'.format(date_pieces[1], date_pieces[2], date_pieces[4], date_pieces[3])).replace(
            tzinfo=pytz.UTC)
        header['file'] = os.path.basename(dba_file)
        header['path'] = os.path.dirname(dba_file)
        header['bytes'] = os.path.getsize(dba_file)

        columns = header.keys()

        ordered_columns = ['file']
        for c in columns:
            if c not in ordered_columns:
                ordered_columns.append(c)

        dba_records.append(
            pd.DataFrame([[header[c] for c in ordered_columns]], columns=ordered_columns).set_index('created_time'))

    dbas_df = pd.concat(dba_records)

    dbas_df.sort_index(inplace=True)

    return dbas_df


def ls_dbas(dba_dir, dt0=None, dt1=None):
    """Search for all dba files in dba_dir, optionally filtered by fileopen time.

        Parameters:
            dba_dir: Path to search
            dt0: starting datetime
            dt1: ending datetime

        Returns a list of dba files sorted by fileopen_time
    """

    if not os.path.isdir(dba_dir):
        logger.error('Invalid directory specified: {:}'.format(dba_dir))

    all_dbas = glob.glob(os.path.join(dba_dir, '*'))
    if not all_dbas:
        logger.warning('No files found')
        return

    if dt0:
        dt0 = dt0.replace(tzinfo=pytz.UTC)
    if dt1:
        dt1 = dt1.replace(tzinfo=pytz.UTC)

    dbas = []
    for dba_file in all_dbas:

        if not os.path.isfile(dba_file):
            continue

        header = parse_dba_header(dba_file)
        if not header:
            continue

        date_pieces = header['fileopen_time'].split('_')

        dt = parser.parse(
            '{:} {:}, {:} {:}'.format(date_pieces[1], date_pieces[2], date_pieces[4], date_pieces[3])).replace(
            tzinfo=pytz.UTC)

        if dt0:
            if dt < dt0:
                continue

        if dt1:
            if dt > dt1:
                continue

        dba = {'file': dba_file, 'dt0': dt}

        dbas.append(dba)

    if dbas:
        dbas.sort(key=itemgetter('dt0'))
        dbas = [dba['file'] for dba in dbas]

    return dbas


def parse_dba_header(dba_file):
    """Parse the header lines of a Slocum dba ascii table file

    Args:
        dba_file: dba file to parse

    Returns:
        An dictionary mapping heading keys to values
    """

    if not os.path.isfile(dba_file):
        logger.error('Invalid DBA file specified: {:}'.format(dba_file))
        return

    try:
        with open(dba_file, 'r') as fid:

            dba_headers = {}

            # Get the first line of the file to make sure it starts with 'dbd_label:'
            f = fid.readline()
            if not f.startswith('dbd_label:'):
                return

            tokens = f.strip().split(': ')
            if len(tokens) != 2:
                logger.error('Invalid dba file {:}'.format(dba_file))
                return

            dba_headers[tokens[0]] = tokens[1]

            for f in fid:

                tokens = f.strip().split(': ')
                if len(tokens) != 2:
                    break

                dba_headers[tokens[0]] = tokens[1]
    except IOError as e:
        logger.error('Error parsing {:s} dba header: {}'.format(dba_file, e))
        return

    if not dba_headers:
        logger.warning('No headers parsed: {:s}'.format(dba_file))

    return dba_headers


def parse_dba_sensor_defs(dba_file):
    """Parse the sensor definitions in a Slocum dba ascii table file.

    Args:
        dba_file: dba file to parse

    Returns:
        An array of dictionaries containing the file sensor definitions
    """

    if not os.path.isfile(dba_file):
        logger.error('Invalid DBA file specified: {:}'.format(dba_file))
        return

    # Parse the file header lines
    dba_headers = parse_dba_header(dba_file)
    if not dba_headers:
        return

    if 'num_ascii_tags' not in dba_headers:
        logger.warning('num_ascii_tags header missing: {:s}'.format(dba_file))
        return

    # Sensor definitions begin on the line number after that contained in the
    # dba_headers['num_ascii_tags']
    num_header_lines = int(dba_headers['num_ascii_tags'])

    sensor_metadata = None
    try:
        with open(dba_file, 'r') as fid:

            line_count = 0
            while line_count < num_header_lines:
                fid.readline()
                line_count += 1

            # Get the sensor names line
            sensors_line = fid.readline().strip()
            # Get the sensor units line
            units_line = fid.readline().strip()
            # Get the datatype byte storage information
            bytes_line = fid.readline().strip()

            sensors = sensors_line.split()
            units = units_line.split()
            datatype_bytes = bytes_line.split()

            if len(sensors) != len(units) or len(sensors) != len(datatype_bytes):
                logger.warning('Incomplete sensors, units or dtypes definition lines: {:}'.format(dba_file))
                return

            sensor_metadata = [
                {'native_sensor_name': sensors[s],
                 'units': units[s],
                 'dtype': dba_data_types.get(datatype_bytes[s], 'f8')
                 } for s in
                range(len(sensors))]

    except IOError as e:
        logger.error('Error parsing {:s} dba header: {:s}'.format(dba_file, e))

    return sensor_metadata
