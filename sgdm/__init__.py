import os
import logging
import pandas as pd
import numpy as np
import math
import glob
import yaml
import datetime
from copy import deepcopy
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glidertools as gt
# from matplotlib import cm
from sgdm.osdba import build_dbas_data_frame, parse_dba_header, parse_dba_sensor_defs
from sgdm.constants import dba_data_types, default_encoding
# from sgdm.attributes import default_attributes
from sgdm.gps import dm2dd, interpolate_gps
import sgdm.ctd as ctd
from sgdm.yo import find_profiles
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from sgdm.process import calculate_series_first_differences


class Dba(object):

    def __init__(self, dba_files, keep_gld_dups=False, gps=False, ctd=False, profiles=False):

        self._logger = logging.getLogger(os.path.basename(__name__))

        # Location of metadata files for
        self._metadata_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'metadata'))
        if not os.path.isdir(self._metadata_path):
            raise OSError('Invalid metadata path: {:}'.format(self._metadata_path))
        self._default_attributes = {}
        self._load_default_attributes()

        # Convert dba_files to an array if it's a single file (str)
        if isinstance(dba_files, str):
            dba_files = [dba_files]

        if not dba_files:
            raise OSError('No dba files specified')

        # self._dba_files = [dba for dba in dba_files if os.path.isfile(dba)]
        self._dba_files = pd.DataFrame()

        self._process_gps = gps
        self._process_ctd = ctd
        self._index_profiles = profiles

        # Valid profile indices for dumping to xarray dataset
        self._profile_index_types = ['profile_id', 'profile_time']

        # List of renamed/derived paramters to initialize
        self._new_columns = [
            'pressure_raw',
            'temperature_raw',
            'conductivity_raw',
            'depth_raw',
            'practical_salinity_raw',
            'density_raw',
            'sound_speed_raw',
            'latitude',
            'longitude',
            'profile_time',
            'profile_dir',
            'profile_id'
        ]

        # Flag to keep gld_dup* sensors
        self._keep_gld_dups = bool(keep_gld_dups)

        # Pandas data frame
        self._data_frame = None

        # Start/stop time of indexed profiles
        self._profiles = pd.DataFrame()

        # dba header dict
        self._dba_headers = {}
        self._dba_sensor_metadata = []
        # Metadata for data frame columns
        self._column_defs = {}

        # Valid depth variables
        self._depth_vars = []
        # Name of depth sensor used to index profiles
        self._depth_sensor = None

        # Valid timestamp variables
        self._time_vars = []

        # Parse all specified dba files to an aggregated pandas dataframe
        self._load_dbas_to_data_frame(dba_files)

        # # Build the dba data files data frame with the file headers
        # self._build_files_data_frame()

        # Build the default column definitions from self._dba_sensor_metadata
        self._build_default_column_defs()

        # Add self._new_columns to the resulting data frame
        for c in self._new_columns:
            self._logger.debug('Initializing derived column: {:}'.format(c))
            self._data_frame[c] = np.nan

            self._column_defs[c] = {'nc_var_name': c, 'type': 'f8', 'attrs': {}}

        # Set the available depth/pressure sensors
        depth_units = ['bar', 'decibar', 'm']
        depth_sensors = [s['native_sensor_name'] for s in self._dba_sensor_metadata if s['units'] in depth_units]
        self._depth_vars = [s for s in depth_sensors if s.find('altitude') < 0]

        if 'sci_water_pressure' in self._depth_vars:
            self._depth_sensor = 'sci_water_pressure'
        else:
            self._depth_sensor = self._depth_vars[0]

        if gps:
            self._logger.info('Processing GPS...')
            self.process_gps()
            self._logger.info('GPS processing complete')

        if ctd:
            self._logger.info('Processing raw CTD ...')
            self.process_raw_ctd()
            self._logger.info('Raw CTD processing complete')

        if profiles:
            self._logger.info('Indexing profiles...')
            self.index_profiles()
            self._logger.info('Profile indexing complete. {:} profiles indexed'.format(self._profiles.shape[0]))

    @property
    def metadata_path(self):
        if not os.path.isdir(self._metadata_path):
            self._logger.warning('Invalid metadata path {:}'.format(self._metadata_path))
            self._logger.warning('Default attributes will not be added to columns')
        return self._metadata_path

    @property
    def default_attributes(self):
        return self._default_attributes

    @property
    def dba_files(self):
        return self._dba_files

    @property
    def keep_gld_dups(self):
        return self._keep_gld_dups

    @keep_gld_dups.setter
    def keep_gld_dups(self, boolean):
        self._keep_gld_dups = bool(boolean)

    @property
    def data(self):
        return self._data_frame

    @property
    def profiles(self):
        return self._profiles

    @property
    def segment_profiles(self):
        if self._profiles.empty:
            return pd.DataFrame()

        segment_profiles = self._profiles.groupby('segment', sort=False).size().reset_index(name='num_profiles')

        return pd.merge(left=segment_profiles, right=self.dba_files.reset_index(), left_on='segment',
                        right_on='segment_filename_0')[['created_time', 'segment', 'num_profiles']].set_index(
            'created_time')

    @property
    def profiles_summary(self):
        if self._profiles.empty:
            return pd.DataFrame()

        return self._profiles[['start_depth', 'end_depth', 'num_points']]

    @property
    def dba_info(self):
        return self._dba_files[['segment_filename_0', 'bytes']]

    @property
    def dba_sensor_defs(self):
        return self._dba_sensor_metadata

    @property
    def column_defs(self):
        return self._column_defs

    @property
    def time_vars(self):
        return list(self._time_vars)

    @property
    def depth_vars(self):
        return self._depth_vars

    @property
    def depth_sensor(self):
        return self._depth_sensor

    @depth_sensor.setter
    def depth_sensor(self, sensor_name):
        if sensor_name not in self._depth_vars:
            self._logger.error('{:} is not a valid depth/pressure sensor'.format(sensor_name))
            return
        self._depth_sensor = sensor_name

    def add_depth_sensor(self, var_name):
        if var_name not in self._depth_vars:
            self._logger.info('Adding new depth sensor: {:}'.format(var_name))
            self._depth_vars.append(var_name)

        self._depth_sensor = var_name

    def process_gps(self):
        """Convert all GPS type sensors from native NMEA coordinates to decimal degrees and interpolate m_gps_lat
        and m_gps_lon and add as latitude and longitude, respectively, if present"""

        # Find all GPS sensors. GPS sensors have units of 'lat' or 'lon' inside the dba file
        gps_sensors = [{'sensor': s['native_sensor_name'], 'units': s['units']} for s in self._dba_sensor_metadata if
                       s['units'] == 'lat' or s['units'] == 'lon']
        if not gps_sensors:
            self._logger.warning('No GPS variables found')
            return

        # List of all sensors in the dba
        sensors = [s['native_sensor_name'] for s in self._dba_sensor_metadata]

        # Convert each GPS variable from NMEA coordinates to decimal degrees and replace values
        self._logger.info('Converting {:} GPS sensors to decimal degrees'.format(len(gps_sensors)))
        for gps_sensor in gps_sensors:

            if gps_sensor['sensor'] not in self.data.columns:
                self._logger.debug('Skipping missing GPS sensor: {:}'.format(gps_sensor['sensor']))
                continue

            # convert to decimal degrees and replace original NMEA coordinates
            self._logger.info('Converting {:} from NMEA to decimal degree coordinates'.format(gps_sensor['sensor']))
            self._data_frame[gps_sensor['sensor']] = dm2dd(self._data_frame[gps_sensor['sensor']])

            # Find the dba sensor definition
            i = sensors.index(gps_sensor['sensor'])

            # Create the column definition
            column_def = {'nc_var_name': self._dba_sensor_metadata[i]['native_sensor_name'],
                          'type': self._dba_sensor_metadata[i]['dtype'],
                          'attrs': {s: self._dba_sensor_metadata[i][s] for s in self._dba_sensor_metadata[i]}}

            # Add attributes from default_attributes
            if gps_sensor['units'] == 'lat':
                column_def['attrs'].update(self._default_attributes.get('latitude', {}))
            else:
                column_def['attrs'].update(self._default_attributes.get('longitude', {}))

            # Set the long_name attribute to the dba sensor name
            column_def['attrs']['long_name'] = self._dba_sensor_metadata[i]['native_sensor_name']

            # Store the column definition
            self._column_defs[self._dba_sensor_metadata[i]['native_sensor_name']] = column_def

        if 'm_gps_lat' not in self._data_frame and 'm_gps_lon' not in self._data_frame:
            self._logger.warning('Skipping m_gps_lat/m_gps_lon interpolation: Sensors not found in data frame')
            return

        # Interpolate m_gps_lat and m_gps_lon and add as new columns:
        #   m_gps_lat: latitude
        #   m_gps_lon: longitude
        ts = np.array([self.datetime2epoch(dtime) for dtime in self._data_frame.index.values])
        i_lat, i_lon = interpolate_gps(ts, self._data_frame.m_gps_lat, self._data_frame.m_gps_lon)
        self._data_frame['latitude'] = i_lat
        self._data_frame['longitude'] = i_lon

        # Copy the metadata from m_gps_lat and m_gps_lon to latitude and longitude, respectively
        self._column_defs['latitude'] = deepcopy(self._column_defs['m_gps_lat'])
        # Update the nc_var_name
        self._column_defs['latitude']['nc_var_name'] = 'latitude'
        # Update the comment
        self._column_defs['latitude']['attrs']['comment'] = 'Interpolated m_gps_lat values'
        # Update the long_name
        self._column_defs['latitude']['attrs']['long_name'] = 'Latitude'

        self._column_defs['longitude'] = deepcopy(self._column_defs['m_gps_lon'])
        # Update the nc_var_name
        self._column_defs['longitude']['nc_var_name'] = 'longitude'
        # Update the comment
        self._column_defs['longitude']['attrs']['comment'] = 'Interpolated m_gps_lon values'
        # Update the long_name
        self._column_defs['longitude']['attrs']['long_name'] = 'Longitude'

        return

    def process_raw_ctd(self, p='sci_water_pressure', t='sci_water_temp', c='sci_water_cond'):
        """Rename time index, measured CTD parameters and derive CTD products.

        The following columns are added:
        - depth_raw
        - practical_salinity_raw
        - density_raw
        - sound_speed_raw

        The following columns are renamed:
        - data.index.name -> time
        - p -> pressure_raw
        - c -> conductivity_raw
        - t -> temperature_raw
        """

        if t not in self._data_frame:
            self._logger.error('dba temperature variable {:} not found'.format(t))
            return
        if c not in self._data_frame:
            self._logger.error('dba conductivity variable {:} not found'.format(c))
            return

        self._logger.info('Calculating CTD parameters...')

        self._logger.info('Renaming {:} -> time'.format(self._data_frame.index.name))
        self._data_frame.index.rename('time', inplace=True)
        # Create time column definition
        self._column_defs['time'] = deepcopy(self._column_defs['m_present_time'])
        self._column_defs['time']['attrs'].update(self._default_attributes.get('time', {}))
        self._column_defs['time']['nc_var_name'] = 'time'

        self._logger.info('Renaming {:} -> temperature_raw'.format(t))
        self._logger.info('Renaming {:} -> conductivity_raw'.format(c))

        # Drop the default (nan column) sensors before renaming
        self._data_frame.drop(['temperature_raw', 'conductivity_raw'], axis=1, inplace=True)

        self._data_frame.rename(
            columns={t: 'temperature_raw', c: 'conductivity_raw'},
            inplace=True)
        # Update temperature_raw column definitions
        self._column_defs['temperature_raw'] = deepcopy(self._column_defs[t])
        self._column_defs['temperature_raw']['nc_var_name'] = 'temperature_raw'
        self._column_defs['temperature_raw']['attrs'].update(self._default_attributes.get('temperature_raw', {}))
        # Update conductivity_raw column definitions
        self._column_defs['conductivity_raw'] = deepcopy(self._column_defs[c])
        self._column_defs['conductivity_raw']['nc_var_name'] = 'conductivity_raw'
        self._column_defs['conductivity_raw']['attrs'].update(self._default_attributes.get('conductivity_raw', {}))

        # Process pressure sensor:
        # 1. convert from bar to decibar
        # 2. rename to pressure_raw
        if p not in self._column_defs:
            self._logger.warning('Pressure sensor {:} not found in column definitions'.format(p))
            return

        # Get the sensor definition
        p_def = self._column_defs[p]
        # Get the units
        units = p_def['attrs'].get('units', None)
        # Convert units from bar to decibar, if necessary
        if units == 'bar':
            self._logger.info('Converting {:} units from bar to decibar'.format(p))
            self._data_frame[p] *= 10.
            self._column_defs[p]['attrs']['units'] = 'decibar'

        # Rename p to pressure_raw
        self._logger.info('Renaming {:} to pressure_raw'.format(p))
        # Drop the default pressure_raw (nan-column) before renaming
        self._data_frame.drop('pressure_raw', axis=1, inplace=True)
        self._data_frame.rename(columns={p: 'pressure_raw'}, inplace=True)
        self._column_defs['pressure_raw'] = deepcopy(self._column_defs[p])
        # Update column definition
        self._column_defs['pressure_raw']['attrs'].update(self._default_attributes.get('pressure_raw', {}))
        # Update attributes
        self._column_defs['pressure_raw']['nc_var_name'] = 'pressure_raw'

        # Add pressure_raw to self._depth_vars
        self._depth_vars.append('pressure_raw')

        # Calculate and add depth_raw from pressure_raw and latitude
        if 'latitude' in self._data_frame:
            self._logger.info('Calculating & adding depth_raw from {:} and latitude'.format(p))
            self._data_frame['depth_raw'] = ctd.calculate_depth(self._data_frame.pressure_raw.values,
                                                                self._data_frame.latitude.values)
            # Add column definition
            self._column_defs['depth_raw'] = {'nc_var_name': 'depth_raw', 'dtype': 'f4', 'attrs': {}}
            self._column_defs['depth_raw']['attrs'] = self._default_attributes.get('depth_raw', {})

            # Add depth_raw to self._depth_vars
            self._depth_vars.append('depth_raw')

            # Set self._depth_sensor to depth_raw
            self._depth_sensor = 'depth_raw'

        # Calculate and add practical_salinity_raw
        self._logger.info('Calculating & adding practical_salinity_raw')
        self._data_frame.drop('practical_salinity_raw', axis=1, inplace=True)
        self._data_frame['practical_salinity_raw'] = ctd.calculate_practical_salinity(
            self._data_frame.conductivity_raw.values, self._data_frame.temperature_raw.values,
            self._data_frame.pressure_raw.values)
        # Update column definition
        self._column_defs['practical_salinity_raw']['attrs'] = self._default_attributes.get('practical_salinity_raw',
                                                                                            {})

        # Calculate and add density_raw
        self._logger.info('Calculating & adding density_raw')
        self._data_frame.drop('density_raw', axis=1, inplace=True)
        self._data_frame['density_raw'] = ctd.calculate_density(self._data_frame.temperature_raw.values,
                                                                self._data_frame.pressure_raw.values,
                                                                self._data_frame.practical_salinity_raw.values,
                                                                self._data_frame.latitude.values,
                                                                self._data_frame.longitude.values)
        # Update column definition
        self._column_defs['density_raw']['attrs'] = self._default_attributes.get('density_raw', {})

        # Calculate and add sound_speed_raw
        self._logger.info('Calculating & adding sound_speed_raw')
        self._data_frame.drop('sound_speed_raw', axis=1, inplace=True)
        self._data_frame['sound_speed_raw'] = ctd.calculate_sound_speed(self._data_frame.temperature_raw.values,
                                                                        self._data_frame.pressure_raw.values,
                                                                        self._data_frame.practical_salinity_raw.values,
                                                                        self._data_frame.latitude.values,
                                                                        self._data_frame.longitude.values)

        # Update column definition
        self._column_defs['sound_speed_raw']['attrs'] = self._default_attributes.get('sound_speed_raw', {})

    def index_profiles(self, mindepth=0):

        # Initialize profiles start and stop time array
        indexed_profiles = np.empty((0, 2))

        for segment in self._data_frame.segment.unique():

            # Select the depth_sensor time-series and return a pandas Series
            yo_series = self._data_frame.loc[self._data_frame['segment'] == segment, self._depth_sensor]

            yo_series.mask(yo_series < mindepth, inplace=True)

            epochs = np.array(
                [(ts - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's') for ts in
                 yo_series.index.values])
            yo = np.array([epochs, yo_series.values]).T

            # If the units of depth_sensor are bar, convert to decibars since find_profiles needs meters or decibars to
            # accurately index profiles
            if not self._column_defs[self._depth_sensor]['attrs']['units']:
                self._logger.error('Depth sensor {:} does not have designated units'.format(self._depth_sensor))
                return

            elif self._column_defs[self._depth_sensor]['attrs']['units'] == 'bar':
                yo[:, 1] *= 10

            profile_epoch_times = find_profiles(yo)
            if profile_epoch_times.shape[0] == 0:
                self._logger.debug('No profiles indexed for segment {:}'.format(segment))
                continue

            # Convert the unix time profile indices to datetimes and concatenate to self._profiles
            indexed_profiles = np.concatenate(
                (indexed_profiles, np.array([self.epoch2datetime(t) for t in profile_epoch_times], dtype=object)))

        if indexed_profiles.size == 0:
            return

        self._data_frame.profile_id = np.nan

        # Loop through all self._profiles rows and fill in 'dive' and 'profile_time'
        self._data_frame.profile_time = pd.NaT
        profile_count = 0
        profiles = []
        for (pt0, pt1) in indexed_profiles:

            # Set the profile mid-point time
            # Interval between dt1 and dt0
            pt_delta = pt1 - pt0

            # 1/2 dt_delta and add to dt0 to get the profile mean time
            pt_mean = pt0 + (pt_delta / 2)

            # Mean profile time
            self._data_frame.loc[pt0:pt1, 'profile_time'] = pt_mean
            # Profile counter
            self._data_frame.loc[pt0:pt1, 'profile_id'] = profile_count

            # Add the profile_dir
            profile = self._data_frame[self._depth_sensor].loc[pt0:pt1].dropna()
            profile_dir = ''
            if profile[0] - profile[-1] < 0:
                profile_dir = 'd'
            elif profile[0] - profile[-1] > 0:
                profile_dir = 'u'

            self._data_frame.loc[pt0:pt1, 'profile_dir'] = profile_dir

            profile_info = [pt_mean,
                            pt_delta.total_seconds(),
                            len(profile),
                            profile_dir,
                            pt0,
                            pt1,
                            profile[0],
                            profile[-1],
                            self._data_frame.segment.loc[pt0:pt1].unique()[0],
                            math.fabs(profile[0] - profile[-1])/pt_delta.total_seconds(),
                            (len(profile)/pt_delta.total_seconds()) ** -1]
            profiles.append(profile_info)

            profile_count += 1

        # Add default attributes for profile_id, profile_time and profile_dir
        self._column_defs['profile_id'] = {'attrs': self._default_attributes.get('profile_id', {})}
        self._column_defs['profile_dir'] = {'attrs': self._default_attributes.get('profile_dir', {})}
        self._column_defs['profile_time'] = {'attrs': self._default_attributes.get('profile_time', {})}

        # Create a DataFrame containing the indexed profiles information and indexed on midpoint_time
        profile_cols = ['midpoint_time',
                        'total_seconds',
                        'num_points',
                        'direction',
                        'start_time',
                        'end_time',
                        'start_depth',
                        'end_depth',
                        'segment',
                        'depth_resolution',
                        'sampling_frequency']
        self._profiles = pd.DataFrame(profiles, columns=profile_cols).set_index('midpoint_time')

    def get_tz_variables(self, variable_names=None):
        """Returns a data frame containing the time-series of depth and the specified variable names in which all rows
        containing at least one NaN are removed. The depth column is taken from self.depth_sensor"""

        if not variable_names:
            variable_names = []
        elif not isinstance(variable_names, list):
            variable_names = [variable_names]

        if variable_names:
            for v in variable_names:
                if v not in self._data_frame.columns:
                    self._logger.error('{:} is not a valid variable name'.format(v))
                    return

        extract_vars = [self._depth_sensor] + variable_names
        return self._data_frame[extract_vars].dropna(axis=0, how='any')

    def plot_yo(self, profiles=False):
        """Plot the depth time-series using self.depth_sensor for depth/pressure measurments. The figure is not rendered
        without calling plt.show().

        Parameters:
            profiles: boolean to overlay indexed profiles [Default=False]

        Returns:
            ax: axis object
            """

        # Get the depth/pressure time-series
        yo = self.get_tz_variables()

        if yo.empty:
            self._logger.warning('Dataset contains no yo time-series')
            return

        # Plot the yo using pd.plot()
        ax = yo.plot(y=self._depth_sensor, marker='o', markersize=1, markerfacecolor='k', markeredgecolor='k',
                     legend=False,
                     linestyle='None')
        # Format the x axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # Plot the indexed profiles, if specified
        if profiles:
            cmap = plt.cm.rainbow(np.linspace(0, 1, self._profiles.shape[0]))
            count = 0
            for i, r in self._profiles.iterrows():
                t = r[['start_time', 'end_time']].values
                z = r[['start_depth', 'end_depth']].values

                ax.plot(t, z, marker='None', linestyle='-', color=cmap[count])

                count += 1

        # Prettify the y axis limits and everse the y-axis
        ax.set_ylim([math.floor(ax.get_ylim()[0]), math.ceil(ax.get_ylim()[1])])
        ax.invert_yaxis()

        locator = mdates.AutoDateLocator(minticks=3, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # Shrink the fontsize
        ax.tick_params(labelsize=10)
        # Center the x-axis tick labels and rotate
        for xlabel in ax.xaxis.get_ticklabels():
            xlabel.set(rotation=0, horizontalalignment='center')

        # Label the y-axis
        plt.ylabel('{:} ({:})'.format(self._depth_sensor,
                                      self._column_defs[self._depth_sensor]['attrs'].get('units', 'unitless')))

        # Title the plot
        ax.set_title('{:} - {:}'.format(self._data_frame.index.min().strftime('%Y-%m-%dT%H:%MZ'),
                                        self._data_frame.index.max().strftime('%Y-%m-%dT%H:%MZ')))

        return ax

    def plot_profiles(self, sensor_name, colormap=plt.cm.rainbow):
        """Plot all indexed profiles for the specified sensor_name.  Depth values taken from self.depth_sensor

        Parameters:
            sensor_name: valid sensor name
            colormap: alternate valid colormap [Default is rainbow]

        Returns:
            ax: axis object
            """
        if sensor_name not in self._data_frame.columns:
            self._logger.error('Invalid sensor name specified: {:}'.format(sensor_name))
            return

        # One color for each profile
        cmap = colormap(np.linspace(0, 1, self._profiles.shape[0]))

        # Create the figure and axis
        fig, ax = plt.subplots()
        # Figure size
        fig.set_size_inches(8.5, 11)
        count = 0
        # Plot each profile
        for i, r in self._profiles.iterrows():
            # Pull out the profile and drop all nan rows
            profile = self._data_frame.loc[r['start_time']:r['end_time'], [self._depth_sensor, sensor_name]].dropna()

            ax.plot(profile[sensor_name], profile[self._depth_sensor], marker='None', color=cmap[count])

            count += 1

        # Prettify the y axis limits and everse the y-axis
        ax.set_ylim([math.floor(ax.get_ylim()[0]), math.ceil(ax.get_ylim()[1])])
        # Reverse the y-axis so that increasing depths go down
        ax.invert_yaxis()

        # Place the x-axis at the top of the plot
        ax.xaxis.tick_top()

        # Label the axes
        ax.set_ylabel('{:} ({:})'.format(self._depth_sensor,
                                         self._column_defs[self._depth_sensor]['attrs'].get('units', 'nodim')))
        ax.set_xlabel('{:} ({:})'.format(sensor_name,
                                         self._column_defs[sensor_name]['attrs'].get('units', 'nodim')))
        ax.xaxis.set_label_position('top')

        # Title the plot
        ax.set_title('{:} - {:}'.format(self._profiles.start_time.min().strftime('%Y-%m-%dT%H:%MZ'),
                                        self._profiles.end_time.max().strftime('%Y-%m-%dT%H:%MZ')))

        return ax

    def plot_profile(self, profile_number, sensor1, sensor2=None, props1=None, props2=None):
        """Plot 1 or 2 variables from the profile identified by profile_number"""
        if sensor1 not in self._data_frame.columns:
            self._logger.error('Invalid sensor1 specified: {:}'.format(sensor1))
            return

        if sensor2 and sensor2 not in self._data_frame.columns:
            self._logger.error('Invalid sensor2 specified: {:}'.format(sensor2))
            return

        marker_props1 = {'marker': '.',
                         'markerfacecolor': 'None',
                         'markeredgecolor': 'k',
                         'color': 'k'}
        marker_props2 = {'marker': '.',
                         'markerfacecolor': 'None',
                         'markeredgecolor': 'r',
                         'color': 'r'}

        if props1:
            if not isinstance(props1, dict):
                self._logger.error('props1 must be a dict of valid Line2D properties')
                return
            marker_props1.update(props1)
        if props2:
            if not isinstance(props2, dict):
                self._logger.error('props2 must be a dict of valid Line2D properties')
                return
            marker_props2.update(props2)

        # Pull out the profile
        profile = self.slice_profile_by_id(profile_number)
        if profile.empty:
            return

        # Data frame containing the plot sensors and dba.depth_sensor
        data1 = profile[[sensor1, self._depth_sensor]].dropna(axis='index', how='any')
        # Get the units
        units1 = self._column_defs[sensor1]['attrs'].get('units', '?')
        data2 = pd.DataFrame()
        if sensor2:
            # Data frame containing the plot sensors and dba.depth_sensor
            data2 = profile[[sensor2, self._depth_sensor]].dropna(axis='index', how='any')

        # Get the depth units
        zunits = self._column_defs[self._depth_sensor]['attrs'].get('units', '?')

        axes1_color = marker_props1['color']
        axes2_color = marker_props2['color']

        # Create the figure and axis
        fig, ax1 = plt.subplots()
        # Figure size
        fig.set_size_inches(8.5, 11)

        # Plot data1 profile
        ax1.plot(data1[sensor1], data1[self._depth_sensor], **marker_props1)

        if not data2.empty:
            # Add a second axis for plotting sensor2
            ax2 = plt.twiny(ax1)
            # Plot data2 profile
            ax2.plot(data2[sensor2], data2[self._depth_sensor], **marker_props2)
            # Share the y-axis between both axes
            ax1.get_shared_y_axes().join(ax1, ax2)
            # color the ticks and tick labels for both axes
            ax1.tick_params(axis='x', colors=axes1_color)
            ax2.tick_params(axis='x', colors=axes2_color)
            # color the x-axis labels
            ax2.xaxis.label.set_color(axes2_color)
            ax1.xaxis.label.set_color(axes1_color)
            # Color the x-axis lines
            ax2.spines['top'].set_edgecolor(axes2_color)
            # Set ax2 bottom x-axis line to None so that the ax1 color shows
            ax2.spines['bottom'].set_edgecolor('None')

            # Get the units
            units2 = self._column_defs[sensor2]['attrs'].get('units', '?')
            # Label the axes
            ax2.set_xlabel('{:} ({:})'.format(sensor2, units2))
        else:
            ax1.xaxis.set_ticks_position('top')
            ax1.xaxis.set_label_position('top')

        # Pretty up the y-limits
        ax1.set_ylim([0, math.ceil(ax1.get_ylim()[1])])
        # Reverse the y-axis direction
        ax1.invert_yaxis()

        # Color the x-axis lines
        ax1.spines['bottom'].set_edgecolor(axes1_color)

        # Label the axes
        ax1.set_xlabel('{:} ({:})'.format(sensor1, units1))
        ax1.set_ylabel('{:} ({:})'.format(self._depth_sensor, zunits))

        # plt.tight_layout()

        profile_time = datetime.datetime.utcfromtimestamp(profile.profile_time.unique()[0].tolist() / 1e9)
        plt.title(profile_time.strftime('%Y-%m-%dT%H:%MZ'), y=1.)

        # return [ax1, ax2]

    def scatter(self, sensor_name, robust=False, colormap=plt.cm.rainbow, ax=None, cmin=None, cmax=None):
        """Colorized scatter plot of the sensor_name time series.  Depth values taken from self.depth_sensor

                Parameters:
                    sensor_name: valid sensor name
                    robust: autoscale the colormap [Default=True]
                    colormap: alternate valid colormap [Default is rainbow]
                    ax: subplot axis to plot in
                    cmin: minimum value for colorbar
                    cmax: maximum value for colorbar

                Returns:
                    ax: axis object

                Wrapper function around glidertools.plot.scatter
                    """
        if sensor_name not in self._data_frame.columns:
            self._logger.error('Invalid sensor name specified: {:}'.format(sensor_name))
            return

        if robust:
            ax = gt.plot.scatter(self._data_frame.index, self._data_frame.depth_raw, self._data_frame[sensor_name],
                                 cmap=colormap,
                                 robust=True)
        else:
            vmin = cmin or self._data_frame[sensor_name].min()
            vmax = cmax or self._data_frame[sensor_name].max()

            ax = gt.plot.scatter(self._data_frame.index,
                                 self._data_frame.depth_raw,
                                 self._data_frame[sensor_name],
                                 cmap=colormap,
                                 vmin=vmin, vmax=vmax)

        # Format the x axis
        locator = mdates.AutoDateLocator(minticks=3, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # Center the x-axis tick labels and rotate
        for xlabel in ax.xaxis.get_ticklabels():
            xlabel.set(rotation=0, horizontalalignment='center')

        cb = ax.get_figure().axes[1]

        cb.set_ylabel('{:}'.format(self._column_defs[sensor_name]['attrs'].get('units', 'nodim')))

        # Title the plot
        ax.set_title('{:}: {:} - {:}'.format(sensor_name,
                                             self._profiles.start_time.min().strftime('%Y-%m-%dT%H:%MZ'),
                                             self._profiles.end_time.max().strftime('%Y-%m-%dT%H:%MZ')))

        return ax

    def pcolormesh(self, sensor_name, xvar='profile_time', mask=pd.Series(), robust=False, colormap=plt.cm.rainbow,
                   ax=None, cmin=None, cmax=None):
        """Plot a section plot of the sensor_name dives.  Depth values taken from self.depth_sensor and xvar must be
        an array of pseudo-discrete of values unique to each individual dive (i.e.: profile_time or profile_id)

                Parameters:
                    sensor_name: valid sensor name
                    xvar: pseudo-discrete values unique to an individual dive
                    robust: autoscale the colormap [Default=True]
                    colormap: alternate valid colormap [Default is rainbow]
                    ax: subplot axis to plot in
                    cmin: minimum value for colorbar
                    cmax: maximum value for colorbar

                Returns:
                    ax: axis object

                Wrapper function around glidertools.plot.pcolormesh
                    """
        if sensor_name not in self._data_frame.columns:
            self._logger.error('Invalid sensor name specified: {:}'.format(sensor_name))
            return
        if xvar not in self._data_frame.columns:
            self._logger.error('Invalid x-axis variable specified: {:}'.format(xvar))
            return

        if mask.empty:
            mask = pd.Series([True for x in range(self._data_frame.shape[0])], index=self._data_frame.index)

        df = self._data_frame.loc[mask, [xvar, self._depth_sensor, sensor_name]].dropna(axis=0, how='any')

        if robust:
            ax = gt.plot.pcolormesh(df[xvar], df[self._depth_sensor], df[sensor_name],
                                    cmap=colormap,
                                    robust=True,
                                    ax=ax)
        else:
            vmin = cmin or df[sensor_name].min()
            vmax = cmax or df[sensor_name].max()

            ax = gt.plot.pcolormesh(df[xvar],
                                    df[self._depth_sensor],
                                    df[sensor_name],
                                    cmap=colormap,
                                    vmin=vmin,
                                    vmax=vmax,
                                    ax=ax)

        # Format the x axis
        locator = mdates.AutoDateLocator(minticks=3, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # Center the x-axis tick labels and rotate
        for xlabel in ax.xaxis.get_ticklabels():
            xlabel.set(rotation=0, horizontalalignment='center')

        # cb = ax.get_figure().axes[1]
        cb = ax.cb.ax

        cb.set_ylabel('{:}'.format(self._column_defs[sensor_name]['attrs'].get('units', 'nodim')))

        # Title the plot
        ax.set_title('{:}: {:} - {:}'.format(sensor_name,
                                             self._profiles.start_time.min().strftime('%Y-%m-%dT%H:%MZ'),
                                             self._profiles.end_time.max().strftime('%Y-%m-%dT%H:%MZ')))

        return ax

    def plot_gps(self, dr=False):
        """Plot the GPS track for the loaded dba files.  Requires processing of the raw GPS sensors as performed by
        self.process_gps().  The following sensors must be contained in the instance:
            m_gps_lat (degrees_north)
            m_gps_lon (degrees_east)
            latitude (degrees_north)
            longitude (degrees_north)

        If the dr kwarg is set to True and m_lat (degrees_north) and m_lon (degrees_east) are found, the dead-reckoned
        positions are also plotted"""

        scale = '10m'
        category = 'cultural'
        shape_file = 'admin_0_countries'

        if not self._process_gps:
            self._logger.error('Raw GPS variables have not been processed')
            return

        gps_sensors = ['m_gps_lat', 'm_gps_lon', 'latitude', 'longitude']

        has_gps = [g for g in gps_sensors if g in self._data_frame.columns]
        if len(has_gps) != len(gps_sensors):
            self._logger.warning('GPS sensor(s) not found')
            return

        lonlat = self._data_frame[['m_gps_lon', 'm_gps_lat']].dropna()
        gps_dt0 = lonlat.index.min()
        gps_dt1 = lonlat.index.max()
        bbox = [lonlat['m_gps_lon'].min(), lonlat['m_gps_lat'].min(), lonlat['m_gps_lon'].max(),
                lonlat['m_gps_lat'].min()]
        lonlat = np.array(lonlat)

        i_lonlat = np.array(self._data_frame[['longitude', 'latitude']].dropna())

        lat_0 = np.array([bbox[1], bbox[3]]).mean()
        lon_0 = np.array([bbox[0], bbox[2]]).mean()
        lat_diff = bbox[3] - bbox[1]
        lon_diff = bbox[2] - bbox[0]

        n = bbox[3]
        s = bbox[1]
        e = bbox[2]
        w = bbox[0]
        if lat_diff > lon_diff:
            offset = lon_0 / abs(lon_0) * lat_diff / 2.0
            w = lon_0 + offset
            e = lon_0 - offset
        else:
            offset = lat_0 / abs(lat_0) * lon_diff / 2.0
            n = lat_0 + offset
            s = lat_0 - offset

        bounds = np.array([w, e, s, n])

        xticker = mticker.MaxNLocator(nbins=5)
        yticker = mticker.MaxNLocator(nbins=5)
        xticks = xticker.tick_values(w, e)
        yticks = yticker.tick_values(s, n)

        # Initialize map
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))
        # plt.subplots_adjust(right=0.85)
        ax.background_patch.set_facecolor('lightblue')
        states = cfeature.NaturalEarthFeature(category=category, scale=scale, facecolor='none', name=shape_file)
        ax.add_feature(states, linewidth=0.5, edgecolor='black', facecolor='tan')
        ax.add_feature(cfeature.RIVERS, zorder=10, facecolor='white')

        ax.coastlines('10m', linewidth=1)
        ax.set_extent(bounds, crs=ccrs.PlateCarree())
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

        # Plot the acquired GPS fixes
        plt.plot(lonlat[:, 0], lonlat[:, 1],
                 color='k',
                 linestyle='None',
                 marker='x',
                 markersize=8,
                 transform=ccrs.PlateCarree())
        plt.plot(lonlat[-1, 0], lonlat[-1, 1],
                 markerfacecolor='r',
                 markeredgecolor='None',
                 markersize=15,
                 marker='o',
                 transform=ccrs.PlateCarree())
        plt.plot(lonlat[0, 0], lonlat[0, 1],
                 markerfacecolor='g',
                 markeredgecolor='None',
                 markersize=15,
                 marker='o',
                 transform=ccrs.PlateCarree())
        # Plot the interpolated gps fixes
        plt.plot(i_lonlat[:, 0], i_lonlat[:, 1],
                 color='k',
                 linestyle='solid',
                 marker='None',
                 linewidth=2.,
                 transform=ccrs.PlateCarree())

        if dr:
            dr_sensors = ['m_lon', 'm_lat']
            has_dr = [g for g in dr_sensors if g in self._data_frame.columns]
            if len(has_dr) != len(dr_sensors):
                self._logger.warning('Dead-reckoned GPS sensor(s) not found')
            else:
                dr_lonlat = np.array(self._data_frame[['m_lon', 'm_lat']].dropna())
                # Plot the dead-reckoned gps fixes
                plt.plot(dr_lonlat[:, 0], dr_lonlat[:, 1],
                         color='r',
                         linestyle='dotted',
                         marker='None',
                         linewidth=1.,
                         transform=ccrs.PlateCarree())

        xlabels = ax.get_xticklabels()
        ylabels = ax.get_yticklabels()
        all_labels = xlabels + ylabels
        _ = [tick.set_fontsize(10) for tick in all_labels]

        # Title the plot
        ax.set_title('Glider Track: {:} - {:}'.format(
            gps_dt0.strftime('%Y-%m-%dT%H:%MZ'),
            gps_dt1.strftime('%Y-%m-%dT%H:%MZ')))

    def slice_profile_by_id(self, profile_number):
        """Return a data frame containing only data from the profile identified by (0-based) profile_number"""
        if profile_number < 0 or profile_number > self._profiles.shape[0] - 1:
            self._logger.error('Invalid profile id specified: {:}'.format(profile_number))

        return self._data_frame.loc[self._data_frame.profile_id == profile_number]

    def slice_segment(self, segment):
        """Return a new instance of the Dba class containing only the data from the specified segment"""
        dba_info = self._dba_files.loc[self._dba_files['segment_filename_0'] == segment]
        if dba_info.empty:
            self._logger.error('Segment {:} not found'.format(segment))
            return

        dba_file = os.path.join(dba_info['path'].values[0], dba_info['file'].values[0])
        segment_df = Dba(dba_file, gps=self._process_gps, ctd=self._process_ctd, profiles=self._index_profiles)

        segment_df.depth_sensor = self._depth_sensor

        return segment_df

    def to_xarray(self, profile_index_type=None):
        """Convert the pandas Dataframe (self._data) to an xarray Dataset, attach the attributes from self.column_defs
        as variable attributes and set default encodings for writing NetCDF files"""

        if profile_index_type:
            if profile_index_type not in self._profile_index_types:
                self._logger.error('Invalid xarray DataSet profile index specified: {:}'.format(profile_index_type))
                return
            profiles = []
            p_nums = self._data_frame[profile_index_type].unique()
            for p_num in p_nums:
                if pd.isna(p_num):
                    continue
                profiles.append(self._data_frame[self._data_frame[profile_index_type] == p_num].reset_index())

            if not profiles:
                self._logger.warning('No profiles to export as an xarray dataset')
                return

            ds = pd.concat(profiles)
            ds.set_index(['time', profile_index_type], inplace=True)
            ds = ds.to_xarray()

        else:
            ds = self._data_frame.to_xarray()

        for column_def in self._column_defs:

            if column_def not in ds:
                continue

            # Update attributes
            ds[column_def] = ds[column_def].assign_attrs(**self._column_defs[column_def]['attrs'])

            # Default variable encoding
            encoding = default_encoding.copy()

            # Special encoding case for datetime64 dtypes
            if ds[column_def].dtype.name == 'datetime64[ns]':
                encoding['units'] = ds[column_def].attrs.get('units', 'seconds since 1970-01-01T00:00:00Z')
                ds[column_def].attrs.pop('units', None)
                ds[column_def].attrs.pop('calendar', None)

            # Special encoding case for strings
            if ds[column_def].dtype.name == 'object':
                encoding['dtype'] = 'str'
            else:
                encoding['dtype'] = ds[column_def].attrs.get('dtype', 'f8')

            # Drop dtype attribute if exists
            ds[column_def].attrs.pop('dtype', None)

            # Set the encoding
            ds[column_def].encoding = encoding

        return ds

    # def calculate_first_diffs(self, sensor_name):
    #
    #     if sensor_name not in self._column_defs:
    #         self._logger.error('Invalid sensor_name specified: {:}'.format(sensor_name))
    #         return
    #
    #     first_diffs = calculate_series_first_differences(self._data_frame[sensor_name])
    #
    #     new_sensor_name = '{:}_first_diffs'.format(sensor_name)
    #
    #     self._data_frame[new_sensor_name] = first_diffs
    #
    #     self._column_defs[new_sensor_name] = deepcopy(self._column_defs[sensor_name])
    #     self._column_defs[new_sensor_name]['nc_var_name'] = new_sensor_name
    #     self._column_defs[new_sensor_name]['attrs']['comment'] = 'First differences of {:}'.format(sensor_name)

    def _load_dbas_to_data_frame(self, dba_files):

        dbas = []
        data_frames = []
        for dba_file in dba_files:

            if not os.path.isfile(dba_file):
                self._logger.warning('Skipping Invalid DBA file {:}'.format(dba_file))
                continue

            dbas.append(dba_file)

            df = self._dba_to_pd_data_frame(dba_file, keep_gld_dups=self._keep_gld_dups)

            data_frames.append(df)

        # Aggregate the individual dba data frames in to a single data frame
        data_frame = pd.concat(data_frames)

        # Create and store the data frame containing the successfully parsed dba files
        self._dba_files = build_dbas_data_frame(dbas)

        # Convert all sensors with units == 'bar' to decibars and update self._dba_sensor_metadata
        bar_sensors = [s['native_sensor_name'] for s in self._dba_sensor_metadata if s['units'] == 'bar']
        native_sensor_names = [s['native_sensor_name'] for s in self._dba_sensor_metadata]
        for bar_sensor in bar_sensors:
            data_frame[bar_sensor] = data_frame[bar_sensor] * 10
            si = native_sensor_names.index(bar_sensor)
            # Update the units
            self._dba_sensor_metadata[si]['units'] = 'decibar'

        # Remove some default bad values with the following dba units:
        #   timestamp
        #   lat
        #   lon
        replace_rules = {t: 0 for t in self._time_vars}
        gps_sensors = [s['native_sensor_name'] for s in self._dba_sensor_metadata if
                       s['units'] == 'lat' or s['units'] == 'lon']
        for gps_sensor in gps_sensors:
            replace_rules[gps_sensor] = 69696969

        data_frame.replace(replace_rules, np.nan, inplace=True)

        self._data_frame = data_frame.sort_index()

    def _dba_to_pd_data_frame(self, dba_file, keep_gld_dups):
        """Parse a Slocum DBA file and return the data as a pandas DataFrame. 2 columns are added to the data frame:
        time: m_present_time converted to datetime
        sci_time: sci_m_present_time converted to datetime, if it exists.  If not all values are pd.NaT"""

        if not os.path.isfile(dba_file):
            self._logger.error('Invalid DBA file specified: {:}'.format(dba_file))
            return

        self._logger.info('Loading dba: {:}'.format(os.path.basename(dba_file)))

        # Parse the dba header
        dba_headers = parse_dba_header(dba_file)

        # Parse the dba sensor metadata
        sensors = parse_dba_sensor_defs(dba_file)
        sensor_names = [s['native_sensor_name'] for s in sensors]

        all_sensor_names = [s['native_sensor_name'] for s in self._dba_sensor_metadata]
        new_sensors = [s for s in sensor_names if s not in all_sensor_names]
        for new_sensor in new_sensors:
            self._dba_sensor_metadata.append(sensors[sensor_names.index(new_sensor)])

        # Calculate the number of header lines to skip
        num_header_lines = int(dba_headers['num_ascii_tags']) + int(dba_headers['num_label_lines'])

        # Array of native slocum sensors with units=timestamp.  These should be parsed as
        # unix timestamps
        timestamps = [s['native_sensor_name'] for s in sensors if s['units'] == 'timestamp']

        if not self._keep_gld_dups:
            timestamps = [t for t in timestamps if not t.startswith('gld_dup_')]

        all_timestamps = list(self._time_vars) + timestamps
        self._time_vars = set(all_timestamps)

        # sensor_dtypes = {s: 'f8' for s in sensor_names}

        # Read the dba data table into a DataFrame
        self._logger.debug('Parsing {:} to DataFrame'.format(dba_file))
        df = pd.read_table(dba_file,
                           # dtype=sensor_dtypes,
                           delim_whitespace=True,
                           names=sensor_names,
                           error_bad_lines=False,
                           warn_bad_lines=True,
                           header=None,
                           parse_dates=timestamps,
                           date_parser=self.epoch2datetime,
                           skiprows=num_header_lines)

        dba_headers['num_rows'] = df.shape[0]
        # Store the headers using the filename_label as the key
        file_parts = dba_headers['filename_label'].split('(')
        self._dba_headers[file_parts[0]] = dba_headers

        # Drop gld_dup_* sensors if keep_gld_dups=False
        if not keep_gld_dups:
            gld_dups = [s for s in df.keys() if s.startswith('gld_dup')]
            if gld_dups:
                self._logger.debug('Dropping {:} gld_dup sensors from {:}'.format(len(gld_dups), dba_file))
                df = df.drop(gld_dups, axis=1)

            # Remove gld_dup sensors from self._dba_sensor_metadata
            self._dba_sensor_metadata = [s for s in self._dba_sensor_metadata if
                                         not s['native_sensor_name'].startswith('gld_dup')]

        # Add the segment and the8x3_filename columns to the data frame
        df['segment'] = dba_headers['segment_filename_0']
        df['the8x3_filename'] = file_parts[1][:-1]

        # Add default attributes to self._column_defs
        self._column_defs['segment'] = {'attrs': self._default_attributes.get('segment', {})}
        self._column_defs['the8x3_filename'] = {'attrs': self._default_attributes.get('the8x3_filename', {})}

        # Set the timestamp index to m_present_time
        df.index = df['m_present_time']

        # Drop rows with duplicate index timestamps
        df = df[~df.index.duplicated(keep='first')]

        return df

    def _build_default_column_defs(self):

        self._logger.info('Building default column metadata records')

        for dba_def in self._dba_sensor_metadata:

            column_def = {'nc_var_name': dba_def['native_sensor_name'],
                          'type': dba_def['dtype'],
                          'attrs': {s: dba_def[s] for s in dba_def}}

            if dba_def['native_sensor_name'] in self.time_vars:
                column_def['attrs'].update(self._default_attributes.get('time', {}))

            column_def['attrs']['long_name'] = dba_def['native_sensor_name']

            self._column_defs[dba_def['native_sensor_name']] = column_def

    def _load_default_attributes(self):

        attributes_path = os.path.join(self._metadata_path, 'attributes')
        if not os.path.isdir(attributes_path):
            self._logger.warning('Invalid default attributes YAML location: {:}'.format(attributes_path))
            return

        yaml_files = glob.glob(os.path.join(attributes_path, '*.yml'))
        for yaml_file in yaml_files:
            filename = os.path.basename(yaml_file)
            variable_name = filename.split('.')[0]
            try:
                with open(yaml_file, 'r') as fid:
                    self._default_attributes[variable_name] = yaml.load(fid)
            except ValueError as e:
                self._logger.error('Error loading variable attribute file {:}: {:}'.format(filename, e))
                continue

    @staticmethod
    def epoch2datetime(t):
        return pd.to_datetime(t, unit='s', errors='coerce')

    @staticmethod
    def datetime2epoch(dtime):
        return (dtime - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

    def __repr__(self):
        return '<Dba(num_dbas={:}, profiles={:}, gps={:}, ctd={:}, keep_gld_dups={:})>'.format(len(self._dba_files),
                                                                                               len(self._profiles),
                                                                                               self._process_gps,
                                                                                               self._process_ctd,
                                                                                               self._keep_gld_dups)
