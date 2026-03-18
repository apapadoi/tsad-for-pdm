# Copyright 2026 Anastasios Papadopoulos, Apostolos Giannoulidis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fastf1
import pandas as pd
import random

years = [i for i in range(2018, 2026)]
COLUMNS_TO_KEEP_FROM_MERGED_DF = ['Date', 'Speed', 'RPM', 'nGear', 'Throttle', 'Brake', 'DRS', 'X', 'Y', 'Z', 'Status_OffTrack', 'Status_OnTrack', 'DistanceToDriverAhead', 'Distance', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed']
USE_UTC = True
OUTPUT_FOLDER = '../../DataFolder/formula1/out'

for year in years:
    total = 0
    current_event_schedule = fastf1.get_event_schedule(year)
    
    for current_event_index, current_event_df_row in current_event_schedule.iterrows():
        current_event_object = fastf1.get_event(year, current_event_df_row['OfficialEventName'])
        print(current_event_object)

        if current_event_object.is_testing():
            continue
        
        for session_identifier in ['S', 'SS', 'SQ', 'R']: #['FP1', 'FP2', 'FP3', 'Q', 'S', 'SS', 'SQ', 'R']:
            try:
                current_session = current_event_object.get_session(session_identifier)

                current_session.load()
            
                drivers_who_retired_df = current_session.results[current_session.results['ClassifiedPosition'] == 'R']

                for _, current_driver_who_retired_row in drivers_who_retired_df.iterrows():
                    current_driver_who_retired_laps = current_session.laps[current_session.laps['DriverNumber'] == current_driver_who_retired_row['DriverNumber']]
                    
                    current_telemetry_df = current_driver_who_retired_laps.telemetry.copy()

                    current_weather_data = current_driver_who_retired_laps.get_weather_data()

                    current_telemetry_df['Status'] = current_telemetry_df['Status'].astype('category').cat.set_categories(['OffTrack', 'OnTrack'])

                    current_telemetry_df = pd.get_dummies(current_telemetry_df, columns=['Status'], prefix='Status')

                    current_telemetry_df = current_telemetry_df[current_telemetry_df['RPM'] > 0]

                    merged_df = pd.merge_asof(
                                    left=current_telemetry_df,
                                    right=current_weather_data,
                                    on='Time',
                                    direction='nearest'
                                )
                    
                    merged_df = merged_df[COLUMNS_TO_KEEP_FROM_MERGED_DF]

                    merged_df.to_csv(
                        f'{OUTPUT_FOLDER}/{year}/telemetry_{current_event_df_row["OfficialEventName"].replace(" ", "_")}_{current_session.name}_{current_driver_who_retired_row["FullName"].replace(" ", "_")}_{random.randint(1, 1000)}_{current_driver_who_retired_row["Status"].replace(" ", "_")}.csv',
                        index=False
                    )

                    total += 1

                    # break
            except Exception as e:
                print(year, current_event_df_row['OfficialEventName'], session_identifier)
                print(f'Exception {e}')

        # break

    print(f'{total} for year {year}')
    # break