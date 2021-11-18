import datetime
import glob
import json
import os
from os.path import split, splitext, join
from typing import Dict, Union, List

import numpy as np
import pandas as pandas

config = None


def get_config() -> dict:
    global config
    if config:
        return config
    with open("./config.json") as fid:
        config = json.load(fid)
    return config


def fix_category_alias(category_name: str) -> str:
    if any([category_name in cat for cat in get_config()['waves'].values()]):
        return category_name
    try:
        return [cat_name for cat_name, category_aliases in get_config()["category_aliases"].items() if
                category_name in category_aliases][0]
    except Exception as ex:
        raise ValueError(f"Invalid/unknown category name \"{category_name}\"", ex)


def get_place_points(place: Union[int, str], multiplier=1) -> int:
    """
    Points for a given place in the wave.
    :param place: place in wave, None for DNF
    :return: placing. DNF=4
    """
    # Handle DNF
    if place is None or place == 'DNF':
        return 4
    if place in ['DSQ', 'DQ']:
        return 0
    try:
        place = int(place)
        # Score table
        score_table = [120, 112, 106, 102, 100, 99]
        custom_score_length = len(score_table)
        base_score = score_table[min(place, custom_score_length) - 1] - max(0, place - custom_score_length)
        if base_score < 4:
            raise Exception("Invalid score!")
        return base_score * multiplier
    except ValueError as ve:
        raise ValueError(f"Unknown place={place}", ve)


def get_category_place_points(place: Union[int, str], category: str) -> int:
    try:
        wave_number = \
            [wave_num for wave_num, wave_categories in get_config()['waves'].items() if category in wave_categories][0]
        multiplier = int(get_config()['wave_multiplier'][wave_number])
        return get_place_points(place, multiplier)
    except:
        raise ValueError(f"Invalid place={place} or category={category}")


def get_race_data(race_pages: Dict[str, pandas.DataFrame]) -> pandas.DataFrame:
    # Stack the results
    if len(race_pages) > 1:
        race_data = pandas.concat(race_pages.values())
    else:
        race_data = list(race_pages.values())[0]
    # Strip column names
    race_data = race_data.rename(columns=lambda x: x.strip())
    # Wipe out the index to prevent the "disappearing cody sowers" bug of duplicate indexes.
    race_data = race_data.reset_index()
    # Apply complete data corrections (aliases, null removal, etc)
    race_data = fix_column_aliases(race_data)
    race_data = remove_null_data(race_data)
    race_data = assign_rider_name(race_data)
    return race_data


def remove_drop_waves(race_data: pandas.DataFrame) -> pandas.DataFrame:
    race_data['Category'] = race_data['Category'].apply(
        lambda cat: "DROP" if cat in get_config()['drop_waves'] else cat)
    return race_data.drop(race_data.index[(race_data['Category'] == 'DROP')])


def fix_column_aliases(race_data: pandas.DataFrame) -> pandas.DataFrame:
    for col_name, col_aliases in get_config()['column_aliases'].items():
        race_data = race_data.rename(columns=dict([(col_alias, col_name) for col_alias in col_aliases]))
    return race_data


def get_wave_data(race_data: pandas.DataFrame) -> Dict[str, pandas.DataFrame]:
    current_wave_results = dict()

    race_data = remove_drop_waves(race_data)
    race_data = set_bib_license_integer(race_data)
    race_data = sort_by_laps_and_time(race_data)
    race_data = fix_place_scoring(race_data)
    race_data['Category'] = race_data['Category'].apply(lambda x: fix_category_alias(x))
    race_data['Place Points'] = race_data[['Rider Place', 'Category']].apply(
        lambda x: get_category_place_points(*x), axis=1)
    # Reorder
    cols = race_data.columns.tolist()
    cols = cols[:2] + cols[-1:] + cols[2:-1]
    race_data = race_data[cols]
    # Split into waves
    for (wave_number, categories_in_wave) in get_config()['waves'].items():
        current_wave_results[wave_number] = race_data[race_data['Category'].isin(categories_in_wave)]

    return current_wave_results


def fix_place_scoring(race_data: pandas.DataFrame) -> pandas.DataFrame:
    for (wave_number, categories_in_wave) in get_config()['waves'].items():
        is_wave = np.bitwise_and(race_data['Category'].isin(categories_in_wave), race_data['Rider Place'] != 'DNF')
        race_data.loc[is_wave, 'Rider Place'] = np.r_[1:is_wave.sum() + 1]
    return race_data


def elapsed_seconds(time_str: str) -> float:
    if isinstance(time_str, float):
        return time_str
    elif isinstance(time_str, str):
        time_toks = time_str.split(':')
        return float(time_toks[-1]) + 60 * float(time_toks[-2]) + 3600 * float(
            time_toks[-3] if len(time_toks) == 3 else 0)
    elif isinstance(time_str, datetime.time):
        return 3600.0 * time_str.hour + 60.0 * time_str.minute + time_str.second + time_str.microsecond / 1.0E6
    else:
        raise TypeError(f'time_str={time_str} is an unknown type={type(time_str)}')


def sort_by_laps_and_time(race_data: pandas.DataFrame) -> pandas.DataFrame:
    col_names = race_data.columns.tolist()
    sort_cols = ['Laps', 'Time']
    if all([sort_col in col_names for sort_col in sort_cols]):
        # Create time index
        race_data['Elapsed Seconds'] = race_data['Time'].apply(elapsed_seconds)
        race_data = race_data.sort_values(by=['Laps', 'Elapsed Seconds'], ascending=[False, True])
        race_data = race_data.drop(columns=['Elapsed Seconds'])
    return race_data


def assign_rider_name(race_data: pandas.DataFrame) -> pandas.DataFrame:
    col_names = race_data.columns.tolist()
    if 'Rider Name' not in col_names:
        race_data['Rider Name'] = race_data['Rider First Name'] + ' ' + race_data['Rider Last Name']

    # Camel case
    race_data['Rider Name'] = race_data['Rider Name'].apply(lambda x: x.title().replace('.', ''))

    if 'Rider First Name' not in col_names:
        race_data['Rider First Name'] = race_data['Rider Name'].apply(lambda x: x.split(' ')[0])
        race_data['Rider Last Name'] = race_data['Rider Name'].apply(lambda x: " ".join(x.split(' ')[1:]))
    return race_data


def remove_null_data(race_data):
    # Remove the blank lines
    race_data = race_data.drop(race_data.index[race_data['Rider Place'].isnull()])
    race_data = race_data.drop(race_data.index[race_data['Rider Place'] == 0])
    # Remove the DNS data
    race_data = race_data.drop(race_data.index[(race_data['Rider Place'].isin(['DNS', 'DQ', 'DSQ']))])
    # Remove unnamed riders
    race_data = race_data.drop(race_data.index[race_data['Category'] == 'NONE'])
    return race_data


def set_bib_license_integer(race_data):
    # Set bib # and license # to integers
    col_names = race_data.columns.tolist()
    if 'Rider License #' in col_names:
        race_data['Rider License #'] = np.where(
            np.bitwise_or(race_data['Rider License #'].isin(get_config()['drop_licenses']),
                          race_data['Rider License #'].isnull()),
            0, race_data['Rider License #'])
        race_data = race_data.astype({'Rider License #': 'int32'})

    if 'Bib #' in col_names:
        race_data['Bib #'] = np.where(
            np.bitwise_or(race_data['Bib #'] == 'nan', race_data['Bib #'].isnull()),
            0, race_data['Bib #'])
        race_data = race_data.astype({'Bib #': 'int32'})
    return race_data


def alternate_row_colors(result):
    return np.tile(np.where(np.r_[1:result.shape[0] + 1] % 2 == 1, 'background-color: Khaki',
                            'background-color: LightGoldenRodYellow'), (result.shape[1], 1)).transpose()


def load_races(data_folder: str) -> Dict[str, Dict[str, pandas.DataFrame]]:
    race_wave_results: Dict[str, Dict[str, pandas.DataFrame]] = dict()
    for race_file in glob.glob(data_folder):
        # Parse the race name
        race_name = get_race_name(race_file)

        race_data: Dict[str, pandas.DataFrame] = pandas.read_excel(race_file, sheet_name=None)
        wave_data: Dict[str, pandas.DataFrame] = get_wave_data(get_race_data(race_data))

        write_ovcx_html(race_name, wave_data)
        write_cross_results_csv(race_name, wave_data)

        race_wave_results[race_name] = wave_data

    return race_wave_results


def get_race_category(category: str) -> str:
    for wave_cats in get_config()['waves'].values():
        if category in wave_cats:
            return " | ".join(wave_cats)
    raise ValueError(f"Unknown Category={category}")


def write_cross_results_csv(race_name, wave_data):
    # Consolidate the data into one grid.
    race_data = pandas.concat(list(wave_data.values()))
    # Insert USAC data
    race_data['Race Date'] = get_config()['race_dates'][race_name]
    race_data['Race Discipline'] = 'CX'
    race_data['Race Category'] = race_data['Category'].apply(lambda x: get_race_category(x))
    race_data['Race Gender'] = race_data['Race Category'].map(lambda x: 'Men' if 'Men' in x else 'Women')
    race_data['Race Class'] = ''
    race_data['Age'] = '9-99'
    race_data['Time'] = ''

    race_data = race_data.rename(columns={
        'Rider First Name': 'First name',
        'Rider Last Name': 'Last name',
        'Rider Team Name': 'Team'
    })
    csv_output_file = f'{get_config()["output_data"]}/{race_name}.csv'
    race_data.to_csv(path_or_buf=csv_output_file, index=False, columns=[
        'Race Date', 'Race Discipline', 'Race Category', 'Race Gender', 'Race Class', 'Age', 'Rider Place',
        'Rider License #', 'First name', 'Last name', 'Time', 'Team'
    ])


def write_ovcx_html(race_name: str, wave_data: Dict[str, pandas.DataFrame]):
    with open(f'{get_config()["output_data"]}/{race_name}.html', 'w') as fid:
        fid.writelines(['<HTML>', f'<h1>{race_name}</h1>'])
        # Save HTML
        for (wave_number, wave_results) in wave_data.items():
            try:
                fid.write(f'<h3>Wave #{wave_number} - {", ".join(get_config()["waves"][wave_number])}</h3>\n')
                styled_results = wave_results.reset_index(drop=True).style
                styled_results.hide_index()
                try:
                    styled_results.hide_columns(['index', 'Bib #', 'Rider First Name', 'Rider Last Name', 'Gap'])
                except KeyError:
                    # Ignore columns that don't exist
                    pass
                styled_results.apply(alternate_row_colors, axis=None)
                fid.write(styled_results.to_html())
            except KeyError as ke:
                raise KeyError(f"race name={race_name}  wave #={wave_number}", ke)


def write_ovcx_overall_html(overall_category_results: Dict[str, pandas.DataFrame]):
    with open(f'{get_config()["output_data"]}/OVCX_Overall.html', 'w') as fid:
        fid.writelines(['<HTML>', f'<h1>OVCX Series Standings</h1>'])
        # Save HTML
        for (category_name, category_results) in overall_category_results.items():
            fid.write(f'<h3>{category_name}</h3>\n')
            category_results.index.name = None
            styled_results = category_results.style
            styled_results.apply(alternate_row_colors, axis=None)
            fid.write(styled_results.to_html())


def get_overall_column(wave_data: pandas.DataFrame, race_name: str, category_name: str) -> pandas.DataFrame:
    is_category_data = wave_data['Category'] == category_name
    overall_wave_view = wave_data[['Rider Name', 'Place Points']][is_category_data]

    overall_wave_view = overall_wave_view.rename(columns={'Place Points': race_name})
    overall_wave_view = overall_wave_view.set_index('Rider Name')
    try:
        if category_name in get_config()['UCI_wave_drops'][race_name]:
            overall_wave_view[race_name] = 0
    except KeyError:
        pass
    return overall_wave_view


def get_participation_column(race_name: str, rider_names: pandas.Series) -> pandas.DataFrame:
    participation_frame = pandas.DataFrame(rider_names, columns=['Rider Name'])
    participation_frame[race_name] = 1
    participation_frame = participation_frame.set_index('Rider Name')
    return participation_frame


def count_not_zero(s: pandas.Series) -> int:
    n_na = np.count_nonzero(np.isnan(s))
    return np.count_nonzero(s) - n_na


def main() -> None:
    process_race_results()
    process_raffle_results()
    process_callups()


def process_raffle_results():
    race_overall_participation: Dict[str, List[str]] = dict()
    for race_file in glob.glob(get_config()['race_data']):
        # Parse the race name
        race_name = get_race_name(race_file)

        race_data: Dict[str, pandas.DataFrame] = pandas.read_excel(race_file, sheet_name=None)
        complete_race_data = get_race_data(race_data)
        race_count = complete_race_data.groupby(['Rider Name']).count()
        # Add a point for each name - lower case only for ease
        race_overall_participation[race_name] = complete_race_data['Rider Name'].unique().transpose()

    category_overall = pandas.concat([get_participation_column(race_name, rider_names)
                                      for (race_name, rider_names) in race_overall_participation.items()], axis=1)
    category_overall['# Races'] = category_overall.sum(axis=1, skipna=True, numeric_only=True)
    category_overall.to_csv(f'{get_config()["output_data"]}/OVCX_Raffle_Races.csv')


def process_callups():
    # Process callups
    callup_folder = get_config()['callups_data']
    for callup_file in glob.glob(join(callup_folder, '*.csv')):
        race_name = get_race_name(callup_file)
        race_date = get_config()['race_dates'][race_name]
        race_date = datetime.datetime.strptime(race_date, '%Y-%m-%d')
        callup_data = pandas.read_csv(callup_file)
        # Apply the aliases
        callup_data = callup_data.rename(columns={'Category Entered / Merchandise Ordered': 'Category'})
        callup_data = remove_drop_waves(callup_data)
        callup_data['Category'] = callup_data['Category'].apply(fix_category_alias)
        callup_data['Entry Date and Time'] = callup_data['Entry Date and Time'].apply(
            lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p'))
        callup_data['2-week bump'] = callup_data['Entry Date and Time'].apply(lambda x: (race_date - x).days >= 14)
        # Section it by wave
        for (wave_name, wave_categories) in get_config()['waves'].items():
            wave_callups = callup_data[callup_data['Category'].isin(wave_categories)]
            wave_callups = wave_callups.sort_values(axis=0, by=['2-week bump', 'CrossResults.com Points'],
                                                    ascending=[False, True])
            try:
                os.mkdir(join(callup_folder, race_name))
            except FileExistsError:
                pass
            wave_callups.to_csv(join(callup_folder, race_name, f"Wave {wave_name}.csv", ), index=False)


def get_race_name(race_file: str) -> str:
    folder, filename = split(race_file)
    race_name, file_ext = splitext(filename)
    return race_name


def process_race_results():
    # Open excel race files
    race_wave_results = load_races(get_config()['race_data'])
    # Combine horizontally by name, not license number (since licenses are inconsistently available)
    overall_category_results: Dict[str, pandas.DataFrame] = dict()
    for (wave_name, wave_categories) in get_config()['waves'].items():
        for category_name in wave_categories:
            process_overall_category(category_name, overall_category_results, race_wave_results, wave_name)
    write_ovcx_overall_html(overall_category_results)


def process_overall_category(category_name, overall_category_results, race_wave_results, wave_name):
    category_overall = pandas.concat([get_overall_column(race_data[wave_name], race_name, category_name)
                                      for (race_name, race_data) in race_wave_results.items()], axis=1)
    n_races = int(get_config()['best_of_count']) if category_name not in get_config()[
        'UCI_best_of_waves'] else int(get_config()['UCI_best_of_count'])
    best_races_data = category_overall.apply(pandas.Series.nlargest, axis=1, n=n_races)
    category_overall['Series Total'] = best_races_data.sum(axis=1, skipna=True, numeric_only=True)
    category_overall['# Races'] = best_races_data.apply(count_not_zero, axis=1)
    category_overall = category_overall.sort_values(by=['Series Total', 'Rider Name'],
                                                    ascending=[False, True]).fillna(0).astype('int64')
    # Put points and count first after name
    cols = category_overall.columns.tolist()
    cols = cols[-2:] + cols[0:-2]
    category_overall = category_overall[cols]
    overall_category_results[category_name] = category_overall


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
