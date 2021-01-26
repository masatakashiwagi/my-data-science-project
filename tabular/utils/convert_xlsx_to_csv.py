import os
import pandas as pd
import pathlib


def convert_xlsx_to_csv(filename: str, encoding: str = 'Shift_JIS') -> None:
	"""
	Convert xlsx file to csv file.
	"""
	path = pathlib.Path(filename)
	if os.path.isfile(path):
		df = pd.read_excel(path, sheet_name=None)
		keys = list(df.keys())
		dir_path = os.path.dirname(path)
		for key in keys:
			dframe = df[key]
			output_name = os.path.join(dir_path, key) + '.csv'
			dframe.to_csv(output_name, encoding=encoding, index=False)
	else:
		ValueError(f"{path} is not exist.")
