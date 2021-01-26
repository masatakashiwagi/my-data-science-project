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
		output_name = os.path.splitext(path)
		for key in keys:
			dframe = df[key]
			output_file = output_name[0] + '.csv'
			dframe.to_csv(output_file, encoding=encoding, index=False)
	else:
		raise ValueError(f"{path} is not exist.")
