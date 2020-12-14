import sys
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import math


if Path.cwd().name == '':
	from tabular.utils.reduce_mem_usage import reduce_mem_usage
elif Path.cwd().name == 'utils':
	sys.path.append("../")
	from utils.reduce_mem_usage import reduce_mem_usage


def read_data(data_dir, data_type="train"):
	if data_type == 'train':
		# train dataset
		dtype = {}
		if len(dtype) > 0:
			df = pd.read_csv(data_dir + f"{data_type}.csv", dtype=dtype)
		else:
			df = pd.read_csv(data_dir + f"{data_type}.csv")
		# 以下に読み込んだ後の処理を追加していく
	else:
		# test dataset
		dtype = {}
		if len(dtype) > 0:
			df = pd.read_csv(data_dir + f"{data_type}.csv", dtype=dtype)
		else:
			df = pd.read_csv(data_dir + f"{data_type}.csv")
		# 以下に読み込んだ後の処理を追加していく
	return df


def read_preprocessing_data(data_dir, data_type="train", write_mode=False, pickle=False):
	if write_mode is True:
		data = read_data(data_dir, data_type)
		print("finish read data.")
		if pickle is True:
			data.to_pickle(data_dir + f"{data_type}_processed.pkl")
		else:
			data.to_csv(data_dir + f"{data_type}_processed.csv", header=True, index=False)
	else:
		if pickle is True:
			data = pd.read_pickle(data_dir + f"{data_type}_processed.pkl")
		else:
			data = pd.read_csv(data_dir + f"{data_type}_processed.csv")
	return data


if __name__ == "__main__":
	data_dir = ""
	train = read_preprocessing_data(data_dir, "train", write_mode=True, pickle=True)
	test = read_preprocessing_data(data_dir, "test", write_mode=True, pickle=True)
