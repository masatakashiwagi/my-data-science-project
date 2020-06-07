import numpy as np


def convert_pred_to_one_hot(pred, num_class=11, max_class=10):
	"""
	Convert the prediction into probabilities
	Example: 1.6 --> [0, 0.4, 0.6, 0, 0, ...]
	1.6 is closer to 2 than to 1
	All rows will sum 1

	単一の予測値をマルチクラスの予測値に分解できる
	"""

	# clip results lower or higher than limits
	pred = np.clip(pred, 0, max_class)

	# convert to "one-hot"
	pred = 1 - np.abs(pred.reshape((-1, 1)) - np.arange(num_class))

	# clip results lower than 0
	pred = np.clip(pred, 0, 1)

	return pred
