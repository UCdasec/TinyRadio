# import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Concatenate
import sys
sys.path.append("/home/erc/PycharmProjects/rf/TF")
from complexnn import utils


class Modrelu(Layer):

	def __init__(self, **kwargs):
		super(Modrelu, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self._b = self.add_weight(name='b', 
									shape=(input_shape[-1]//2,),
									initializer='zeros',
									trainable=True)
		super(Modrelu, self).build(input_shape)  # Be sure to call this at the end
		# self.built = True

	def call(self, x):
		real = utils.GetReal()(x)
		imag = utils.GetImag()(x)

		abs1 = K.relu(utils.GetAbs()(x))
		abs2 = K.relu(utils.GetAbs()(x) - self._b)


		real = real * abs2 / (abs1+0.0000001)
		imag = imag * abs2 / (abs1+0.0000001)

		merged = Concatenate()([real, imag])
		# print("!!!!!")
		# print(merged.shape)
		return merged

	def compute_output_shape(self, input_shape):
		return input_shape