import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class maxpool(tf.keras.Model):
    def __init__(self,downrate):
        super().__init__()
        self.pooling = tf.keras.layers.MaxPool2D(
            pool_size=(downrate,downrate)
        )
    
    def call(self,inputs):
        output = self.pooling(inputs)
        return output

class avgpool(tf.keras.Model):
    def __init__(self,downrate):
        super().__init__()
        self.pooling = tf.keras.layers.AveragePooling2D(
            pool_size=(downrate,downrate),
        )
    
    def call(self,inputs):
        output = self.pooling(inputs)
        return output


image_temp = tf.io.read_file('./pooling/pearl.jpg')
# paint.show()
paint = tf.image.decode_jpeg(image_temp)
paint_rot = tfa.image.rotate(paint,np.pi/16)
paint = np.array(paint)
paint_rot = np.array(paint_rot)
print(paint_rot.shape)
paint = np.expand_dims(paint,axis=0)
paint_rot = np.expand_dims(paint_rot,axis=0)
downrate = 40

paint_downsamp = paint_rot[:,::downrate,::downrate,:]
paint_downsamp = tfa.image.rotate(paint_downsamp,-np.pi/16)
paint_downsamp = np.squeeze(paint_downsamp)
print(paint_downsamp.shape)
plt.imshow(paint_downsamp)

# maxpooling = maxpool(downrate)
# paint_maxpool = maxpooling(paint_rot)
# paint_maxpool = np.squeeze(paint_maxpool)
# plt.imshow(paint_maxpool)

# # tf.to_float(
# #     x,
# #     name='ToFloat'
# # )
# avgpooling = avgpool(downrate)
# paint = tf.cast(paint_rot,tf.float32)
# paint_avgpooling = avgpooling(paint)
# paint_avgpooling = tf.cast(np.squeeze(paint_avgpooling),tf.int8)
# print(paint_avgpooling)
# plt.imshow(paint_avgpooling)

plt.show()


