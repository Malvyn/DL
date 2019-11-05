import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.misc import imsave


# 1. 启动一个交互式的会话
sess = tf.InteractiveSession()

image_path = "./xiaoren.png"
file_contents = tf.read_file(image_path)
image_tensor = tf.image.decode_png(contents=file_contents, channels=0, dtype=tf.uint8)
print("原始数据形状:{}".format(np.shape(image_tensor.eval())))


def show_image(image):
    shape = np.shape(image)
    if len(shape) == 3 and shape[2] == 1:
        plt.imshow(image[:, :, 0], cmap='gray')
        plt.show()
    elif len(shape) == 3:
        plt.imshow(image)
        plt.show()

#显示图像
# show_image(image_tensor.eval())

# 图像缩减
# resize_image_tensor = tf.image.resize_images(images=image_tensor, size=(128, 80),
#                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# print("新的数据形状:{}".format(np.shape(resize_image_tensor.eval())))
# show_image(resize_image_tensor.eval())

# crop_or_pad_image_tensor = tf.image.resize_image_with_crop_or_pad(image_tensor,
#                                                                   target_height=100, target_width=200)
# print("新的数据形状:{}".format(np.shape(crop_or_pad_image_tensor.eval())))
# show_image(crop_or_pad_image_tensor.eval())

# crop_to_bounding_box_image_tensor = tf.image.crop_to_bounding_box(image_tensor, 100, 20, 500, 490)
# print("新的数据形状:{}".format(np.shape(crop_to_bounding_box_image_tensor.eval())))
# show_image(crop_to_bounding_box_image_tensor.eval())

# pad_to_bounding_box_image_tensor = tf.image.pad_to_bounding_box(image_tensor, 200, 100, 1000, 1000)
# print("新的数据形状:{}".format(np.shape(pad_to_bounding_box_image_tensor.eval())))
# show_image(pad_to_bounding_box_image_tensor.eval())

# flip_up_down_image_tensor = tf.image.flip_up_down(image_tensor)
# print("新的数据形状:{}".format(np.shape(flip_up_down_image_tensor.eval())))
# show_image(flip_up_down_image_tensor.eval())

# flip_left_right_image_tensor = tf.image.flip_left_right(image_tensor)
# print("新的数据形状:{}".format(np.shape(flip_left_right_image_tensor.eval())))
# show_image(flip_left_right_image_tensor.eval())


# transpose_image_tensor = tf.image.transpose_image(image_tensor)
# print("新的数据形状:{}".format(np.shape(transpose_image_tensor.eval())))
# show_image(transpose_image_tensor.eval())


# rot90_image_tensor = tf.image.rot90(image_tensor, k=3)
# print("新的数据形状:{}".format(np.shape(rot90_image_tensor.eval())))
# show_image(rot90_image_tensor.eval())

# gray_image_tensor = tf.image.rgb_to_grayscale(image_tensor)
# print("新的数据形状:{}".format(np.shape(gray_image_tensor.eval())))
# show_image(gray_image_tensor.eval())

# adjust_brightness_image_tensor = tf.image.adjust_brightness(image_tensor, delta=0.8)
# print("新的数据形状:{}".format(np.shape(adjust_brightness_image_tensor.eval())))
# show_image(adjust_brightness_image_tensor.eval())

# adjust_hue_image_tensor = tf.image.adjust_hue(image_tensor, delta=-0.6)
# print("新的数据形状:{}".format(np.shape(adjust_hue_image_tensor.eval())))
# show_image(adjust_hue_image_tensor.eval())

# adjust_saturation_image_tensor = tf.image.adjust_saturation(image_tensor, saturation_factor=20)
# print("新的数据形状:{}".format(np.shape(adjust_saturation_image_tensor.eval())))
# show_image(adjust_saturation_image_tensor.eval())


noisy_image_tensor = image_tensor + tf.cast(5 * tf.random_normal(shape=[600, 510, 3], mean=0, stddev=0.1), tf.uint8)
print("新的数据形状:{}".format(np.shape(noisy_image_tensor.eval())))
show_image(noisy_image_tensor.eval())


imsave('test.png', noisy_image_tensor.eval())





















