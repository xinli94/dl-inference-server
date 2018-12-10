import tensorflow as tf
from PIL import Image
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_image',
                    type=str,
                    default='/data5/xin/commercials_2400k/val_tiny/FUS_1527514253_1527514288.mp4_00000029.png',
                    help='path to input image')
parser.add_argument('--output_path',
                    type=str,
                    default='/data5/xin/test/',
                    help='output path to save the resized image')


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)

    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")

    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])

    image_str = tf.image.encode_jpeg(tf.squeeze(tf.cast(resized, tf.uint8)), name='image_str')

    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    # sess = tf.Session()
    # result = sess.run(normalized)
    # return result

    return file_reader, image_reader, resized, normalized, image_str

args = parser.parse_args()
input_image = args.input_image

image_name = os.path.basename(input_image)
output_image = os.path.join(args.output_path, os.path.splitext(image_name)[0] + '_resized.jpg')

# print('>>>>>>>>>> original: ', open(input_image).read()[:100])

file_reader, image_reader, resized_reader, normalized_reader, image_str = read_tensor_from_image_file(input_image)

with tf.Session() as sess:
    sess = tf.Session()
    # encoded = sess.run(file_reader)
    # print('>>>>>>>>>> encoded: ', encoded[:10])
    # image = sess.run(image_reader)
    # print('>>>>>>>>>> image: ', image[:10])
    # resized = sess.run(resized_reader)
    # print('>>>>>>>>>> resized: ', resized[:10])


    encoded = sess.run(image_str)
    with open(output_image, 'wa+') as f:
        f.write(encoded)

    # print('>>>>>>>>>> resized_fixed: ', resized[:10] / 255.0)

    # normalized = sess.run(normalized_reader)
    # print('>>>>>>>>>> normalized: ', normalized[:10])

    # print('>>>>>>> bytes: ', normalized[:10].tobytes()[:100])
