import tensorflow as tf

w = tf.Variable(2.0, dtype=tf.float64)

with tf.GradientTape() as tape:
    y = w**2 + 3*w + 2

gradient = tape.gradient(y, w)
print(float(gradient.numpy()))  # 7.0
