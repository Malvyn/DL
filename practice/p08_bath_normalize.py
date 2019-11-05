import tensorflow as tf
import numpy as np

def my_batch_normalize(x, name,momentum=0.99, training=True, eps=le-10, gamma=True,beta=True):
    h, w, c = x.shape[1].value, x.shape[2].value, x.shape[3].value
    with tf.variable_scope(name):
        mean = tf.reduce_mean(x, axis=0) #[h, w, c]
        std = tf.reduce_mean(x**2, axis=0) #[h, w, c]
        std = tf.sqrt(std - mean**2) #[h, w, c]

        mom_mean = tf.get_variable('mom_mean',[h,w,c], tf.float32, tf.initializers.zeros, trainable=False)
        mom_std = tf.get_variable('mom_std', [h, w, c], tf.float32, tf.initializers.zeros, trainable=False)

        update_mean = tf.assign(mom_mean, gamma * mom_mean + (1-gamma) * mean)
        update_std = tf.assign(mom_std, gamma * mom_mean + (1 - gamma) * mean)

        u_std = tf.maximum(update_std, eps)
        u_std = tf.maximum(mom_std, eps)
        x = tf.cond(training, (x - update_mean)/update_std, (x - mom_mean)/mom_std)

        if gamma:
            g = tf.get_variable('gamma', [h,w,c], tf.float32, tf.initializers.ones)
            x = g * x

        if beta:
            b = tf.get_variable('beta', [h,w,c], tf.float32, tf.initializers.zeros)
            x += b

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 64, 64, 30], 'x')
    y = my_batch_normalize(x, 'my_bn', training=training)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer)
        xs = np.random.random([10, 64, 64, 30])
        ys = session.run(y, {x: xs, training: True})
        print(xs)
        print('_'*20)
        print(ys)


