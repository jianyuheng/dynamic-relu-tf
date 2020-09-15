import tensorflow as tf

def xavier_initializer():
    return tf.variance_scaling_initializer(
            scale=1.0,
            mode="fan_in",
            distribution="uniform")

def conv_kernel_initializer():
    return xavier_initializer()

def dense_kernel_initializer():
    return xavier_initializer()
    
class DyReLU(object):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        self.channels = channels
        self.k = k 
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = tf.layers.Dense(
                channels // reduction,
                kernel_initializer=dense_kernel_initializer())
        self.relu = tf.nn.relu
        self.fc2 = tf.layers.Dense(
                2*k*channels,
                kernel_initializer=dense_kernel_initializer())
        self.sigmoid = tf.math.sigmoid

        self.lambdas = tf.constant([1.]*k + [0.5]*k, dtype=tf.float32)
        self.init_v = tf.constant([1.] + [0.]*(2*k - 1), dtype=tf.float32)

    def get_relu_coefs(self, x):
        theta = tf.reduce_mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = tf.reduce_mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)
        relu_coefs = tf.reshape(theta, [-1, self.channels, 2*self.k]) * self.lambdas + self.init_v

        # BxCxHxW -> HxWxBxCx1
        x_perm = tf.expand_dims(tf.transpose(x, [2, 3, 0, 1]), axis=-1)
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        # HxWxBxCx2 -> BxCxHxW
        result = tf.transpose(tf.reduce_max(output, axis=-1), [2, 3, 0, 1])
        return result
