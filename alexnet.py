import tensorflow as tf
import numpy as np

def weights_variable(shape, stddev=0.01):
    weights = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=stddev, dtype=tf.float32))
    return weights

def bias_variable(shape, stddev=0.01):
    biases = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=stddev, dtype=tf.float32))
    return biases

class AlexNet:

    learning_rate   = 0.001
    training_iters  = 200000
    batch_size      = 1024
    
    n_inputs_w = 227      # 227 * 227
    n_inputs_h = 227
    n_classes  = 2        # dog, cat
    dropout  = 0.8

    x = tf.placeholder (tf.float32, [None, n_inputs_w, n_inputs_h, 3])
    y = tf.placeholder (tf.float32, [None, n_classes])

    dropout_prob = tf.placeholder (tf.float32)  # dropout probability

    weights = {
        'wc1': weights_variable([11, 11, 3, 96]),
        'wc2': weights_variable([3, 3, 96, 256]),
        'wc3': weights_variable([3, 3, 256, 384]),
        'wc4': weights_variable([3, 3, 384, 384]),
        'wc5': weights_variable([3, 3, 384, 256]),

        'wd1': weights_variable([9216, 4096]),
        'wd2': weights_variable([4096, 4096]),

        'out': weights_variable([4096, n_classes]),
    }

    biases = {
        'bc1': bias_variable([96]),
        'bc2': bias_variable([256]),
        'bc3': bias_variable([384]),
        'bc4': bias_variable([384]),
        'bc5': bias_variable([256]),

        'bd1': bias_variable([4096]),
        'bd2': bias_variable([4096]),
    }

    def read_data(self):
        pass

    def conv2d(self, x, weights, stride, padding='SAME'):
        conv = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding=padding)
        return conv

    def max_pool(self, x, ksize, stride, padding='SAME'):
        pool = tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)
        return pool

    """
    def conv_layer(x, side_l, stride, out_depth, padding='SAME', **kwargs):
    새로운 컨볼루션 층을 추가함.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, 필터의 한 변의 길이.
    :param stride: int, 필터의 각 방향으로의 이동 간격.
    :param out_depth: int, 입력값에 적용할 필터의 총 개수.
    :param padding: str, 'SAME' 또는 'VALID',
                         컨볼루션 연산 시 입력값에 대해 적용할 패딩 알고리즘.
    :param kwargs: dict, 추가 인자, 가중치/바이어스 초기화를 위한 하이퍼파라미터들을 포함함.
        - weight_stddev: float, 샘플링 대상이 되는 정규분포의 표준편차 값.
        - biases_value: float, 바이어스의 초기화 값.
    :return: tf.Tensor.
    """
    
        
    def build_model(self, weights, biases):

        with tf.name_scope('conv_layer1'):
            conv1 = self.conv2d(self.x, weights['wc1'], stride=4, padding='VALID')
            print('conv1 : ', conv1.get_shape().as_list())

            bias1 = tf.nn.bias_add(conv1, biases['bc1'])
            relu1 = tf.nn.relu(bias1)

            pool1 = self.max_pool(relu1, ksize=3, stride=2, padding='VALID')
            print('pool1 : ', pool1.get_shape().as_list())

        with tf.name_scope('conv_layer2'):
            conv2 = self.conv2d(pool1, weights['wc2'], stride=1)
            print('conv2 : ', conv2.get_shape().as_list())
            bias2 = tf.nn.bias_add(conv2, biases['bc2'])
            relu2 = tf.nn.relu(bias2)
            pool2 = self.max_pool(relu2, ksize=3, stride=2, padding='VALID')
            print('pool2 : ', pool2.get_shape().as_list())

        with tf.name_scope('conv_layer3'):
            conv3 = self.conv2d(pool2, weights['wc3'], stride=1)
            print('conv3 : ', conv3.get_shape().as_list())
            bias3 = tf.nn.bias_add(conv3, biases['bc3'])
            relu3 = tf.nn.relu(bias3)

        with tf.name_scope('conv_layer4'):
            conv4 = self.conv2d(relu3, weights['wc4'], stride=1)
            print('conv4 : ', conv4.get_shape().as_list())
            bias4 = tf.nn.bias_add(conv4, biases['bc4'])
            relu4 = tf.nn.relu(bias4)

        with tf.name_scope('conv_layer5'):
            conv5 = self.conv2d(relu4, weights['wc5'], stride=1)
            print('conv5 : ', conv5.get_shape().as_list())
            bias5 = tf.nn.bias_add(conv5, biases['bc5'])
            relu5 = tf.nn.relu(bias5)
            pool5 = self.max_pool(relu5, ksize=3, stride=2, padding="VALID")
            print('pool5 : ', pool5.get_shape().as_list())

        with tf.name_scope('fc_layer6'):
            dense_dim = int(np.prod(pool5.get_shape()[1:]))
            dense1 = tf.reshape(pool5, [-1, dense_dim])
            print('dense1 : ', dense1.get_shape().as_list())
            
            fc6 = tf.matmul(dense1, weights['wd1'])
            bias6 = tf.nn.bias_add(fc6, biases['bd1'])
            relu6 = tf.nn.relu(bias6)
            print('fc6 : ', fc6.get_shape().as_list())

        with tf.name_scope('fc_layer7'):
            fc7 = tf.matmul(relu6, weights['wd2'])
            bias7 = tf.nn.bias_add(fc7, biases['bd1'])
            relu7 = tf.nn.relu(bias6)
            print('fc7 : ', fc7.get_shape().as_list())

        with tf.name_scope('fc_layer8'):
            fc8 = tf.matmul(relu7, weights['out'])
            print('fc7 : ', fc8.get_shape().as_list())

        return fc8
            
        
        def build_loss(self):
            weight_decay = 0.0005
            
            predict = self.build_model(self.weights, self.biases)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predict, labels=self.y))





            

        



        


