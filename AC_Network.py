import threading
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.nn import elu, softmax
import sys

TRY_TENSORFLOW_GPU = False
if TRY_TENSORFLOW_GPU:  # tries to use tensorflow-gpu
    try:
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
        TENSORFLOW_GPU = True
    except:
        TENSORFLOW_GPU = False
else:
    TENSORFLOW_GPU = False

if TENSORFLOW_GPU:
    from tensorflow.python.keras.layers import CuDNNLSTM as LSTM
else:
    from tensorflow.python.keras.layers import LSTM


def normalized_columns_initializer(std=1.0):
    """
    Used to initialize weights for policy and value output layers
    :param std:
    :return: An array of normalized random values.
    """
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class AC_Network(object):
    ConvLayer1Params = {'num_outputs': 16, 'kernel_size': 8, 'stride': 4, 'activation_fn': elu, 'padding': 'VALID'}
    ConvLayer2Params = {'num_outputs': 32, 'kernel_size': 4, 'stride': 2, 'activation_fn': elu, 'padding': 'VALID'}
    FCLayerParams = {'num_outputs': 256, 'activation_fn': elu}
    LSTMParams = {'units': 256, 'return_state': True, 'unit_forget_bias': True}  #, 'recurrent_activation': None}  # , 'state_is_tuple': True}
    # When 'activation_fn' is set to None it means to skip the activation function and return the product of the linear
    # function. Likewise, when 'biases_initializer' is set to None the biases' initialization is skipped and the layer
    # doesn't use biases.
    ValueLayerParams = {'num_outputs': 1, 'activation_fn': None,
                        'weights_initializer': normalized_columns_initializer(1.0), 'biases_initializer': None}
    ClipNorm = 40.0
    Beta = 0.01  # The Entropy coefficient in the combined loss function.
    SmallValue = 1e-20  # added to the policy vector and used to make sure no value in the policy vector is 0 or Nan as
    #                     log(0) does not exist. (log(0) is minus infinity)

    def __init__(self, s_size, a_size, scope, trainer, train_network_scope='global'):
        """
        Defines the Actor critic network. Handles computing gradients.
        Based heavily on Arthur Juliani's implementation posted on github and on MatheusMRFM's published on GitHub.
        (https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
        https://github.com/MatheusMRFM/A3C-LSTM-with-Tensorflow/blob/master/Network.py)
        :param s_size: Input size, essentially the size of pixels in the screen.(int)
        :param a_size: Action space. The number of actions possible in our environment. (int)
        :param scope: Tensorflow Varables scope. (string)
        :param trainer: A tf.trainer() instance
        :param train_network_scope: Tensorflow Varables scope of the global model. (string)
        """
        self.scope = scope
        self.train_network_scope = train_network_scope
        self.a_size = a_size
        self.s_size = s_size
        self.trainer = trainer
        # When 'biases_initializer' is set to None the biases' initialization is skipped and the layer doesn't use
        # biases. By the way, we only initialized this dict here because we need to have a_size to initialize it.
        self.PolicyLayerParams = {'num_outputs': self.a_size, 'activation_fn': softmax,
                                  'weights_initializer': normalized_columns_initializer(0.01),
                                  'biases_initializer': None}

        """Build the networks graph"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # Here we define our network graph.
            self.build_network()
            # Here we focus on the loss functions and the training process happening in the threads. Not needed in the
            # global network.
            if self.scope != train_network_scope:
                self.build_loss_function()

    def build_network(self):
        """
        Builds the actor critic network's graph
        :return: None
        """
        # input layer
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.s_size], name='inputs')

        # convolution layers
        self.image_input = tf.reshape(self.inputs, [-1, 84, 84, 1])  # it's -1 in shape index 0 because we want to
        #                                                              be capable of handling input of multiple
        #                                                              images at once
        self.conv1 = conv2d(inputs=self.image_input, **self.ConvLayer1Params)
        self.conv2 = conv2d(inputs=self.conv1, **self.ConvLayer2Params)
        # fully connected (dense) layer
        self.fc = fully_connected(tf.layers.flatten(self.conv2), **self.FCLayerParams)

        # LSTM Recurrent Network to deal with temporal differences
        lstm_layer = LSTM(**self.LSTMParams)

        # make init states to be used at time step 0
        state_size = lstm_layer.cell.state_size  # the initial lstm state
        h_init = tf.zeros(shape=[1, state_size[0]], dtype=tf.float32)
        c_init = tf.zeros(shape=[1, state_size[1]], dtype=tf.float32)
        self.init_context = LSTMStateTuple(c_init, h_init)

        batch_size = tf.shape(self.inputs)[0]
        max_time = 1  # since we pass the state manually from time step to time step, we let tensorflow believe it
        #               is 1.
        depth = self.FCLayerParams['num_outputs']  # the input to the lstm is the output of the fc layer, the num of
        #                                            features given at each step is called the depth.
        rnn_in = tf.reshape(self.fc, [batch_size, max_time, depth])  # adds a dimension for fc output.
        #                                   We do that because the RNN call func takes input of shape
        #                                   [batch_size, max_time, depth] when time_major=false.
        #                                   Our batch size is the first value of the shape array describing
        #                                   self.inputs.
        # as for why the c and h are sized the way they are, read:
        # https://www.quora.com/In-LSTM-how-do-you-figure-out-what-size-the-weights-are-supposed-to-be
        h_in = tf.placeholder(dtype=tf.float32, shape=[None, depth], name="h_in")
        c_in = tf.placeholder(dtype=tf.float32, shape=[None, depth], name="c_in")
        self.context_in = h_in, c_in  # will be used to store the lstm state tuple
        # debug_state = LSTMStateTuple(tf.zeros(shape=[batch_size, depth]), tf.zeros(shape=[batch_size, depth]))
        self.lstm_outputs, h_out, c_out = lstm_layer(rnn_in, initial_state=LSTMStateTuple(c_in, h_in))
        self.context_out = [h_out, c_out]

        """
        # ---------------------DEBUG---------------------
        with tf.control_dependencies([tf.print("lstm_outs:", tf.shape(lstm_outputs), "lstm_c:", tf.shape(c_out),
                                               "lstm_h:", tf.shape(h_out), "lstm_all:", 
                                               tf.shape(lstm_outputs_state_tensor))]):
            self.print = tf.constant(1, dtype=tf.int32)
        # ---------------------DEBUG---------------------
        """

        # output layers for policy and values estimations
        self.policy = fully_connected(inputs=self.lstm_outputs, **self.PolicyLayerParams)
        self.value = fully_connected(inputs=self.lstm_outputs, **self.ValueLayerParams)
        # with tf.control_dependencies([tf.print("lstm: ", lstm_outputs, "\nc_out: ", c_out, "\nh_out: ", h_out, output_stream=sys.stdout)]):
        self.value = tf.squeeze(self.value)  # removes dimensions of size 1 from the shape of self.value

        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.train_network_scope)
        self.log_histograms_for_all_global_vars = tf.summary.merge([tf.summary.histogram(var.name, var)
                                                                    for var in global_vars])
        # activations = [self.value, self.policy]
        # self.log_histograms_for_activations = tf.summary.merge([tf.summary.histogram(var.name, var)
        #                                                            for var in activations])

    def build_loss_function(self):
        """
        Builds the graph to calculate the loss functions and enable the agents' training capabilities
        :return: None
        """
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name="actions")
        self.actions_one_hot = tf.one_hot(self.actions, self.a_size, dtype=tf.float32)  # i.e. considering:
        #                                                           (actions = [0, 2], a_zise = 3) =>
        #                                                          actions_one_hot = [[1, 0, 0], [0, 0, 2]]
        self.target_v = tf.placeholder(dtype=tf.float32, shape=[None], name="target_v")
        self.advantages = tf.placeholder(dtype=tf.float32, shape=[None], name="advantages")

        """
        # ---------------------DEBUG---------------------
         with tf.control_dependencies([tf.print("TargetV:", self.target_v, "\nValue: ",
                                               tf.reshape(self.value, shape=[-1]), "\nAdvantage:",
                                               self.advantages)]):
        # use to check that all values make sense (in the basic doom scenario: targevt_v should be
        # between ~-350 and 100, Value should oscillate between iterations and remain in the same range as the
        # target_v and advantage should have a small abs value (-5<adv<5)
        # ---------------------DEBUG---------------------
        """
        """with tf.control_dependencies([tf.print("advantages:", self.advantages,
                                               "targetv :",self.target_v,
                                               "advantages shape:", tf.shape(self.advantages),
                                               "\nlstm_out:", lstm_outputs[-1], "\nh_out:", h_out[-1],
                                               "\nc_out:", c_out[-1], "\nex:", tf.reduce_sum(self.policy * self.actions_one_hot, axis=1) * self.advantages)]):"""


        # Loss functions (formulae in Arthur Juliani's article. The formulae are also explained at
        # https://www.reddit.com/r/MachineLearning/comments/8hu2y5/d_a2ca3c_sharing_the_weights_and_same_loss/ )

        # Value loss function.
        self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - self.value))

        # sometimes, the vector will contain Nan or even strightforward 0 values. Since log(0) is undefined (minus
        # infinity) we make all our values at least of self.SmallValue size. We assume the value is small enough not to
        # harm our calculations while preventing the log from yielding errors.
        log_policy = tf.log(tf.clip_by_value(self.policy, clip_value_min=self.SmallValue,
                                             clip_value_max=1.0))

        responsible_outputs = tf.reduce_sum(log_policy * self.actions_one_hot, axis=1)

        self.entropy = - tf.reduce_sum(self.policy * log_policy)  # entropy is highest when agent is unsure i.e when it
        #                                                           output an actions' probabilities vector of the
        #                                                           following fashion: [0.25, 0.25, 0.25, 0.25] or
        #                                                                              [0.33, 0.33, 0.33].

        # Policy loss function.
        self.policy_loss = - tf.reduce_sum(responsible_outputs * self.advantages)

        # The combined loss function. This is the function we will use to train the model.
        self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * self.Beta  # we use entropy to try and
        #                                                                                  prevent our agent from
        #                                                                                  becoming too sure of
        #                                                                                  itself.
        #                                                                                  It's basically used to
        #                                                                                  encourage exploration.

        # Get gradients from local network using local losses
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        self.gradients = [g for g, v in self.trainer.compute_gradients(self.loss, var_list=local_vars)]  # tf.gradients(self.loss, local_vars)  #
        self.var_norms = tf.global_norm(local_vars)

        """
        Compute the gradient with respect to the local weights and biases and apply them on the train network's vars 
        weights and biases.
        """
        # get vars from the global ACNN. (global ACNN = the train model)
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.train_network_scope)

        # clipped_grads = self.gradients * clip_norm / max(global_norm, clip_norm)
        clipped_grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, clip_norm=self.ClipNorm)
        #  pair clipped gradients with their matching vars in the global ACNN
        grads_and_corresponding_global_vars = list(zip(clipped_grads, global_vars))
        # Apply local gradients to global network
        #with tf.control_dependencies([tf.print("Num of Vars:", len(global_vars), "Vars Names:", [v.name for v in global_vars])]):
        self.apply_grads = self.trainer.apply_gradients(grads_and_corresponding_global_vars,
                                                   name="ApplyGradsOnGlobal")

        self.log_histograms_for_all_global_vars = tf.summary.merge([tf.summary.histogram(var.name, var)
                                                                    for var in global_vars])
        """
        # ---------------------DEBUG---------------------
        #with tf.control_dependencies([tf.print("\nConv1: ", self.conv1[0], "\nConv2: ", self.conv2[0], "\nFC: ", self.fc[0],
        #                                       "\nlstm: ", lstm_outputs[0], "\nPolicy: ", self.policy[0], "\nValue: ",
        #                                       self.value[0], output_stream=sys.stdout)]):
        #with tf.control_dependencies([tf.print("Grad Mean:", tf.reduce_mean([tf.reduce_mean(g) for g in self.gradients]),
        #                                       "Mean Global Vars:", tf.reduce_mean([tf.reduce_mean(g) for g in global_vars]))]):
        before = [gv[0] for gv in global_vars]

        with tf.control_dependencies([self.apply_grads]):
            with tf.control_dependencies([tf.print("TargetV:", self.target_v, "\nValue: ",
                                                   self.value, "\nAdvantage:", self.advantages, "\nResponsible Outputs:",
                                                   self.responsible_outputs, "\nLoss: ", self.loss, "\nValue Loss: "
                    , self.value_loss, "\nPolicy Loss: ", self.policy_loss), tf.print("\nConv1: ", self.conv1[0], "\nConv2: ", self.conv2[0], "\nFC: ", self.fc[0],
                                               "\nlstm: ", lstm_outputs[:,0], "\nPolicy: ", self.policy[:, 0], "\nValue: ",
                                               self.value[0], output_stream=sys.stdout)]):
                self.apply_grads = tf.print("Global Vars Update: ", [["Var: ", before.name, "Before Update: ",
                                                                      before, *["After Update: ", after[0],
                                                                                "Clipped Grad: ", clip_g[0],
                                                                                "Real Grad:", real_g[0]]]
                                                                     for before, after, clip_g, real_g
                                                                     in zip(before, global_vars, clipped_grads,
                                                                     self.gradients)][5])
        # ---------------------DEBUG---------------------
        """

