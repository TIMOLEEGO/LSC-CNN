
from tensorflow.keras.layers import \
    Conv2D,Input,add,Activation,BatchNormalization,Multiply,Subtract,Concatenate,Conv2DTranspose
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow as tf
import six


try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations
import keras.backend as K

def tanh_layer(x):
    return tf.tanh(x)

class ADresnetBuilder(object):
    def build(self, block_fn, repetitions):
        input_shape = (None,None,1)
        self._handle_dim_ordering()
        block_fn = self._get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = self._conv_bn_relu(filters=128, kernel_size=(7, 7), strides=(1, 1), padding='same')(input)#0708
        conv2 = self._conv_bn_relu(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
        conv3 = self._conv_bn_relu(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv2)  #0708

        block = conv3
        filters = 64
        for i, r in enumerate(repetitions):
            block = self._residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
        block = self._bn_relu(block)
        block = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(block)
        block = self._bn_relu(block)
        block = Conv2D(filters=1,kernel_size=3,padding='same')(block)
        out = Concatenate(3)([input,block])
        out = Activation('relu')(out)
        out = Conv2D(filters=1,kernel_size=3,padding='same')(out)
        out = Multiply()([out,block])
        out2 = Subtract()([input,out])
        model = Model(inputs=input, outputs=out2)
        return model


    # @staticmethod
    def build_resnet18(self):
        return self.build(self.basic_block, [2, 2, 2, 2])

    # @staticmethod
    def build_resnet34(self):
        return self.build( self.basic_block, [3, 4, 6, 3])

    # @staticmethod
    def build_resnet50(self):
        return self.build( self.bottleneck, [3, 4, 6, 3])

    # @staticmethod
    def build_resnet101(self):
        return self.build(self.bottleneck, [3, 4, 23, 3])

    # @staticmethod
    def build_resnet152(self):
        return self.build(self.bottleneck, [3, 8, 36, 3])

    def _bn_relu(self,input):
        """Helper to build a BN -> relu block
        """
        norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
        return Activation("relu")(norm)

    def _conv_bn_relu(self,**conv_params):
        """Helper to build a conv -> BN -> relu block
        """
        filters = conv_params["filters"]
        kernel_size = conv_params["kernel_size"]
        strides = conv_params.setdefault("strides", (1, 1))
        padding = conv_params.setdefault("padding", "same")
        kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

        def f(input):
            conv = Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          kernel_regularizer=kernel_regularizer)(input)
            return self._bn_relu(conv)

        return f


    def _bn_relu_conv(self,**conv_params):
        """Helper to build a BN -> relu -> conv block.
        This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
        """
        filters = conv_params["filters"]
        kernel_size = conv_params["kernel_size"]
        strides = conv_params.setdefault("strides", (1, 1))
        padding = conv_params.setdefault("padding", "same")
        kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

        def f(input):
            activation = self._bn_relu(input)
            return Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          kernel_regularizer=kernel_regularizer)(activation)

        return f


    def _shortcut(self,input, residual):
        """Adds a shortcut between input and residual block and merges them with "sum"
        """
        # Expand channles of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.
        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride_width = 1
        stride_height = 1
        equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

        shortcut = input
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding="valid",
                              kernel_regularizer=l2(0.0001))(input)

        return add([shortcut, residual])


    def _residual_block(self,block_function, filters, repetitions, is_first_layer=False):
        """Builds a residual block with repeating bottleneck blocks.
        """
        def f(input):
            for i in range(repetitions):
                init_strides = (1, 1)
                if i == 0 and not is_first_layer:
                    init_strides = (1, 1)
                input = block_function(filters=filters, init_strides=init_strides,
                                       is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
            return input

        return f


    def basic_block(self,filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
        """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
        Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
        """
        def f(input):

            if is_first_block_of_first_layer:
                # don't repeat bn->relu since we just did bn->relu->maxpool
                conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                               strides=init_strides,
                               padding="same",
                               #kernel_initializer="he_normal",
                               kernel_regularizer=l2(1.e-4))(input)  #1e-6
                               #use_bias=False)(input)   #+use_bias=False
            else:
                conv1 = self._bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                      strides=init_strides)(input)
                                           #use_bias=False)(input)   #+use_bias=False

            residual = self._bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
            return self._shortcut(input, residual)

        return f


    def bottleneck(self,filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
        """Bottleneck architecture for > 34 layer resnet.
        Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
        Returns:
            A final conv layer of filters * 4
        """
        def f(input):

            if is_first_block_of_first_layer:
                # don't repeat bn->relu since we just did bn->relu->maxpool
                conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                                  strides=init_strides,
                                  padding="same",
                                  #kernel_initializer="he_normal",
                                  kernel_regularizer=l2(1.e-4))(input)#1e-4-->6-->4
                                  #use_bias=False)(input)    #+use_bias=False
            else:
                conv_1_1 = self._bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                         strides=init_strides)(input)

            conv_3_3 = self._bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
            residual = self._bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
            return self._shortcut(input, residual)

        return f

    def _handle_dim_ordering(self):
        global ROW_AXIS
        global COL_AXIS
        global CHANNEL_AXIS
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = -1#3

    def _get_block(self,identifier):
        if isinstance(identifier, six.string_types):
            res = globals().get(identifier)
            if not res:
                raise ValueError('Invalid {}'.format(identifier))
            return res
        return identifier