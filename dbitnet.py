def get_dilation_rates(input_size):
    """Helper function to determine the dilation rates of DBitNet given an input_size. """
    drs = []
    while input_size >= 8:
        drs.append(int(input_size / 2 - 1))
        input_size = input_size // 2

    return drs

# def make_model(input_size=64, n_filters=32, n_add_filters=16):
#     """Create a DBITNet model.

#     :param input_size: e.g. for SPECK32/64 the input_size is 64 bit.
#     :return: DBitNet model.
#     """

#     import tensorflow as tf
#     from tensorflow.keras.models import Model
#     from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Lambda, concatenate, BatchNormalization, Activation, Add
#     from tensorflow.keras.regularizers import l2

#     # determine the dilation rates from the given input size
#     dilation_rates = get_dilation_rates(input_size)

#     # prediction head parameters (similar to Gohr)
#     d1 = 256 # TODO this can likely be reduced to 64.
#     d2 = 64
#     reg_param = 1e-5

#     # define the input shape
#     inputs = Input(shape=(input_size, 1))
#     x = inputs

#     # normalize the input data to a range of [-1, 1]:
#     x = Lambda(lambda t: (t - 0.5) / 0.5)(x)

#     for dilation_rate in dilation_rates:
#         ### wide-narrow blocks
#         x = Conv1D(filters=n_filters,
#                    kernel_size=2,
#                    padding='valid',
#                    dilation_rate=dilation_rate,
#                    strides=1,
#                    activation='relu')(x)
#         x = BatchNormalization()(x)
#         x_skip = x
#         x = Conv1D(filters=n_filters,
#                    kernel_size=2,
#                    padding='causal',
#                    dilation_rate=1,
#                    activation='relu')(x)
#         x = Add()([x, x_skip])
#         x = BatchNormalization()(x)

#         n_filters += n_add_filters

#     ### prediction head
#     out = tf.keras.layers.Flatten()(x)

#     dense0 = Dense(d1, kernel_regularizer=l2(reg_param))(out);
#     dense0 = BatchNormalization()(dense0);
#     dense0 = Activation('relu')(dense0);
#     dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(dense0);
#     dense1 = BatchNormalization()(dense1);
#     dense1 = Activation('relu')(dense1);
#     dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
#     dense2 = BatchNormalization()(dense2);
#     dense2 = Activation('relu')(dense2);
#     out = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_param))(dense2)

#     model = Model(inputs, out)

#     return model

# make model with multi head attention
def make_model(input_size=64, n_filters=32, n_add_filters=16, num_heads=4):
    """
    Create a DBITNet model with an added Self-Attention block.
    
    This architecture integrates a Transformer-style attention mechanism before
    the standard DBitNet dilated convolutions. This allows the model to learn
    global bit-level dependencies (O(1) path length) regardless of their distance,
    which is beneficial for ciphers with complex permutation layers.

    :param input_size: e.g. for SPECK32/64 the input_size is 64 bit.
    :param n_filters: Initial number of filters (channels).
    :param n_add_filters: Filters to add at each depth level.
    :param num_heads: Number of attention heads for the MultiHeadAttention layer.
    :return: DBitNet model with Attention.
    """
    
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Conv1D, Dense, Dropout, Lambda, concatenate, 
        BatchNormalization, Activation, Add, MultiHeadAttention, LayerNormalization
    )
    from tensorflow.keras.regularizers import l2

    # Determine dilation rates based on input size 
    dilation_rates = get_dilation_rates(input_size)

    # Prediction head parameters
    d1 = 256 
    d2 = 64
    reg_param = 1e-5

    # Define the input shape
    inputs = Input(shape=(input_size, 1))
    x = inputs

    # 1. Normalization
    x = Lambda(lambda t: (t - 0.5) / 0.5)(x)
    
    # Projection: Expand from 1 channel to n_filters (32) so Attention heads 
    # have sufficient dimensionality to operate on.
    x = Conv1D(filters=n_filters, kernel_size=1, padding='same', activation='linear')(x)
    
    # Multi-Head Self-Attention: 
    # Allows every bit to attend to every other bit (Global Receptive Field).
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=n_filters)(x, x)
    
    # Residual Connection & Normalization (Standard Transformer Block structure)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    for dilation_rate in dilation_rates:
        ### Wide-Narrow Blocks 
        
        # Wide (Dilated) Convolution
        # Note: padding='valid' reduces the sequence length
        x = Conv1D(filters=n_filters,
                   kernel_size=2,
                   padding='valid',
                   dilation_rate=dilation_rate,
                   strides=1,
                   activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Skip connection setup
        x_skip = x
        
        # Narrow (Local) Convolution
        x = Conv1D(filters=n_filters,
                   kernel_size=2,
                   padding='causal', # Preserves length for the add operation
                   dilation_rate=1,
                   activation='relu')(x)
        
        # Residual add
        x = Add()([x, x_skip])
        x = BatchNormalization()(x)

        # Increase filter depth for the next stage
        n_filters += n_add_filters

    ### Prediction Head
    out = tf.keras.layers.Flatten()(x)

    dense0 = Dense(d1, kernel_regularizer=l2(reg_param))(out)
    dense0 = BatchNormalization()(dense0)
    dense0 = Activation('relu')(dense0)
    
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(dense0)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    
    out = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_param))(dense2)

    model = Model(inputs, out)

    return model