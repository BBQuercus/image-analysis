import tensorflow as tf

option_dict_conv = {'activation': 'relu', 'padding': 'same'}
option_dict_bn = {'axis': -1, 'momentum': 0.9}

def standard_unet(categorical=True, img_size=None):
    '''
    '''
    #TODO add shape assertion (%2**depth==0) or add padding
    x = tf.keras.layers.Input((img_size, img_size, 1))

    # Down
    a = tf.keras.layers.Conv2D(8, (3, 3), **option_dict_conv) (x)
    a = tf.keras.layers.BatchNormalization(**option_dict_bn) (a)
    a = tf.keras.layers.Conv2D(8, (3, 3), **option_dict_conv) (a)
    a = tf.keras.layers.BatchNormalization(**option_dict_bn) (a)
    y = tf.keras.layers.MaxPool2D((2, 2)) (a)

    b = tf.keras.layers.Conv2D(16, (3, 3), **option_dict_conv) (y)
    b = tf.keras.layers.BatchNormalization(**option_dict_bn) (b)
    b = tf.keras.layers.Conv2D(16, (3, 3), **option_dict_conv) (b)
    b = tf.keras.layers.BatchNormalization(**option_dict_bn) (b)
    y = tf.keras.layers.MaxPool2D((2, 2)) (b)

    c = tf.keras.layers.Conv2D(32, (3, 3), **option_dict_conv) (y)
    c = tf.keras.layers.BatchNormalization(**option_dict_bn) (c)
    c = tf.keras.layers.Conv2D(32, (3, 3), **option_dict_conv) (c)
    c = tf.keras.layers.BatchNormalization(**option_dict_bn) (c)
    y = tf.keras.layers.MaxPool2D((2, 2)) (c)

    d = tf.keras.layers.Conv2D(64, (3, 3), **option_dict_conv) (y)
    d = tf.keras.layers.BatchNormalization(**option_dict_bn) (d)
    d = tf.keras.layers.Conv2D(64, (3, 3), **option_dict_conv) (d)
    d = tf.keras.layers.BatchNormalization(**option_dict_bn) (d)
    y = tf.keras.layers.MaxPool2D((2, 2)) (d)

    # Up
    e = tf.keras.layers.Conv2D(128, (3, 3), **option_dict_conv) (y)
    e = tf.keras.layers.BatchNormalization(**option_dict_bn) (e)
    e = tf.keras.layers.Conv2D(128, (3, 3), **option_dict_conv) (e)
    e = tf.keras.layers.BatchNormalization(**option_dict_bn) (e)
    e = tf.keras.layers.UpSampling2D() (e)
    y = tf.keras.layers.concatenate([e, d], axis=3)

    f = tf.keras.layers.Conv2D(64, (3, 3), **option_dict_conv) (y)
    f = tf.keras.layers.BatchNormalization(**option_dict_bn) (f)
    f = tf.keras.layers.Conv2D(64, (3, 3), **option_dict_conv) (f)
    f = tf.keras.layers.BatchNormalization(**option_dict_bn) (f)
    f = tf.keras.layers.UpSampling2D() (f)
    y = tf.keras.layers.concatenate([f, c], axis=3)

    g = tf.keras.layers.Conv2D(32, (3, 3), **option_dict_conv) (y)
    g = tf.keras.layers.BatchNormalization(**option_dict_bn) (g)
    g = tf.keras.layers.Conv2D(32, (3, 3), **option_dict_conv) (g)
    g = tf.keras.layers.BatchNormalization(**option_dict_bn) (g)
    g = tf.keras.layers.UpSampling2D() (g)
    y = tf.keras.layers.concatenate([g, b], axis=3)

    h = tf.keras.layers.Conv2D(16, (3, 3), **option_dict_conv) (y)
    h = tf.keras.layers.BatchNormalization(**option_dict_bn) (h)
    h = tf.keras.layers.Conv2D(16, (3, 3), **option_dict_conv) (h)
    h = tf.keras.layers.BatchNormalization(**option_dict_bn) (h)
    h = tf.keras.layers.UpSampling2D() (h)
    y = tf.keras.layers.concatenate([h, a], axis=3)

    y = tf.keras.layers.Conv2D(8, (3, 3), **option_dict_conv) (y)
    y = tf.keras.layers.Conv2D(8, (3, 3), **option_dict_conv) (y)

    activation = 'softmax' if categorical else 'sigmoid'
    channels = 3 if categorical else 1

    y = tf.keras.layers.Conv2D(channels, (1, 1), activation=activation) (y)

    model = tf.keras.models.Model(inputs=[x], outputs=[y])
    
    # if categorical:
    #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
    # if not categorical:
    #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model