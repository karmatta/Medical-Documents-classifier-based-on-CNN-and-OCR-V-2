# Python script to define model Architecture
# Author: Karthik D

from keras.models import Sequential
from keras_self_attention import SeqSelfAttention
from keras import layers
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Flatten, RepeatVector
from keras.layers import Bidirectional, concatenate, SpatialDropout1D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adagrad

# Function to compute precision, recall and F1
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# Define model architecture
def build_doc_classifier(input_text_shape, input_img_shape, n_classes, vocab_size, dropout=0.4):
    
    # define two sets of inputs
    input_txt = Input(shape=(input_text_shape,))
    img_feats = Input(shape=(input_img_shape, ))
    
    # input and embedding for words
    emb_word = Embedding(input_dim=vocab_size, output_dim=100,
                     input_length=input_text_shape, mask_zero=False, trainable=True)(input_txt)
    bn_wemb = BatchNormalization()(emb_word)
    text_embeddings = Dropout(rate=dropout)(bn_wemb)

    # Entity extraction using self attention
    txt_attn = SeqSelfAttention(attention_activation='sigmoid')(text_embeddings)

    #---------------------------------------------------------------------------------------------------------
    # main LSTM
    main_lstm = Bidirectional(LSTM(units=64, return_sequences=True,
                               recurrent_dropout=0.2))(txt_attn)
    #---------------------------------------------------------------------------------------------------------
    lstm_out = Flatten()(main_lstm)
    txt_features = Dropout(rate=dropout)(lstm_out)

    full_features = concatenate([img_feats, txt_features])

    x = Dense(1000, activation="tanh")(full_features)
    x = Dropout(rate=dropout)(x)
    out = Dense(n_classes, activation="softmax")(x)

    model = Model([img_feats, input_txt], out)

    #print(model.summary())
    opt = Adagrad(lr = 1e-3)
    #sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=["accuracy", f1])
    
    return model



