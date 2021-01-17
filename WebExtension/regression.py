import tensorflow as tf
from transformers import PreTrainedTokenizer,BertTokenizer,TFBertModel, BertConfig
from tqdm.auto import tqdm
import numpy as np

distil_bert = 'bert-base-uncased'
distil_bert = 'bert-base-uncased'

tokenizer_new = BertTokenizer.from_pretrained(distil_bert, do_lower_case=True, add_special_tokens=True,max_length=128, pad_to_max_length=True)

def tokenize_new(sentences, tokenizer_new):
    input_ids, input_masks, input_segments = [],[],[]
    for sentence in tqdm(sentences):
        inputs = tokenizer_new.encode_plus(sentence, add_special_tokens=True, max_length=128, pad_to_max_length=True, 
                                             return_attention_mask=True, return_token_type_ids=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])        
        
    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments, dtype='int32')

config1 = BertConfig(dropout=0.2, attention_dropout=0.2)
config1.output_hidden_states = False
transformer_model =TFBertModel.from_pretrained(distil_bert, config = config1)

input_ids_in = tf.keras.layers.Input(shape=(128,), name='input_token', dtype='int32')
input_masks_in = tf.keras.layers.Input(shape=(128,), name='masked_token', dtype='int32') 

embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1,kernel_initializer='normal'))(embedding_layer)
X = tf.keras.layers.GlobalMaxPool1D()(X)
X = tf.keras.layers.Dense(50, activation='relu',kernel_initializer='normal')(X)
X = tf.keras.layers.Dropout(0.2)(X)
X = tf.keras.layers.Dense(1, activation='linear',kernel_initializer='normal')(X)
model1 = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs = X)

for layer in model1.layers[:3]:
  layer.trainable = False

weights_file = 'Weights-030--0.82970.hdf5' 
model1.load_weights(weights_file)
model1.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
# print(model1.summary())

# sent=["hi how are tou"]

# a,b,c=tokenize_new(sent,tokenizer_new)

# out=model1.predict([a,b])

# print(out)
