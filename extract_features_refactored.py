import modeling
import tokenization
import tensorflow as tf
import numpy as np
from toolz import curry

MAX_SEQ_LENGTH = 128;

BERT_VOCAB_FILE= '/tmp/uncased_L-12_H-768_A-12/vocab.txt'
BERT_CONFIG_FILE = '/tmp/uncased_L-12_H-768_A-12/bert_config.json'
INIT_CHECKPOINT = '/tmp/uncased_L-12_H-768_A-12/bert_model.ckpt'

tokenizer = tokenization.FullTokenizer(vocab_file=BERT_VOCAB_FILE, do_lower_case=True)

bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)

LAYER_INDEXES = [-2]
MASTER = None
NUM_TPU_CORES = 8
BATCH_SIZE = 8
USE_TPU = False
USE_ONE_HOT_EMBEDDINGS = False

def log_trainable_variables(tvars, initialized_variable_names):
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ", *INIT_FROM_CKPT*" if var.name in initialized_variable_names else ""

      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

def model_fn(features, labels, mode, params):
    predictions = get_predictions_fn(features, labels, mode, params)
    scaffold_fn = (lambda: tf.train.Scaffold()) if USE_TPU else None
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)

def get_predictions_fn(features, labels, mode, params):
    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=USE_ONE_HOT_EMBEDDINGS)
    if mode != tf.estimator.ModeKeys.PREDICT:
        raise ValueError("Only PREDICT modes are supported: %s" % (mode))
    tvars = tf.trainable_variables()

    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, INIT_CHECKPOINT)

    tf.train.init_from_checkpoint(INIT_CHECKPOINT, assignment_map)

    log_trainable_variables(tvars, initialized_variable_names)

    all_layers = model.get_all_encoder_layers()

    predictions = dict(zip(map(lambda i: 'layer_output_%d' % i, LAYER_INDEXES), list(all_layers[i] for i in LAYER_INDEXES)))

    return predictions


def init_estimator():
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=MASTER,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=NUM_TPU_CORES,
            per_host_input_for_training=is_per_host))
    return tf.contrib.tpu.TPUEstimator(use_tpu=USE_TPU, model_fn=model_fn, config=run_config, predict_batch_size=BATCH_SIZE)


def convert_to_input_ids(line):
    line = tokenization.convert_to_unicode(line)
    tokens = tokenizer.tokenize(line)
    tokens = ["[CLS]"] + tokens[0:MAX_SEQ_LENGTH-2] + ["[SEP]"]

    return tokenizer.convert_tokens_to_ids(tokens)

def padding_zero(size, list):
    return list + [0] * (size - len(list))

def input_fn(params):
    features = get_features(params)
    return tf.data.Dataset.from_tensor_slices(features).batch(batch_size=BATCH_SIZE, drop_remainder=False)


def get_features(params):
    num_examples = len(lines)

    all_unique_ids = np.cumsum([1]*num_examples)

    all_input_ids = map(convert_to_input_ids, lines)

    all_input_mask = map(lambda input_ids: padding_zero(MAX_SEQ_LENGTH, [1]* len(input_ids)), all_input_ids)

    all_input_type_ids = np.zeros([num_examples, MAX_SEQ_LENGTH], dtype=int).tolist()

    padded_all_input_ids = map(curry(padding_zero)(MAX_SEQ_LENGTH), all_input_ids)

    data_block_shape = [num_examples, MAX_SEQ_LENGTH]
    print(list(np.shape(all_input_type_ids)))
    print(data_block_shape)
    assert list(np.shape(padded_all_input_ids)) == data_block_shape
    assert list(np.shape(all_input_type_ids)) == data_block_shape
    assert list(np.shape(all_input_mask)) == data_block_shape

    return {
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        "input_ids":
            tf.constant(
                padded_all_input_ids, shape=data_block_shape,
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=data_block_shape,
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=data_block_shape,
                dtype=tf.int32),
    }


def extract_features(estimator, lines):
    fn = input_fn(lines)
    for result in estimator.predict(input_fn, yield_single_examples=True):
        print("result: ", result)


lines = [
  "New 2019 TOYOTA CAMRY 2532 4dr Sdn"
]


estimator = init_estimator()

extract_features(estimator, lines)

features = get_features(None)
fn = get_predictions_fn(features, None, tf.estimator.ModeKeys.PREDICT, None)

with tf.Session() as sess:
     writer = tf.summary.FileWriter("log_dir", sess.graph)
     print(sess.run(fn))
     writer.close()
