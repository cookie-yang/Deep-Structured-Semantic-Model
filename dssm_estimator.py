import tensorflow as tf
import numpy as np
import os
import sys
model_save_path = "../model/dssm.ckpt"
NUM_NEGATIVE = 4
LAYER_1_SIZE = 300
LAYER_2_SIZE = 300
LAYER_3_SIZE =300
LAYER_3_SIZE = 128

def model_fn(features,mode):
    #Input layer
    features = tf.convert_to_tensor(features['x'],dtype=tf.float32)
    dim_num = features.shape[-1]
    input_layer = tf.reshape(features,[-1,dim_num],name='input')
    with tf.variable_scope("layer_1"):
        layer1 = tf.layers.dense(input_layer,LAYER_1_SIZE,activation=tf.nn.elu,reuse=tf.AUTO_REUSE,kernel_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope("layer_2"):
        layer2 = tf.layers.dense(layer1,LAYER_2_SIZE,activation=tf.nn.elu,reuse=tf.AUTO_REUSE,kernel_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope("layer_3"):
        layer3 = tf.layers.dense(layer2,LAYER_2_SIZE,activation=tf.nn.elu,reuse=tf.AUTO_REUSE,kernel_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope("prediction"):
        prediction = tf.layers.dense(layer3,LAYER_3_SIZE,activation=tf.nn.elu,reuse=tf.AUTO_REUSE,kernel_initializer=tf.contrib.layers.xavier_initializer())
    with tf.name_scope("cosine_similarity"):
        
        # gama = tf.get_variable("gama",shape = [],trainable=True,initializer=tf.constant_initializer(1.0))
        gama = 1.0
        prediction = tf.reshape(prediction,[-1,NUM_NEGATIVE+2,128])
        n_features = tf.slice(prediction,[0,2,0],[-1,-1,-1])
        q_prediction = tf.slice(prediction,[0,0,0],[-1,1,-1],name='query_prediction')
        d_prediction = tf.slice(prediction,[0,1,0],[-1,-1,-1],name='sample_prediction')
        
        q_prediction_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(q_prediction),2,keep_dims=True)),[1,NUM_NEGATIVE+1,1]) #shape:[batch_size,psample+nsample,128]
        d_prediction_norm = tf.sqrt(tf.reduce_sum(tf.square(d_prediction),2,keep_dims=True))

        cov = tf.reduce_sum(tf.multiply(tf.tile(q_prediction,[1,NUM_NEGATIVE+1,1]),d_prediction,),2,True)
        cov_norm = tf.multiply(q_prediction_norm,d_prediction_norm)
        cos_sim_distance = tf.squeeze(tf.multiply(tf.truediv(cov,cov_norm),gama),name='cos_sim_distance')


    #Prediction
    predictions = {
        "classes":  tf.argmax(input=cos_sim_distance, axis=1),
        "probabilities": tf.nn.softmax(cos_sim_distance, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    labels = np.zeros(shape=cos_sim_distance.shape[0],dtype=np.int16)
    onehot = tf.one_hot(labels,depth=NUM_NEGATIVE+1)   
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot, logits=cos_sim_distance)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
   
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {"accuracy": accuracy}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Add evaluation metrics (for EVAL mode)
   
     # Configure the Training Op (for TRAIN mode)
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.002,momentum=0.9,use_nesterov=True)
    train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
    tf.summary.scalar("my_accuracy",accuracy[1])
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)



def main(unused_argv):
    #read data
    DATAPATH = '../data/wordVector.npz'
    if DATAPATH != None:
        with np.load(DATAPATH) as data:
            data_features = data["vector"]

    # data_features, data_labels = createData("../data/data_s")
    num_data = data_features.shape[0]
    print(num_data)
    train_data_features = data_features[:int(0.995*num_data)]
    test_data_features = data_features[int(0.995*num_data):]
    #create estimator
    cnn_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir="../model/dsssm_estimator"
        )
    #create log
    tensors_to_log = {"probabilities":"softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=1)
    # with tf.Session() as sees:
        # sees.run(iterator.initializer,feed_dict={features_placeholder:features,labels_placeholder:labels})
        # print(sees.run(data))
        # return
        #    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":train_data_features},
        batch_size=8,
        num_epochs=None,
        shuffle=True
    )

    cnn_classifier.train(
        input_fn=train_input_fn,
        steps = 10000)

    #evaluate the model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":test_data_features},
        num_epochs=1,
        shuffle=False
    )
    
    
    result = cnn_classifier.evaluate(
        input_fn=eval_input_fn
    )
    print(result)

if __name__ == "__main__":
    tf.app.run()

