import tensorflow as tf
import numpy as np


model_save_path = "../model/dssm.ckpt"
NUM_NEGATIVE = 4
LAYER_1_SIZE = 300
LAYER_2_SIZE = 300
LAYER_3_SIZE =300
LAYER_3_SIZE = 128

def getData(DATAPATH,dim_n,batch_size):
    # record_iterator = tf.python_io.tf_record_iterator(path="../data/data_tf_record")
    # for x in record_iterator:
    #     example = tf.train.Example()
    #     example.ParseFromString(x)
    #     height = example.features.feature['q'].float_list
    #     print(tf.size(height))
    #     break

    
    filename_queue = tf.train.string_input_producer([DATAPATH])  
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
  # Defaults are not specified since both keys are required.
        features={
            'q': tf.FixedLenFeature([], tf.string),
            'p': tf.FixedLenFeature([], tf.string),
            'n_1': tf.FixedLenFeature([], tf.string),
            'n_2': tf.FixedLenFeature([], tf.string),
            'n_3': tf.FixedLenFeature([], tf.string),
            'n_4': tf.FixedLenFeature([], tf.string),
            'dim_n':tf.FixedLenFeature([],tf.int64)
        })

    q = tf.reshape(tf.decode_raw(features['q'],tf.float32),[dim_n])
    p = tf.reshape(tf.decode_raw(features['p'],tf.float32),[dim_n])
    n_1 = tf.reshape(tf.decode_raw(features['n_1'],tf.float32),[dim_n])
    n_2 = tf.reshape(tf.decode_raw(features['n_2'],tf.float32),[dim_n])
    n_3 = tf.reshape(tf.decode_raw(features['n_2'],tf.float32),[dim_n])
    n_4 = tf.reshape(tf.decode_raw(features['n_3'],tf.float32),[dim_n])

    data_features = tf.concat([q,p,n_1,n_2,n_3,n_4],axis=0)

    feature_batch = tf.train.shuffle_batch( [data_features],
                                                 batch_size=batch_size,
                                                 capacity=2*batch_size,
                                                 num_threads=32,
                                                 min_after_dequeue=batch_size)
    feature = tf.contrib.slim.prefetch_queue.prefetch_queue([feature_batch])
    
    return feature

def model_fn(features,dim_num):
    input_layer = tf.reshape(features,[-1,dim_num],name='input')
    with tf.variable_scope("layer_1"):
        layer1 = tf.layers.dense(input_layer,LAYER_1_SIZE,activation=tf.nn.elu,reuse=tf.AUTO_REUSE,kernel_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope("layer_2"):
        layer2 = tf.layers.dense(layer1,LAYER_2_SIZE,activation=tf.nn.elu,reuse=tf.AUTO_REUSE,kernel_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope("layer_3"):
        layer3 = tf.layers.dense(layer2,LAYER_2_SIZE,activation=tf.nn.elu,reuse=tf.AUTO_REUSE,kernel_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope("prediction"):
        prediction = tf.layers.dense(layer3,LAYER_3_SIZE,activation=tf.nn.elu,reuse=tf.AUTO_REUSE,kernel_initializer=tf.contrib.layers.xavier_initializer())
    return prediction
def tower_loss(scope,features,dim_num):
    with tf.name_scope("cosine_similarity"):
        
        # gama = tf.get_variable("gama",shape = [],trainable=True,initializer=tf.constant_initializer(1.0))
        gama = tf.get_variable("gama",[],initializer=tf.constant_initializer(1.0),trainable=True)
        prediction = tf.reshape(model_fn(features,dim_num),[-1,NUM_NEGATIVE+2,128])
        n_features = tf.slice(prediction,[0,2,0],[-1,-1,-1])
        q_prediction = tf.slice(prediction,[0,0,0],[-1,1,-1],name='query_prediction')
        d_prediction = tf.slice(prediction,[0,1,0],[-1,-1,-1],name='sample_prediction')
        
        q_prediction_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(q_prediction),2,keep_dims=True)),[1,NUM_NEGATIVE+1,1]) #shape:[batch_size,psample+nsample,128]
        d_prediction_norm = tf.sqrt(tf.reduce_sum(tf.square(d_prediction),2,keep_dims=True))

        cov = tf.reduce_sum(tf.multiply(tf.tile(q_prediction,[1,NUM_NEGATIVE+1,1]),d_prediction,),2,True)
        cov_norm = tf.multiply(q_prediction_norm,d_prediction_norm)
        cos_sim_distance = tf.squeeze(tf.multiply(tf.truediv(cov,cov_norm),gama),name='cos_sim_distance')
    
    with tf.name_scope("loss"):
        label = np.zeros(shape=cos_sim_distance.shape[0],dtype=np.int16)
        onehot = tf.one_hot(label,depth=NUM_NEGATIVE+1)   
        tf.losses.softmax_cross_entropy(logits=cos_sim_distance,onehot_labels=onehot)
    
        # prob = tf.nn.softmax(cos_sim_distance)
        # hit_prob = tf.slice(prob, [0, 0], [-1, 1])
        # loss = -tf.reduce_sum(tf.log(hit_prob)) / 10
        loss = tf.get_collection('losses',scope=scope)
        total_loss = tf.add_n(loss,name='total_loss')
        tf.summary.scalar("total_loss",total_loss)
        return total_loss,cos_sim_distance

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g,_ in grad_and_vars:
            grads.append(tf.expand_dims(g,0))
        grad = tf.concat(grads,0)   
        grad = tf.reduce_mean(grad,0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad,v)
        average_grads.append(grad_and_var)
    return average_grads



def train():
    dim_num = 7157
    with tf.Graph().as_default(),tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False,dtype = tf.float64)
        learningRate = tf.train.exponential_decay(0.05,global_step=global_step,decay_rate=0.5,decay_steps=100000,staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learningRate,momentum=0.95,use_nesterov=True)
        
        batch_size = 1024
       
        data_Path = '../data/data.tf'
        grads = []
        batch_features = getData(data_Path,dim_num,batch_size)
        with tf.variable_scope(tf.get_variable_scope()):   
            for i in range(2):
                with tf.device('/gpu:{:d}'.format(i)):
                  with tf.name_scope("tower_{:d}".format(i)) as scope:
                    loss = tower_loss(scope,batch_features.dequeue(),dim_num)
                    tower_grad = optimizer.compute_gradients(loss[0])
                    grads.append(tower_grad)
                    tf.get_variable_scope().reuse_variables()
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,scope=scope)
        
        
        summaries.append(tf.summary.scalar('learning_rate',learningRate))
        
        average_grad = average_gradients(grads)
        for g,v in average_grad:
            if g is not None:
                summaries.append(tf.summary.histogram(v.op.name+'/gradients',g))
        apply_grad_op = optimizer.apply_gradients(average_grad,global_step=global_step)
        for v in tf.trainable_variables():
            summaries.append(tf.summary.histogram(v.op.name,v))

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        print(ema)

        average_variable_op = ema.apply(tf.trainable_variables())
        

        train_op = tf.group(apply_grad_op,average_variable_op)

        saver = tf.train.Saver(tf.global_variables(),max_to_keep=4)

        summary_op = tf.summary.merge(summaries)

        init = tf.global_variables_initializer()
        config=tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        train_writer = tf.summary.FileWriter(model_save_path,sess.graph)
        

        for i in range(1000000):
            _,s,l = sess.run([train_op,summary_op,loss])
            train_writer.add_summary(s,i)
            print(l[0])
            assert not np.isnan(l[0]),"model diverged with loss == nan"
            if i == 0 :
                saver.save(sess=sess,save_path=model_save_path,global_step=i,write_meta_graph=True)
            elif i % 100 == 0:
                saver.save(sess=sess,save_path=model_save_path,global_step=i,write_meta_graph=False)


# def test():
#     data_Path = "../data/test"
#     graph = tf.Graph()
#     with graph.as_default():
#         with tf.Session() as sess:
#             saver = tf.train.import_meta_graph(model_save_path+".meta")
#             saver.restore(sess,model_save_path)
#             features = getData(data_Path,7157)
#             data_features = tf.convert_to_tensor(features,dtype=tf.float32,name="test_input_values")
#             dim_num = data_features.shape[-1]
#             prediction = model_fn(data_features,dim_num)

      
        


        
if __name__ == "__main__":
    train()