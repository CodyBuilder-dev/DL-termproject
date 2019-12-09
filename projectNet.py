#Network Training 파일

import tensorflow as tf
import os
import numpy as np
from CIFAR_10 import CIFAR_10
from Func import *
from Layer import *
from time import time

######################
#초기화 및 시작시간 측정
######################

tf.reset_default_graph()
start_time = time()

#######################
#hyper-parameter 설정
#######################

#output label 개수 설정
label_dim = 10

#res_n = 네트워크 깊이 설정(18,34,50,101,152중 선택)
#ch = 네트워크 channel 수 설정
res_n = 18
ch = 8 # ResNet paper is 64
learning_rate = 5e-3

#training 변수 설정
is_training = True
total_epoch = 66
batch_size = 64
total_batch = 10000//batch_size


#######################
#데이터,입출력 설정
#######################

dataset = CIFAR_10()
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])

#######################
#Network 설정
#######################

#네트워크 깊이가 50 이상일 경우 bottleneck layer 사용
if res_n < 50 :
	residual_block = resblock
else :
	residual_block = bottle_resblock

residual_list = get_residual_layer(res_n)


#Res Layer-1
x = conv(X, channels=ch, kernel=3, stride=1, scope='conv')

for i in range(residual_list[0]) :
	x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))
#->출력 크기 32x32xch

########################################################################################################

#Res Layer-2
x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')
#->출력 크기 16x16xch*2

for i in range(1, residual_list[1]) :
	x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))
#->출력 크기 16x16xch*2
########################################################################################################

#Res Layer-3
x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')
#->출력 크기 8x8xch*4
for i in range(1, residual_list[2]) :
	x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))
#->출력 크기 8x8xch*4
########################################################################################################

#Res Layer-4
x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')
#->출력 크기 4x4xch*8
for i in range(1, residual_list[3]) :
	x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))
#->출력 크기 4x4xch*8
########################################################################################################

#최종 FNN layer
x = batch_norm(x, is_training, scope='batch_norm')
x = relu(x)

y = global_avg_pooling(x)

logits = fully_conneted(y, units=label_dim, scope='logit')
#->출력 크기 10
y_pred = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#######################
#모델 저장경로 선언
#######################

saver = tf.train.Saver()
if not os.path.isdir('./model'):
	os.mkdir('./model')
#model 저장 path 및 이름 지정
model_path = './model/model'

########################
#트레이닝 시작시간 측정
########################

train_time = time()
print('Initialization complete!')
print('Init Time : %.2f seconds'%(train_time-start_time))

##############################
#트레이닝 시작
##############################
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Training Loop 시작
for e in range(total_epoch):
    epoch_start_time = time()
    total_cost = 0
    for i in range(total_batch):

        batch_x,batch_y = dataset.next('Train', batch_size)
        batch_x = data_augmentation(batch_x,32)
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

    train_accuracy = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
    loss_print = sess.run(loss, feed_dict={X: batch_x, Y: batch_y})

    valid_batch_x,valid_batch_y = dataset.next('Validation', 2000)
    valid_accuracy = sess.run( accuracy, feed_dict={X: valid_batch_x, Y: valid_batch_y})
    print('epoch : ', e + 1, ' loss : ', loss_print)
    print( 'train acc: ', train_accuracy * 100, 'Valdation acc : ', valid_accuracy*100)
    print('Time : %.2f seconds'%(time()-epoch_start_time))        

print()
print('최적화 완료!')

valid_batch_x,valid_batch_y = dataset.next('Validation', 2000)
valid_accuracy = sess.run( accuracy, feed_dict={X: valid_batch_x, Y: valid_batch_y})
print('최종 정확도 : ',valid_accuracy*100)

saver.save(sess, model_path)
print()
print('Parameter batch = %d ch = %d layer = %d'%(batch_size,ch,res_n))
print('Learning Rate = %d'%learning_rate)
print()
print("Initialization TIme: %.2f seconds"%(train_time-start_time))
print("Training TIme : %.2f seconds"%(time()-train_time))
print("Total Time : %.2f seconds"% (time()-start_time))