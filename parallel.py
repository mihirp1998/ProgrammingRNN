import tensorflow as tf
import numpy as np
a = np.random.randn(10000)
sum = 0
import time
t1 = time.time()
for i in range(len(a)):
	sum = sum + a[i] *a[i]
t2 = time.time()
print("first time - {} answer {}".format(t2-t1,sum))
t3 = time.time()
finalsum = np.matmul(a,a)
t4 = time.time()

psum = tf.matmul(np.reshape(a,(a.shape[0],1)),np.reshape(a,(1,a.shape[0])))
ses = tf.Session()
t5 = time.time()
val  = ses.run(psum)
t6 = time.time()
print("second time - {} answer {}".format(t4-t3,finalsum))
print("third time - {} answer {}".format(t6-t5,val))
