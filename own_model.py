import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt

def forward_prop(params,placeholder):
	w1 = params["w1"]
	w2 = params["w2"]
	w3 = params["w3"]
	b1 = params["b1"]
	b2 = params["b2"]
	b3 = params["b3"]

	x= placeholder["x"]
	z1 = tf.matmul(x,w1) + b1
	print(x,z1)
	a1 = tf.nn.tanh(z1)
	# a1 = tf.nn.dropout(a1,0.7)
	
	z2 = tf.matmul(a1,w2) + b2
	a2 = tf.nn.relu(z2,name='a2')
	# a2 = tf.nn.dropout(a2,0.7)

	z3 = tf.matmul(a2,w3) + b3
	a3 = tf.nn.relu(z3,name='a3')
	return a3

def initialize_param():
	regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)                                                                                                                                                                                
	w1 = tf.get_variable("w1",[18,14],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(),regularizer=regularizer)
	b1 = tf.get_variable("b1",[1,14],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(),regularizer=regularizer)
	w2 = tf.get_variable("w2",[14,6],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(),regularizer=regularizer)
	b2 = tf.get_variable("b2",[1,6],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(),regularizer=regularizer)
	w3 = tf.get_variable("w3",[6,1],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(),regularizer=regularizer)
	b3 = tf.get_variable("b3",[1,1],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(),regularizer=regularizer)
	return {"w1":w1,"b1":b1,"w2":w2,"b2":b2,"w3":w3,"b3":b3}

def placeholders():
	x = tf.placeholder(tf.float32,shape=[None,18],name='x')
	y = tf.placeholder(tf.float32,shape=[None,1],name='y')
	return {"x":x,"y":y}

def main():	
	params = initialize_param()
	placeholder = placeholders()		
	x= placeholder["x"]
	y = placeholder["y"]
	train_batch,train_label_batch,test_batch,test_label_batch = getData()
	a2 = forward_prop(params,placeholder)
	print(a2)
	loss = tf.losses.mean_squared_error(placeholder['y'],a2)
	reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	reg_constant = 0.01  # Choose an appropriate one.
	loss = tf.add(loss,reg_constant * tf.reduce_sum(reg_losses))
	optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss,name = "min")	
	sess = tf.Session()
	# new_restore = tf.train.import_meta_graph('my_model-1000.meta')
	# new_restore.restore(sess,tf.train.latest_checkpoint('./'))
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = sess,coord=coord)
	init = tf.global_variables_initializer()
	sess.run(init)
	saver = tf.train.Saver()
	for i in range(10001):
		print('num of epoch ',i)
		X_train,Y_train = sess.run([train_batch,train_label_batch])
		loss_val, opt =  sess.run([loss,optimize],feed_dict={x:X_train,y:Y_train})
		if i%2500 ==0:
			print('saving')
			tf.add_to_collection("optimizer", optimize)
			saver.save(sess,'abu_dhabi/abu_model',global_step = i)
		print("loss",loss_val,"opt",opt)
	coord.request_stop()
	coord.join(threads)
	sess.close()

def mainContinue():
	tf.reset_default_graph()
	learning_rate =0.001
	sess = tf.Session()
	new_restore = tf.train.import_meta_graph('model/mymodel-20000.meta')
	new_restore.restore(sess,tf.train.latest_checkpoint('./model/'))	
	saver = tf.train.Saver()
	graph = tf.get_default_graph()
	xP = graph.get_tensor_by_name("x:0")
	yP = graph.get_tensor_by_name("y:0")
	a2 = graph.get_tensor_by_name("a3:0")
	
	loss = tf.losses.mean_squared_error(yP,a2)

	# # b = graph.get_tensor_by_name("b1:0")
	train_batch,train_label_batch,test_batch,test_label_batch = getData()
	# opt = tf.get_collection("optimizer")[0]
	opt = tf.train.RMSPropOptimizer(learning_rate=0.0001,name='il')
	train_opt = opt.minimize(loss)
	optimizer_slots = [
	opt.get_slot(var, name)
	for name in opt.get_slot_names()
	for var in tf.trainable_variables()
	]
	optimizer_slots=[i for i in optimizer_slots if i != None]
	if isinstance(opt, tf.train.AdamOptimizer):
		optimizer_slots.extend(i for i in tf.global_variables() if 'beta'   in i.name)
	
	
	# a2 = graph.get_operation_by_name("min")
	saver = tf.train.Saver()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = sess,coord=coord)
	print(optimizer_slots,'what')
	init = tf.variables_initializer(optimizer_slots)
	sess.run(init)
	for i in range(20001,30001):
		print('num of epoch ',i)
		X_train,Y_train = sess.run([train_batch,train_label_batch])
		loss_val,_ =  sess.run( [loss,train_opt],feed_dict={xP:X_train,yP:Y_train})
		if i%2500 == 0:
			print('saving')
			saver.save(sess,'./model/mymodel',global_step = i)
		print("loss",loss_val)
	coord.request_stop()
	coord.join(threads)
	sess.close()



def evaluate():
	train_batch,train_label_batch,test_batch,test_label_batch = getData()	
	sess = tf.Session()
	new_restore = tf.train.import_meta_graph('model/mymodel-20000.meta')
	new_restore.restore(sess,tf.train.latest_checkpoint('./model/'))	
	graph = tf.get_default_graph()
	xP = graph.get_tensor_by_name("x:0")
	yP = graph.get_tensor_by_name("y:0")
	a2 = graph.get_tensor_by_name("a3:0")
	loss = tf.losses.mean_squared_error(yP,a2)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = sess,coord=coord)
	X_test,Y_test = sess.run([test_batch,test_label_batch])
	cost,ans = sess.run([loss,a2],feed_dict={xP:X_test,yP:Y_test})
	# print(cost,ans-Y_test)
	coord.request_stop()
	coord.join(threads)
	sess.close()
	return ans,cost

def evaluateView():
	test_X,test_Y = getEvaluateData(None)	
	sess = tf.Session()
	new_restore = tf.train.import_meta_graph('./abu_dhabi/abu_model-7500.meta')
	new_restore.restore(sess,tf.train.latest_checkpoint('./abu_dhabi/'))	
	graph = tf.get_default_graph()
	xP = graph.get_tensor_by_name("x:0")
	yP = graph.get_tensor_by_name("y:0")
	a2 = graph.get_tensor_by_name("a3:0")
	loss = tf.losses.mean_squared_error(yP,a2)

	cost,ans = sess.run([loss,a2],feed_dict={xP:test_X,yP:test_Y})
	# print(cost,ans-Y_test)
	sess.close()
	fig = plt.figure()
	d = np.array(ans) - np.array(test_Y)
	# return d
	val = hist(d,100)
	# plt.show()
	fig.savefig('plot.png')
	ans =pd.DataFrame(ans)

	ans = pd.concat([ans.reindex(),pd.DataFrame(np.array(test_Y)).reindex()],axis=1)
	ans.columns = ["Predicted","Actual"]

	return ans,cost

def getDataYuv():
	df = pd.read_csv('10_num.csv', delimiter=',')
	feature_cols = ['Price', 'Shares', 'Employee_Size', 'IPO_Expense','Revenue','NetIncome','Assets','Liabilites']
	X_df = df[feature_cols]
	pred1 = ['Closing_Price']
	y1_df = df[pred1]
	X = X_df.as_matrix(columns=None)
	y1 = y1_df.as_matrix(columns=None)
	X_train = X[:400,:]
	X_test = X[400:,:]
	Y_train = y1[:400]
	Y_test = y1[400:]
	scaler = StandardScaler()
	scaler = scaler.fit(X_train)

	train_queue = tf.train.slice_input_producer(
                                    [X_train, Y_train],
                                    shuffle=False)
	train_batch, train_label_batch = tf.train.batch(
                                    train_queue,
                                    batch_size=30
                                    #,num_threads=1
                                    )
	test_queue = tf.train.slice_input_producer(
                                    [X_test, Y_test],
                                    shuffle=False)
	test_batch, test_label_batch = tf.train.batch(
                                    train_queue,
                                    batch_size=30
                                    #,num_threads=1
                                    )
	return train_batch,train_label_batch,test_batch,test_label_batch

	

def getData():
	d = pd.read_csv('abu_final.csv',dtype=np.float64)
	# d = pd.read_csv('abu_final.csv',dtype=np.float32)
	# d = pd.concat([d,l],axis=0)
	# return d
	np.random.seed(0)
	train_i = train_test_split(range(d.shape[0]),train_size=0.8) 
	train= d.iloc[train_i[0]]
	test= d.iloc[train_i[1]]
	X_train = train.iloc[:,1:19]
	scaler = StandardScaler()
	scaler = scaler.fit(X_train)
	pickle.dump(scaler,open('normAbuScaler.p','wb'))

	X_train = scaler.transform(X_train)

	Y_train = train.iloc[:,19:]
	# return X_train,Y_train
	X_test = test.iloc[:,1:19]
	Y_test = test.iloc[:,19:]
	# return X_train,Y_train
	# return Y_train,Y_test
	train_queue = tf.train.slice_input_producer(
                                    [	X_train, Y_train],
                                    shuffle=False)
	train_batch, train_label_batch = tf.train.batch(
                                    train_queue,
                                    batch_size=200
                                    #,num_threads=1
                                    )
	test_queue = tf.train.slice_input_producer(
                                    [X_test, Y_test],
                                    shuffle=False)
	test_batch, test_label_batch = tf.train.batch(
                                    train_queue,
                                    batch_size=30
                                    #,num_threads=1
                                    )
	return train_batch,train_label_batch,test_batch,test_label_batch

def getEvaluateData(scaler):
	d = pd.read_csv('abu_final.csv',dtype=np.float32)
	scaler = pickle.load(open('normAbuScaler.p','rb'))
	np.random.seed(0)
	data_i = train_test_split(range(d.shape[0]),train_size=0.8) 
	train= d.iloc[data_i[0]]
	test= d.iloc[data_i[1]]
	X_test = test.iloc[:,1:19]
	X_train = train.iloc[:,1:19]
	X_test = scaler.transform(X_test)
	X_train = scaler.transform(X_train)
	# return X_train,Y_train


	Y_test = test.iloc[:,19:]
	Y_train = train.iloc[:,19:]



	print(X_train.shape,X_test.shape)

	return X_test,Y_test
	return X_train,Y_train

def check():
	a =evaluateView()
	val = a.to_html()
	html = val.split('</thead>')
	head  =html[0] +"</thead>"
	body = html[1]
	open("head.txt","wb").write(head)
	open("body.txt","wb").write(body)
	
	

if __name__ == "__main__":
	main()
	# mainContinue()
	# evaluate()		