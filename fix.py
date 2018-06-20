import pickle

d= pickle.load(open('/mnt/data1/images/ugan_datasets/anything/labels.pkl', 'rb'))
print(d)
#pkl = open('/mnt/data1/images/ugan_datasets/anything/labels.pkl', 'wb')
#data = pickle.dumps([d,a,b,[]])
#pkl.write(data)
#pkl.close()
