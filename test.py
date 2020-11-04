from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import os

def classify(img_file):
	img_name = img_file
	test_img = image.load_img(img_name, target_size=(64,64))

	test_img = image.img_to_array(test_img)
	test_img = np.expand_dims(test_img, axis=0)
	result = model.predict(test_img)

	if result[0][0] == 1:
		predict = 'Thanos'
	else:
		predict = 'Grimos'
	print(predict,img_name)
	return



json_file = open('model.json','r')
load_model_json = json_file.read()
json_file.close()
model = model_from_json(load_model_json)
model.load_weights('model.h5')
print('Loaded model from disk')


path = "./Dataset/test/"
files = []
for r, d, f in os.walk(path+'_grimos'):
	for file in f:
		if '.jpg' in file:
			files.append(os.path.join(r,file))

for f in files:
	classify(f)

print('_'*50)
path = "./Dataset/test/"
files = []
for r, d, f in os.walk(path+'_thanos'):
	for file in f:
		if '.jpg' in file:
			files.append(os.path.join(r,file))

for f in files:
	classify(f)
