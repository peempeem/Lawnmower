import time

from gnet import GNet
from dataset import ImageDim

img_dims = ImageDim(320, 240, 3)
classes = "./classes.txt"
colors = "./colors.txt"

trainDB = "../ImageDB/aug"
valDB = "../ImageDB/raw"
testDB = "../ImageDB/raw/images"

load_model = "./model2.pth.tar"

if __name__ == "__main__":
	net = GNet(img_dims, classes, colors, batch_size=60, workers=12, load_model=load_model)
	#net.summary()
	#net.train(trainDB, valDB, learning_rate=5e-4, epochs=30, saveas=load_model, shuffle=True)
	net.toONNX('gnet.onnx')
	#images, masks = net.predict_folder(testDB)
	#net.show_images(images, masks, delay=2000)
