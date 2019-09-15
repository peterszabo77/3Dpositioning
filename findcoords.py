from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
try: 
	from PIL import Image
	from PIL.Image import open 
except ImportError: 
	from Image import Image, open
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib.pyplot as plt

from parameters import *

class Wall:
	def __init__(self, coords, textureID):
		self.p1 = coords[0]
		self.p2 = coords[1]
		self.p3 = coords[2]
		self.p4 = coords[3]
		self.textureID = textureID

	def draw(self):
		glColor3f(1.0, 1.0, 0.0)
		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		#glTranslatef
		#glRotatef
		#glScalef
		setupTexture(self.textureID)
		glBegin(GL_QUADS)
		glTexCoord2f(0.0, 0.0); glVertex3fv(self.p1) #(0, 0, 0)
		glTexCoord2f(1.0, 0.0); glVertex3fv(self.p2) #(50, 0, 0)
		glTexCoord2f(1.0, 1.0); glVertex3fv(self.p3) #(50, 50, 0)
		glTexCoord2f(0.0, 1.0); glVertex3fv(self.p4) #(0, 50, 0)
		glEnd()
		glPopMatrix()

class Ground:
	def __init__(self, coords):
		self.p1 = coords[0]
		self.p2 = coords[1]
		self.p3 = coords[2]
		self.p4 = coords[3]
	def draw(self):
		glColor3f(1.0, 1.0, 1.0)
		glMatrixMode (GL_MODELVIEW)
		glPushMatrix()
		#glTranslatef
		#glRotatef
		#glScalef
		#glActiveTexture(GL_TEXTURE0)
		#glBindTexture(GL_TEXTURE_2D, 0)
		glBegin(GL_QUADS)
		glVertex3fv(self.p1) #(0, 0, 0)
		glVertex3fv(self.p2) #(50, 0, 0)
		glVertex3fv(self.p3) #(50, 50, 0)
		glVertex3fv(self.p4) #(0, 50, 0)
		glEnd()
		glPopMatrix()

class Scene:
	def __init__(self, camera):
		self.camera = camera
		self.textureIDs = loadImages(TEXTUREFILES)
		W = ROOMWIDTH
		L = ROOMLENGTH
		H = ROOMHEIGHT
		self.ground = Ground([(0,0,0),(W,0,0),(W,0,-L),(0,0,-L)])
		self.leftWall = Wall([(0,0,0),(0,0,-L),(0,H,-L),(0,H,0)], self.textureIDs[0])
		self.rightWall = Wall([(W,0,-L),(W,0,0),(W,H,0),(W,H,-L)], self.textureIDs[1])
		self.frontWall = Wall([(W,0,0),(0,0,0),(0,H,0),(W,H,0)], self.textureIDs[2])
		self.backWall = Wall([(0,0,-L),(W,0,-L),(W,H,-L),(0,H,-L)], self.textureIDs[3])
		self.snapshotidx = 0

	def draw(self):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		self.camera.setup(ROOMWIDTH*0.5, ROOMLENGTH*0.5, 45)
		self.ground.draw()
		self.leftWall.draw()
		self.rightWall.draw()
		self.frontWall.draw()
		self.backWall.draw()
		glBindTexture(GL_TEXTURE_2D, 0)
		glutSwapBuffers()

	def getRenderedImage(self, x, z, yaw, saverenders=False):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		self.camera.setup(x, z, yaw)
		self.ground.draw()
		self.leftWall.draw()
		self.rightWall.draw()
		self.frontWall.draw()
		self.backWall.draw()
		glBindTexture(GL_TEXTURE_2D, 0)
		# get image
		xp, yp, width, height = glGetIntegerv(GL_VIEWPORT)
		imagedata = glReadPixels(xp, yp, width, height, GL_RGB, GL_UNSIGNED_BYTE)
		imagearray = np.fromstring(imagedata, "uint8", count = IMAGEW*IMAGEH*IMAGECH)
		imagearray = imagearray.reshape((IMAGEW, IMAGEH, IMAGECH)) 
		imagearray = np.moveaxis(imagearray,-1,0)
		imagearray = 2*(imagearray/255)-1 # normalize to [-1,1]
		# save it
		if self.snapshotidx < N_SNAPSHOTS:
			image = Image.frombytes("RGB", (width, height), imagedata)
			image = image.transpose(Image.FLIP_TOP_BOTTOM)
			filepath = os.path.join("snapshots", 'snapshot_'+str(self.snapshotidx)+'.png')
			image.save(filepath)
			self.snapshotidx += 1
		glutSwapBuffers()
		return imagearray # shape: (CH,W,H)

class Camera:
	def __init__(self):
		self.pos = [ROOMWIDTH*0.5, CAMH, -ROOMLENGTH*0.5] # x,y,z
		self.pitch = CAMPITCH 	# look up/down (+ means up)
		self.yaw = 0		# turn right/left (+ means right)
		self.roll = 0		# not used

	def setup(self, x, z, yaw):
		self.pos = [x, CAMH, -z] # x,y,z
		self.yaw = yaw
		glMatrixMode (GL_MODELVIEW)
		glLoadIdentity()
		glRotatef(-self.pitch,1.0,0.0,0.0)
		glRotatef(self.yaw,0.0,1.0,0.0)
		glTranslatef(-self.pos[0], -self.pos[1], -self.pos[2])
		#glRotatef(cam_rotZ,0.0,0.0,1.0)

def InitGL():
	glClearColor(0.8, 0.8, 1.0, 1.0)
	glClearDepth(1.0)
	glDepthFunc(GL_LESS)
	glEnable(GL_DEPTH_TEST)
	glEnable(GL_POLYGON_SMOOTH)
	glEnable(GL_BLEND)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	glShadeModel(GL_SMOOTH)  # glShadeModel(GL_FLAT)
	resizeGLScene(IMAGEW, IMAGEH)
	# initialize texture mapping
	glEnable(GL_TEXTURE_2D)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

def loadImages(filenames):
	textureIDs = glGenTextures(len(filenames))
	for idx, filename in enumerate(filenames):
		filepath = os.path.join(TEXTUREDIR, filename)
		image = open(filepath)
		ix = image.size[0]
		iy = image.size[1]
		image = image.tobytes("raw", "RGBX", 0, -1)
		glBindTexture(GL_TEXTURE_2D, textureIDs[idx])
		glPixelStorei(GL_UNPACK_ALIGNMENT,1)
		glTexImage2D( GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image )
	return textureIDs

def setupTexture(texID):
	"""Render-time texture environment setup"""
	glEnable(GL_TEXTURE_2D)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
	glBindTexture(GL_TEXTURE_2D, texID)

def resizeGLScene (width, height):
	if height==0:
		height=1
	glViewport (0,0,width,height)
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	gluPerspective(45.0, float(width)/float(height), 0.1, 1000.0)

glutInit()
glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH)
glutInitWindowSize(IMAGEW, IMAGEH)
glutInitWindowPosition(0, 0)
wind = glutCreateWindow("3D scene")

InitGL()
camera = Camera()
scene = Scene(camera)
glutDisplayFunc(scene.draw)
glutReshapeFunc(resizeGLScene)

class CNNNetwork(nn.Module):
	def __init__(self):
		super(CNNNetwork, self).__init__()
		CONV_CHANNELS = [6,12,24]		
		self.conv1 = nn.Conv2d(in_channels=IMAGECH, out_channels=CONV_CHANNELS[0], kernel_size=3, stride=1, padding=1)
		self.conv1_bn = nn.BatchNorm2d(num_features=CONV_CHANNELS[0])
		self.conv2 = nn.Conv2d(in_channels=CONV_CHANNELS[0], out_channels=CONV_CHANNELS[1], kernel_size=5, stride=1, padding=2)
		self.conv2_bn = nn.BatchNorm2d(num_features=CONV_CHANNELS[1])
		self.conv3 = nn.Conv2d(in_channels=CONV_CHANNELS[1], out_channels=CONV_CHANNELS[2], kernel_size=5, stride=2, padding=2)
		self.conv3_bn = nn.BatchNorm2d(num_features=CONV_CHANNELS[2])
		self.fc1 = nn.Linear(int(IMAGEW*0.5*IMAGEH*0.5*CONV_CHANNELS[-1]), 120)
		self.fc2 = nn.Linear(120, 60)
		self.fc3 = nn.Linear(60, 2)

	def forward(self, x):
		x = F.leaky_relu(self.conv1(x))
		x = self.conv1_bn(x)
		x = F.leaky_relu(self.conv2(x))
		x = self.conv2_bn(x)
		x = F.leaky_relu(self.conv3(x))
		x = self.conv3_bn(x)
		x = x.view(x.size(0), -1)
		x = F.leaky_relu(self.fc1(x))
		x = F.leaky_relu(self.fc2(x))
		x = torch.sigmoid(self.fc3(x))
		return x

def set_seed_everywhere(seed, cuda):
	np.random.seed(seed)
	torch.manual_seed(seed)
	if cuda:
		torch.cuda.manual_seed_all(seed)

def compute_accuracy(y_pred, y_target):
	diff = y_target - y_pred
	summeddiffsq = torch.sum(diff**2,1)**0.5
	return torch.mean(summeddiffsq).item()

def getBatch(batchsize, device):
	xs = MINWALLDIST/ROOMWIDTH + ((ROOMWIDTH-2*MINWALLDIST)/ROOMWIDTH)*np.random.rand(batchsize)
	zs = MINWALLDIST/ROOMLENGTH + ((ROOMLENGTH-2*MINWALLDIST)/ROOMLENGTH)*np.random.rand(batchsize)
	yaws = np.random.rand(batchsize)
	imagearrays = []
	for x, z, yaw in zip(xs, zs, yaws):
		imagearray = scene.getRenderedImage(x*ROOMWIDTH, z*ROOMLENGTH, yaw*360)
		imagearrays.append(imagearray)
	batchdict = {}
	batchdict['campos'] = torch.from_numpy(np.stack((xs,zs),axis=1)).double().to(device)
	batchdict['image'] = torch.from_numpy(np.stack(imagearrays,axis=0)).double().to(device)
	return batchdict

def plotlearning(accuracy):
	fig = plt.figure()
	ax1 = fig.add_subplot(1,1,1)
	ax1.plot(range(len(accuracy)), accuracy, color='red')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	filename= 'learning.pdf'
	ax1.figure.savefig(filename)
	plt.close(fig)

def learn():
	cuda = torch.cuda.is_available()
	mydevice = torch.device("cuda" if cuda else "cpu")
	print("Using CUDA: {}".format(cuda))
	
	set_seed_everywhere(RANDOMSEED, cuda)

	mynetwork = CNNNetwork().double()
	mynetwork = mynetwork.to(mydevice)

	loss_func = nn.MSELoss()
	optimizer = optim.Adam(mynetwork.parameters(), lr=LEARNING_RATE)
	#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)

	running_train_accuracy = []
	running_eval_accuracy = [] # train and eval accuracies are the same in this case, due to the getbatch process (unlimited dataset)
	for epoch_idx in range(N_EPOCHS):
		if epoch_idx==0:
			print("Epoch: {}/{}".format(epoch_idx+1,N_EPOCHS))
		else:
			print("Epoch: {}/{}  trainacc: {}  evalacc: {}".format(epoch_idx+1,N_EPOCHS,running_train_accuracy[-1],running_eval_accuracy[-1]))
		# Iterate over training dataset
		mynetwork.train()
		batchsum_train_accuracy = 0
		for batch_idx in range(N_BATCHES):
			batch_dict = getBatch(BATCH_SIZE,mydevice)
			# the training routine is these 5 steps:
			# --------------------------------------
			# step 1. zero the gradients
			optimizer.zero_grad()
			# step 2. compute the output
			pred = mynetwork(batch_dict['image'])
			# step 3. compute the loss
			loss = loss_func(pred, batch_dict['campos'])
			# step 4. use loss to produce gradients
			loss.backward()
			# step 5. use optimizer to take gradient step
			optimizer.step()
			## ------------------------------------------
			# compute the accuracy
			batchsum_train_accuracy += compute_accuracy(pred, batch_dict['campos'])

		# Iterate over validation/test dataset
		mynetwork.eval()
		batchsum_eval_accuracy = 0
		for batch_idx in range(N_BATCHES):
			batch_dict = getBatch(BATCH_SIZE,mydevice)
			# compute the output
			pred =  mynetwork(batch_dict['image'])
			# ------------------------------------------
			# compute the accuracy
			batchsum_eval_accuracy += compute_accuracy(pred, batch_dict['campos'])

		# -----------------------------------------
		# append the running accuracies
		running_train_accuracy.append(batchsum_train_accuracy/N_BATCHES)
		running_eval_accuracy.append(batchsum_eval_accuracy/N_BATCHES)
	glutIdleFunc(None)
	plotlearning(running_eval_accuracy)
	exit(-1)

glutIdleFunc(learn)
glutMainLoop()
