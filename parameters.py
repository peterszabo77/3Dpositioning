SNAPSHOTSDIR = 'snapshots'
N_SNAPSHOTS = 100

RANDOMSEED = 1337

# textures
TEXTUREDIR = 'wallpapers'
TEXTUREFILES = ['wp3left.jpg','wp3right.jpg','wp3front.jpg','wp3back.jpg']

# room
ROOMWIDTH = 500
ROOMLENGTH = 600
ROOMHEIGHT = 400

# camera
CAMPITCH = 0 #20
CAMH = ROOMHEIGHT*0.2
MINWALLDIST = 10

# for CNNNetwork
IMAGEW = 90
IMAGEH = 60
IMAGECH = 3 # number of channels/features (R/G/B)

N_EPOCHS = 1000
N_BATCHES = 1
BATCH_SIZE = 100
LEARNING_RATE = 0.001

from math import floor
def get2dConvOutsize(Hin,Win,Cin,kernelsize,stride,padding):
	Hout=floor((Hin+2*padding-(kernelsize-1)-1)/stride+1)
	Wout=floor((Win+2*padding-(kernelsize-1)-1)/stride+1)
	print("IN-  H: {}  W: {}  CH: {}  ks: {}  stride: {}  padding: {}".format(Hin,Win,Cin,kernelsize,stride,padding))
	print("OUT-  H: {}  W: {}".format(Hout,Wout))
