import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
from model import _netG, _netD


nz = 100
batch_size = 128


# load saved model parameters
state_dict = torch.load("./output/mnist_out/netG_epoch_10.pth")


# create model and populate state
G = _netG(ngpu=1, nz=100, ngf=64, nc=1)
G.load_state_dict(state_dict)


def generate_image():
    # generate new images
    z = Variable(torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1))
    generated_batch = G.forward(z)

    # save generated images
    image = vutils.make_grid(generated_batch.data)
    image = image.numpy()
    image = np.transpose(image, axes=(1, 2, 0))

    return image


# create image plot
img_plot = plt.imshow(generate_image())


# create button handler that will call generate_image
def button_event(event):
    img_plot.set_data(generate_image())
    img_plot.autoscale()
    plt.draw()


# connect button handler to plot
plt.connect('button_press_event', button_event)

# show plot
plt.show()


