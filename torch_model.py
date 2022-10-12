import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the network components
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (3, 3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size= (2,2), stride=(2,2), padding='same')
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = (3, 3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size= (2,2), stride=(2,2), padding='same')
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = (3, 3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size= (2,2), stride=(2,2), padding='same')
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (3, 3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size= (2,2), stride=(2,2), padding='same')
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = (3, 3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size= (2,2), stride=(2,2), padding='same')
        )
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size = (5, 5), stride=(2, 2), padding='same')
        self.deconv1_BAD = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = (3, 3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
        )
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size = (5, 5), stride=(2, 2), padding='same')
        self.deconv2_BAD = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = (3, 3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
        )
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size = (5, 5), stride=(2, 2), padding='same')
        self.deconv3_BAD = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = (3, 3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
        )
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size = (5, 5), stride=(2, 2), padding='same')
        self.deconv4_BAD = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size = (3, 3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size = (3, 3), stride=(1, 1), padding='same'),
        )
        # Define loss list
        self.loss_list_vocal = []
        self.Loss_list_vocal = []

        # Define the criterion and optimizer
        self.optim = torch.optim.Adam(self.parameters(), lr=4)
        self.crit  = nn.L1Loss()
        self.to('cuda')

    # ==============================================================================
    #   IO
    # ==============================================================================
    def load(self, path):
        if os.path.exists(path):
            print("Load the pre-trained model from {}".format(path))
            state = torch.load(path)
            for (key, obj) in state.items():
                if len(key) > 10:
                    if key[1:9] == 'oss_list':
                        setattr(self, key, obj)
            self.conv1.load_state_dict(state['conv1'])
            self.conv2.load_state_dict(state['conv2'])
            self.conv3.load_state_dict(state['conv3'])
            self.conv4.load_state_dict(state['conv4'])
            self.conv5.load_state_dict(state['conv5'])
            self.conv6.load_state_dict(state['conv6'])
            self.conv7.load_state_dict(state['conv7'])
            self.conv8.load_state_dict(state['conv8'])
            self.conv9.load_state_dict(state['conv9'])
            self.conv10.load_state_dict(state['conv10'])
            self.deconv1.load_state_dict(state['deconv1'])
            self.deconv2.load_state_dict(state['deconv2'])
            self.deconv3.load_state_dict(state['deconv3'])
            self.deconv4.load_state_dict(state['deconv4'])
            self.deconv1_BAD.load_state_dict(state['deconv1_BAD'])
            self.deconv2_BAD.load_state_dict(state['deconv2_BAD'])
            self.deconv3_BAD.load_state_dict(state['deconv3_BAD'])
            self.deconv4_BAD.load_state_dict(state['deconv4_BAD'])
            self.optim.load_state_dict(state['optim'])
        else:
            print("Pre-trained model {} is not exist...".format(path))

    def save(self, path):
        # Record the parameters
        state = {
            'conv1': self.conv1.state_dict(),
            'conv2': self.conv2.state_dict(),
            'conv3': self.conv3.state_dict(),
            'conv4': self.conv4.state_dict(),
            'conv5': self.conv5.state_dict(),
            'conv6': self.conv6.state_dict(),
            'conv7': self.conv7.state_dict(),
            'conv8': self.conv8.state_dict(),
            'conv9': self.conv9.state_dict(),
            'conv10': self.conv10.state_dict(),
            'deconv1': self.deconv1.state_dict(),
            'deconv2': self.deconv2.state_dict(),
            'deconv3': self.deconv3.state_dict(),
            'deconv4': self.deconv4.state_dict(),
            'deconv1_BAD': self.deconv1_BAD.state_dict(),
            'deconv2_BAD': self.deconv2_BAD.state_dict(),
            'deconv3_BAD': self.deconv3_BAD.state_dict(),
            'deconv4_BAD': self.deconv4_BAD.state_dict(),
        }

        # Record the optimizer and loss
        state['optim'] = self.optim.state_dict()
        for key in self.__dict__:
            if len(key) > 10:
                if key[1:9] == 'oss_list':
                    state[key] = getattr(self, key)
        torch.save(state, path)

    # ==============================================================================
    #   Set & Get
    # ==============================================================================
    def getLoss(self, normalize = False):
        loss_dict = {}
        for key in self.__dict__:
            if len(key) > 9 and key[0:9] == 'loss_list':
                if not normalize:
                    loss_dict[key] = round(getattr(self, key)[-1], 6)
                else:
                    loss_dict[key] = np.mean(getattr(self, key))
        return loss_dict

    def getLossList(self):
        loss_dict = {}
        for key in self.__dict__:
            if len(key) > 9 and key[0:9] == 'Loss_list':
                loss_dict[key] = getattr(self, key)
        return loss_dict

    def forward(self, mix):
        """
            Generate the mask for the given mixture audio spectrogram
            Arg:    mix     (torch.Tensor)  - The mixture spectrogram which size is (B, 1, 512, 128)
            Ret:    The soft mask which size is (B, 1, 512, 128)
        """
        conv1_out = self.conv1(mix)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        deconv1_out = self.deconv1(conv5_out, output_size = conv5_out.size())
        deconv1_out = self.deconv1_BAD(deconv1_out)
        conv6_out = self.conv6(torch.cat([deconv1_out, conv4_out], 1))
        deconv2_out = self.deconv2(conv6_out, output_size = conv4_out.size())
        deconv2_out = self.deconv2_BAD(deconv2_out)
        conv7_out = self.conv7(torch.cat([deconv2_out, conv3_out], 1))
        deconv3_out = self.deconv3(conv7_out, output_size = conv3_out.size())
        deconv3_out = self.deconv3_BAD(deconv3_out)
        conv8_out = self.conv8(torch.cat([deconv3_out, conv2_out], 1))
        deconv4_out = self.deconv4(conv8_out, output_size = conv2_out.size())
        deconv4_out = self.deconv4_BAD(deconv4_out)
        conv9_out = self.conv9(torch.cat([deconv4_out, conv1_out], 1))
        conv10_out = self.conv10(conv9_out)
        out = F.LEAKY_RELU(conv10_out)
        return out