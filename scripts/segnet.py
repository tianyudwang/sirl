import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels, input_size):
        super().__init__()

        assert input_size in [16, 64, 128]
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_size = input_size

        self.encoder_conv_00 = nn.Sequential(*[
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)])
        self.encoder_conv_01 = nn.Sequential(*[
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)])
        self.encoder_conv_10 = nn.Sequential(*[
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)])
        self.encoder_conv_11 = nn.Sequential(*[
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)])

        if self.input_size == 16:
            self.encoder_conv_02 = nn.Sequential(*[
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64)])
            self.encoder_conv_12 = nn.Sequential(*[
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128)])
            self.decoder_convtr_12 = nn.Sequential(*[
                nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128)]) 
            self.decoder_convtr_02 = nn.Sequential(*[
                nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64)]) 


        else:
            self.encoder_conv_20 = nn.Sequential(*[
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256)])
            self.encoder_conv_21 = nn.Sequential(*[
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256)])
            self.encoder_conv_22 = nn.Sequential(*[
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256)])

            self.encoder_conv_30 = nn.Sequential(*[
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512)])
            self.encoder_conv_31 = nn.Sequential(*[
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512)])
            self.encoder_conv_32 = nn.Sequential(*[
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512)])
            self.encoder_conv_40 = nn.Sequential(*[
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512)])
            self.encoder_conv_41 = nn.Sequential(*[
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512)])
            self.encoder_conv_42 = nn.Sequential(*[
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512)])

            self.decoder_convtr_42 = nn.Sequential(*[
                nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512)])
            self.decoder_convtr_41 = nn.Sequential(*[
                nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512)])
            self.decoder_convtr_40 = nn.Sequential(*[
                nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512)])
            self.decoder_convtr_32 = nn.Sequential(*[
                nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512)])
            self.decoder_convtr_31 = nn.Sequential(*[
                    nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512)])
            self.decoder_convtr_30 = nn.Sequential(*[
                    nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256)])
            self.decoder_convtr_22 = nn.Sequential(*[
                    nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256)])
            self.decoder_convtr_21 = nn.Sequential(*[
                    nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256)])
            self.decoder_convtr_20 = nn.Sequential(*[
                    nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128)])


        self.decoder_convtr_11 = nn.Sequential(*[
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)]) 
        self.decoder_convtr_10 = nn.Sequential(*[
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)]) 
        self.decoder_convtr_01 = nn.Sequential(*[
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)]) 
        self.decoder_convtr_00 = nn.Sequential(*[
            nn.ConvTranspose2d(64, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels)])


        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxunpool2d = nn.MaxUnpool2d(kernel_size=2, stride=2)


    def forward(self, input_img):

        # Encoder Stage - 1
        dim_0 = input_img.size()
        x_00 = F.relu(self.encoder_conv_00(input_img))
        x_01 = F.relu(self.encoder_conv_01(x_00))
        x_0, indices_0 = self.maxpool2d(x_01)

        # Encoder Stage - 2
        dim_1 = x_0.size()
        x_10 = F.relu(self.encoder_conv_10(x_0))
        x_11 = F.relu(self.encoder_conv_11(x_10))
        x_1, indices_1 = self.maxpool2d(x_11)

        if self.input_size != 16:
            # Encoder Stage - 3
            dim_2 = x_1.size()
            x_20 = F.relu(self.encoder_conv_20(x_1))
            x_21 = F.relu(self.encoder_conv_21(x_20))
            x_22 = F.relu(self.encoder_conv_22(x_21))
            x_2, indices_2 = self.maxpool2d(x_22) 

            # Encoder Stage - 4
            dim_3 = x_2.size()
            x_30 = F.relu(self.encoder_conv_30(x_2))
            x_31 = F.relu(self.encoder_conv_31(x_30))
            x_32 = F.relu(self.encoder_conv_32(x_31))
            x_3, indices_3 = self.maxpool2d(x_32)

            # Encoder Stage - 5
            dim_4 = x_3.size()
            x_40 = F.relu(self.encoder_conv_40(x_3))
            x_41 = F.relu(self.encoder_conv_41(x_40))
            x_42 = F.relu(self.encoder_conv_42(x_41))
            x_4, indices_4 = self.maxpool2d(x_42)   

            # Decoder Stage - 5
            x_4d = self.maxunpool2d(x_4, indices_4, output_size=dim_4)
            x_42d = F.relu(self.decoder_convtr_42(x_4d))
            x_41d = F.relu(self.decoder_convtr_41(x_42d))
            x_40d = F.relu(self.decoder_convtr_40(x_41d))
            dim_4d = x_40d.size()

            # Decoder Stage - 4
            x_3d = self.maxunpool2d(x_40d, indices_3, output_size=dim_3)
            x_32d = F.relu(self.decoder_convtr_32(x_3d))
            x_31d = F.relu(self.decoder_convtr_31(x_32d))
            x_30d = F.relu(self.decoder_convtr_30(x_31d))
            dim_3d = x_30d.size()   

            # Decoder Stage - 3
            x_2d = self.maxunpool2d(x_30d, indices_2, output_size=dim_2)
            x_22d = F.relu(self.decoder_convtr_22(x_2d))
            x_21d = F.relu(self.decoder_convtr_21(x_22d))
            x_20d = F.relu(self.decoder_convtr_20(x_21d))
            dim_2d = x_20d.size()
        else:
            x_20d = x_1

        # Decoder Stage - 2
        x_1d = self.maxunpool2d(x_20d, indices_1, output_size=dim_1)
        x_11d = F.relu(self.decoder_convtr_11(x_1d))
        x_10d = F.relu(self.decoder_convtr_10(x_11d))
        dim_1d = x_10d.size()

        # Decoder Stage - 1
        x_0d = self.maxunpool2d(x_10d, indices_0, output_size=dim_0)
        x_01d = F.relu(self.decoder_convtr_01(x_0d))
        x_00d = self.decoder_convtr_00(x_01d)
        dim_0d = x_00d.size()

        x_out = F.relu(x_00d)
        return x_out
