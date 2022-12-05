# import packages
import numpy as np
import tensorflow as tf


class Conv(tf.keras.layers.Layer):
    """
    A convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 dilation=1, bias=True, w_init_gain='linear', is_causal=False):
        super(Conv, self).__init__()
        self.is_causal = is_causal
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = tf.keras.layers.Conv1D(filter=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           dilation_rate=dilation,
                                           use_bias=bias)

    def forward(self, signal):
        if self.is_causal:
               padding = (int((self.kernel_size - 1) * self.dilation), 0)
               signal = tf.pad(signal, padding, "CONSTANT")
        return self.conv(signal)



class Wavenet(tf.keras.layers.Layer):
    def __init__(self, pad, sd, rd, dilations0,dilations1,device):
        self.dilations1 = dilations1
        self.device=device
        sd = 512
        rd = 128
        self.sd = sd
        self.rd = rd
        self.init_filter=2
        self.field=np.sum(dilations1)+self.init_filter
        wd = 128
        print('sd rd:',sd,rd)
        self.wd=wd
        super(Wavenet, self).__init__()
        self.embedy = tf.keras.layers.Embedding(256, wd)
        self.casual = tf.keras.layers.Conv1D(filter=wd, kernel_size=self.init_filter)
        self.pad = pad
        self.ydcnn  = tf.keras.Sequential()
        self.ydense = tf.keras.Sequential()
        self.yskip = tf.keras.Sequential()

        for i, d in enumerate(self.dilations1):
            self.ydcnn.add(Conv(wd*2,kernel_size=2, dilation=d, w_init_gain='tanh', is_causal=True))
            self.yskip.add(Conv(wd, sd,w_init_gain='relu'))
            self.ydense.add(Conv(wd, wd,w_init_gain='linear'))

        self.post1 = Conv(sd, sd, bias=False, w_init_gain='relu')
        self.post2 = Conv(sd, 256, bias=False, w_init_gain='linear')

    def forward(self, y):
        y = self.embedy(y.long())
        y = y.transpose(1, 2)

        finalout = y.size(2)-(self.field-1)

        output = 0
        for i, d in enumerate(self.dilations1):
            in_act = self.ydcnn[i](y)
            in_act = in_act
            t_act = tf.math.tanh(in_act[:, :self.wd, :])
            s_act = tf.math.sigmoid(in_act[:, self.wd:, :])
            acts = t_act * s_act

            res_acts = self.ydense[i](acts)

            if i == 0:
                output = self.yskip[i](acts[:,:,-finalout:])
            else:
                output = self.yskip[i](acts[:,:,-finalout:]) + output

            y = res_acts + y[:,:,d:]

        output = tf.nn.relu(output)
        output = self.post1(output)
        output = tf.nn.relu(output)
        output = self.post2(output)
        return output


    def infer(self,queue,l = 16000*1):
        y = tf.random.uniform(1, 0, 255)
        l = int(l)
        music=tf.zeros(l)
        output = 0
        for idx in range(l):
            y = self.embedy(y.long())
            y = y.transpose(1, 2)
            for i, d in enumerate(self.dilations1):
                y = tf.concat((queue[i],y),2)
                if d == 1:
                    queue[i] = y[:,:,:1].clone()
                else:
                    queue[i] = tf.concat((queue[i][:, :, 1:], y[:, :, :1]), 2)
                in_act = self.ydcnn[i](y)
                t_act = tf.math.tanh(in_act[:, :self.wd, :])
                s_act = tf.math.sigmoid(in_act[:, self.wd:, :])
                acts = t_act * s_act

                res_acts = self.ydense[i](acts)

                if i == 0:
                    output = self.yskip[i](acts[:,:,-1:])
                else:
                    output = self.yskip[i](acts[:,:,-1:]) + output

                y = res_acts + y[:,:,d:]

            output = tf.nn.relu(output)
            output = self.post1(output)
            output = tf.nn.relu(output)
            output = self.post2(output)
            #################################################
            y = output.max(1, keepdim=True)[1].view(-1)[0]
            y = tf.math.reduce_max(output, keepdims=True)
            y = y.view(1,1)
            music[idx] = y.cpu()[0,0]
        return music