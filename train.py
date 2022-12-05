import numpy as np
import tensorflow as tf
from wavenet import Wavenet

model = Wavenet(pad=field, sd=skipDim, rd=residualDim, dilations0=dilations0,dilations1=dilations1)
criterion = tf.keras.losses.CategoricalCrossentropy() # nn.CrossEntropyLoss()
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3, weight_decay=1e-5) # optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

def test(epoch):
    model.eval()
    start_time = time.time()
    with tf.stop_gradient(): # equivalent of torch.no_grad()?
        for iloader, y, queue in loadval:
            iloader=iloader.item()
            print(music[:1000])
            ans0 = mu_law_decode(music.numpy().astype('int'))

            if not os.path.exists('vsCorpus/'): 
                os.makedirs('vsCorpus/')
            
            write(savemusic.format(epoch), sample_rate, ans0)
            print('test stored done', np.round(time.time() - start_time))


def train(epoch):
    for iloader, ytrain in loadtr:
        startx = np.random.randint(0,sampleSize) #make results worse, biggest reason
        startx = 0
        idx = np.arange(startx + field, ytrain.shape[-1] - field - sampleSize, sampleSize)
        lens = 10
        idx = idx[:lens]

        cnt, aveloss,aveacc = 0, 0, 0
        start_time = time.time()
        model.train()

        for i, ind in enumerate(idx):
            optimizer.zero_grad()
            target0 = ytrain[:, ind - field:ind + sampleSize - 1].to(device)
            target1 = ytrain[:, ind:ind + sampleSize].to(device)
            output = model(target0)
            a = output.max(dim=1, keepdim=True)[1].view(-1)
            b = target1.view(-1)
            assert (a.shape[0] == b.shape[0])
            aveacc += float(float(torch.sum(a.long() == b.long())) / float(a.shape[0]))
            loss = criterion(output, target1)
            loss.backward()
            optimizer.step()
            aveloss+=float(loss)
            if(float(loss) > 10):print(float(loss))
            cnt+=1
            lossrecord.append(float(loss))
            global sampleCnt
            sampleCnt+=1
            
            if sampleCnt % 10000 == 0 and sampleCnt > 0:
                for param in optimizer.param_groups:
                    param['lr'] *= 0.98
            
        print('loss for train:{:.4f},acc:{:.4f},num{},epoch{},({:.3f} sec/step)'.format(aveloss / cnt,aveacc/cnt, iloader, epoch,time.time() - start_time))

    if not os.path.exists('lossRecord/'):
        os.makedirs('lossRecord/')
    
    with open("lossRecord/" + lossname, "w") as f:
        for s in lossrecord:
            f.write(str(s) + "\n")

    if not os.path.exists('model/'): 
        os.makedirs('model/')

    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}

    torch.save(state, resumefile) # what is equivalent of torch save in tensorflow?
    print('write finish')
    print('epoch finished')

print('training...')
for epoch in range(100000):
    train(epoch + start_epoch)
    if (epoch + start_epoch) % 4 == 0 and (epoch + start_epoch) > 0: test(epoch + start_epoch)