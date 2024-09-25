import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import time
from FADformer import FADformer, FADformer_mini

test_w = [256]
test_h = [256]
test_iter = [100]
test_epoch = 5

net = FADformer().to('cuda:0')
net.cuda()
net.eval()

with torch.no_grad():
    for (h, w, it) in zip(test_w, test_h, test_iter):
        rand_img = torch.rand([1, 3, h, w]).cuda()
        trace_network = torch.jit.trace(net, [rand_img])

        fps_list = []
        for i in range(test_epoch):

            torch.cuda.synchronize()
            t1 = time.time()

            for _ in range(it):
                output = trace_network(rand_img)

            torch.cuda.synchronize()
            t2 = time.time()

            fps = it / (t2 - t1)
            fps_list.append(fps)

        fps_list = sorted(fps_list)
        avg_fps = fps_list[test_epoch // 2]

        print('Latency= ' + str(1e3 / avg_fps))