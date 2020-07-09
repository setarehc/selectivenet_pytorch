import torch

import numpy as np

def post_calibrate(model, data_loader, coverage):
    with torch.autograd.no_grad():
        for i, (x,t) in enumerate(data_loader):
            model.eval()
            x = x.to('cuda', non_blocking=True)
            t = t.to('cuda', non_blocking=True)
            # forward
            out_class, out_select, out_aux = model(x)
    threshold = np.percentile(out_select.cpu().detach().numpy(), 100 - 100 * coverage)
    print('>>> Threshold found is : ', threshold)
    return threshold