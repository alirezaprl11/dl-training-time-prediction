import torch
import numpy as np
import time


def benchmark_conv(batchsize,
                   matsize,
                   kernelsize,
                   channels_in,
                   channels_out,
                   strides,
                   padding,
                   activation,
                   use_bias,
                   optimizer,
                   device,
                   warmup_iterations,
                   benchmark_iterations):
    
    conv = torch.nn.Conv2d(channels_in, channels_out, kernel_size=kernelsize, stride=strides, padding=padding, bias=use_bias).to(device)
    if activation != 'None':
        activation_fn = getattr(torch.nn, activation)().to(device)
    if optimizer != 'None':
        optimizer_fn = getattr(torch.optim, optimizer)(conv.parameters(), lr=0.0001)
        criterion = torch.nn.MSELoss()

    target_size = 0
    if padding == 'same':
        target_size = np.ceil((matsize / strides)).astype(int)
    else:
        target_size = np.ceil((matsize - kernelsize + 1) / strides).astype(int)
    
    # Warm-up
    for _ in range(warmup_iterations):
        x = torch.randn(batchsize, channels_in, matsize, matsize, device=device)
        y = activation_fn(conv(x)) if activation != 'None' else conv(x)
        if optimizer != 'None':
            target = torch.randn(batchsize, channels_out, target_size, target_size, device=device)
            optimizer_fn.zero_grad()
            loss = criterion(y, target)
            loss.backward()
            optimizer_fn.step()
    

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(benchmark_iterations):
        x = torch.randn(batchsize, channels_in, matsize, matsize, device=device)
        y = activation_fn(conv(x)) if activation != 'None' else conv(x)
        if optimizer != 'None':
            target = torch.randn(batchsize, channels_out, target_size, target_size, device=device)
            optimizer_fn.zero_grad()
            loss = criterion(y, target)
            loss.backward()
            optimizer_fn.step()
    torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / benchmark_iterations * 1000 # Return time in ms