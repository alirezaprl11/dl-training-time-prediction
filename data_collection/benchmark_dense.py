import torch
import time

def benchmark_dense(batchsize,
                    dim_input,
                    dim_output,
                    activation,
                    optimizer,
                    device,
                    warmup_iterations,
                    benchmark_iterations):

    dense = torch.nn.Linear(dim_input, dim_output).to(device)
    if activation != 'None':
        activation_fn = getattr(torch.nn, activation)().to(device)
    if optimizer != 'None':
        optimizer_fn = getattr(torch.optim, optimizer)(dense.parameters(), lr=0.0001)
        criterion = torch.nn.MSELoss()

    
    # Warm-up
    for _ in range(warmup_iterations):
        x = torch.randn(batchsize, dim_input, device=device)
        y = activation_fn(dense(x)) if activation != 'None' else dense(x)
        if optimizer != 'None':
            target = torch.randn(batchsize, dim_output, device=device)
            optimizer_fn.zero_grad()
            loss = criterion(y, target)
            loss.backward()
            optimizer_fn.step()
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(benchmark_iterations):
        x = torch.randn(batchsize, dim_input, device=device)
        y = activation_fn(dense(x)) if activation != 'None' else dense(x)
        if optimizer != 'None':
            target = torch.randn(batchsize, dim_output, device=device)
            optimizer_fn.zero_grad()
            loss = criterion(y, target)
            loss.backward()
            optimizer_fn.step()
    torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / benchmark_iterations * 1000 # Return time in ms