import torch
import time

def benchmark_rnn(batchsize,
                  seq_len,
                  input_dim,
                  hidden_dim,
                  is_bidirectional,
                  num_layers,
                  rnn_type,
                  activation,
                  optimizer,
                  device,
                  warmup_iterations,
                  benchmark_iterations):
    
    if rnn_type in ['LSTM', 'GRU', 'RNN']:
        rnn_class = getattr(torch.nn, rnn_type)
        rnn = rnn_class(input_dim, int(hidden_dim), num_layers=num_layers, 
                        bidirectional=bool(is_bidirectional), batch_first=True).to(device)


    if activation != 'None':
        activation_fn = getattr(torch.nn, activation)().to(device)


    if optimizer != 'None':
        criterion = torch.nn.MSELoss()
        optimizer_fn = getattr(torch.optim, optimizer)(list(rnn.parameters()), lr=0.0001)

    # Warm-up
    for _ in range(warmup_iterations):
        x = torch.randn(batchsize, seq_len, input_dim, device=device)
        rnn_out, _ = rnn(x)
        y = activation_fn(rnn_out[:, -1, :]) if activation != 'None' else rnn_out[:, -1, :]
        
        if optimizer != 'None':
            target = torch.randn(batchsize, hidden_dim * (2 if is_bidirectional else 1), device=device)
            optimizer_fn.zero_grad()
            loss = criterion(y, target)
            loss.backward()
            optimizer_fn.step()


    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(benchmark_iterations):
        x = torch.randn(batchsize, seq_len, input_dim, device=device)
        rnn_out, _ = rnn(x)
        y = activation_fn(rnn_out[:, -1, :]) if activation != 'None' else rnn_out[:, -1, :]
        
        if optimizer != 'None':
            target = torch.randn(batchsize, hidden_dim * (2 if is_bidirectional else 1), device=device)
            optimizer_fn.zero_grad()
            loss = criterion(y, target)
            loss.backward()
            optimizer_fn.step()
    torch.cuda.synchronize()
    end_time = time.time()


    return (end_time - start_time) / benchmark_iterations * 1000 # Returns time per iteration in milliseconds