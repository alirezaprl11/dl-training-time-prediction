import torch
import time

class BenchmarkDense:
    def __init__(self,
                 batchsize,
                 dim_input,
                 dim_output,
                 activation,
                 optimizer,
                 device,
                 warmup_iterations,
                 benchmark_iterations):

        self.batch_size = batchsize
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.activation = activation
        self.optimizer = optimizer
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        
        self.activation_fn = None
        self.optimizer_fn = None
        self.criterion = None
        
        if activation != 'None':
            self.activation_fn = getattr(torch.nn, activation)().to(self.device)
        
        if optimizer != 'None':
            self.optimizer_fn = getattr(torch.optim, optimizer)(self.dense.parameters(), lr=0.0001)
            self.criterion = torch.nn.MSELoss()
        
        self.dense = torch.nn.Linear(self.dim_input, self.dim_output).to(self.device)
        
        
    def _forward_pass(self):
        x = torch.randn(self.batch_size, self.dim_input, device=self.device)
        y = self.dense(x)
        
        if self.activation != 'None':
            y = self.activation_fn(y)
        
        return y
    
    
    def _backward_pass(self, y):
        target = torch.randn(self.batch_size, self.dim_output, device=self.device)
        self.optimizer_fn.zero_grad()
        loss = self.criterion(y, target)
        loss.backward()
        self.optimizer_fn.step()
        
        return loss
        


    def run_benchmark(self):
        if self.activation != 'None':
            activation_fn = getattr(torch.nn, self.activation)().to(self.device)
        if self.optimizer != 'None':
            optimizer_fn = getattr(torch.optim, self.optimizer)(self.dense.parameters(), lr=0.0001)
            criterion = torch.nn.MSELoss()
            
        # Warm-up
        for _ in range(self.warmup_iterations):
            y = self._forward_pass()
            if self.optimizer != 'None':
                self._backward_pass(y)

        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(self.benchmark_iterations):
            y = self._forward_pass()
            if self.optimizer != 'None':
                self._backward_pass(y)
        torch.cuda.synchronize()
        end_time = time.time()

        return (end_time - start_time) / self.benchmark_iterations * 1000