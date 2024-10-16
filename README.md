# DL Training Time Prediction

This project aims to gather training data for predicting the training time of deep learning models on various GPUs.

## Setup

To get started, follow the steps below:

1. Clone the repository:

```bash
git clone https://github.com/alirezaprl11/dl-training-time-prediction.git
cd dl-training-time-prediction
```

2. Create and activate a virtual environment:
```bash
python -m virtualenv venv
source ./venv/bin/activate
```

3. Install the required dependencies:
```bash
python -m pip install -r requirements.txt
```


## Training Data Generation
To gather training data for a specific GPU, navigate to the `data_collection` directory and run a benchmark. For example, to test RNN models, use the following command:

```bash
cd data_collection
python -m --testRNN --num_val 20000 --repetitions 5 --log_dir ../data 
```

This command will run the benchmark for RNN models with 20,000 validation examples and repeat the process 5 times to gather data. The results will be saved in the `data` directory.
