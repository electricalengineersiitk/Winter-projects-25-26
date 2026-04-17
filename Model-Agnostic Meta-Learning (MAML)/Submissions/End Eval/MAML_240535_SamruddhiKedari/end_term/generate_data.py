import numpy as np #Importing required library for random number generation

NUM_TRAIN_TASKS = 100 #Total tasks for training and testing
NUM_TEST_TASKS = 20

FEATURE_DIM = 10   # signal vector dimension
CHANNEL_DIM = 10   # channel vector dimension

SUPPORT_SIZE = 10 #data used for adaptation
QUERY_SIZE = 60 #used for evaluation

SAVE_PATH = "tasks.npz" #Path to save dadaset


# GENERATE ONE DATA 

def generate_task():

    # Task variations (wireless environment), creating different environments
    snr_db = np.random.uniform(0, 20) #Getting different signal to noise ratios
    noise_std = 10 ** (-snr_db / 20) #convert to noise standard deviation

    # random channel vector
    H = np.random.randn(CHANNEL_DIM)

    # SUPPORT SET
    Xs = np.random.randn(SUPPORT_SIZE, FEATURE_DIM)
    Ys = Xs @ H + noise_std * np.random.randn(SUPPORT_SIZE) #Transmitted signal passing through channel plus noise

    # QUERY SET
    Xq = np.random.randn(QUERY_SIZE, FEATURE_DIM)
    Yq = Xq @ H + noise_std * np.random.randn(QUERY_SIZE) #Similarly transmitted signal used for evaluation

    return {
        "X_support": Xs,  #Getting support and query dataset
        "Y_support": Ys,
        "X_query": Xq,
        "Y_query": Yq
    }


# MAIN

def main():
    train_tasks = [generate_task() for _ in range(NUM_TRAIN_TASKS)] #Getting training and testing data
    test_tasks = [generate_task() for _ in range(NUM_TEST_TASKS)]

    np.savez(SAVE_PATH, train=train_tasks, test=test_tasks) #Saving dataset
    print("Dataset saved to tasks.npz")


if __name__ == "__main__":
    main()