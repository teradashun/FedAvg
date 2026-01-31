import argparse
def get_args():
    parser = argparse.ArgumentParser(description="MNIST pytorch")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--round", type=int, default=80, help="number of communication rounds")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for optimizer")
    parser.add_argument("--device", type=int, default=40, help="number of devices")
    parser.add_argument("--dirichlet", type=float, default=0.1, help="number of devices")
    parser.add_argument("--cohort", type=int, default=5, help="number of choice clients")
    parser.add_argument("--ite_num", type=int, default=5, help="number of iteration")

    args = parser.parse_args()
    return args