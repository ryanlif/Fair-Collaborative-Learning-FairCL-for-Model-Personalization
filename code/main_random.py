import warnings
warnings.filterwarnings('ignore')
import pickle
import sys
from IL import *
from BIL import *
from fairCL import *
from fairCLIF import *
if __name__ == '__main__':
    with open("../data/random/trainX_mnist_random", "rb") as fp:
        trainX = pickle.load(fp)
    with open("../data/random/trainY_mnist_random", "rb") as fp:
        trainY = pickle.load(fp)
    with open("../data/random/testX_mnist_random", "rb") as fp:
        testX = pickle.load(fp)
    with open("../data/random/testY_mnist_random", "rb") as fp:
        testY = pickle.load(fp)

    nsamples = []
    for ind in trainX:
        nsamples.append(len(ind))

    logh = open("../results/table5_real_random_res.txt", "w")
    sys.stdout = logh
    # Individualized learning (IL)
    IL = IndividualLearning(nsamples, trainX, trainY, testX, testY)
    IL.train()
    out_IL = IL.test()
    acc_IL_avg = np.mean(out_IL)
    acc_IL_ws = np.mean(np.sort(out_IL)[0:10])
    acc_IL_bs = np.mean(np.sort(out_IL)[::-1][0:10])
    acc_IL_var = np.var(out_IL) * 10000
    print("Method: IL")
    print("Mean test accuracy:   %.2f %%" % (acc_IL_avg * 100))
    print("Worst 10%% accuracy:  %.2f %%" % (acc_IL_ws * 100))
    print("Best 10%% accuracy:   %.2f %%" % (acc_IL_bs * 100))
    print("Variance:             %.2f" % acc_IL_var)

    ## Collaborative learning (CL)
    ## This is a special case of FairCL (p=0, lambda=0)
    CL = FairCL(nsamples, 3, 0, trainX, trainY, testX, testY)
    CL.train(0, 10)
    out_CL = CL.test()
    acc_CL_avg = np.mean(out_CL)
    acc_CL_ws = np.mean(np.sort(out_CL)[0:10])
    acc_CL_bs = np.mean(np.sort(out_CL)[::-1][0:10])
    acc_CL_var = np.var(out_CL) * 10000
    print("Method: CL")
    print("Mean test accuracy:   %.2f %%" % (acc_CL_avg * 100))
    print("Worst 10%% accuracy:  %.2f %%" % (acc_CL_ws * 100))
    print("Best 10%% accuracy:   %.2f %%" % (acc_CL_bs * 100))
    print("Variance:             %.2f" % acc_CL_var)

    ##  Fair Collaborative learning with Bounded Individual Loss (FairCL-BIL)
    ## Hyperparameters which need tuning
    ## We set the bounds to be the same, and only select its value from {5, 10}.
    ## If it is too small, the performance can largely degrade, and the problem can easily become infeasible.
    max_BIL_avg = 0
    Z = [10] * 30
    BIL = BILCollaborativeLearning(nsamples, 3, 100, Z, 0.1, 0.05, trainX, trainY, testX, testY)
    BIL.train()
    out_BIL = BIL.test()
    acc_BIL_avg = np.mean(out_BIL)
    acc_BIL_ws = np.mean(np.sort(out_BIL)[0:10])
    acc_BIL_bs = np.mean(np.sort(out_BIL)[::-1][0:10])
    acc_BIL_var = np.var(out_BIL) * 10000
    print("Method: FairCL-BIL")
    print("Mean test accuracy:   %.2f %%" % (acc_BIL_avg * 100))
    print("Worst 10%% accuracy:  %.2f %%" % (acc_BIL_ws * 100))
    print("Best 10%% accuracy:   %.2f %%" % (acc_BIL_bs * 100))
    print("Variance:             %.2f" % acc_BIL_var)

    ## Fair Collaborative learning - RRW
    ## These are hyperparameters which needs tuning
    ## When p=0, lbd=0, FairCL-RRW reduces to CL.
    max_FL_avg = 0
    FL = FairCL(nsamples, 6, 1, trainX, trainY, testX, testY)
    FL.train(0.1, 10)
    out_FL = FL.test()
    acc_FL_avg = np.mean(out_FL)
    acc_FL_ws = np.mean(np.sort(out_FL)[0:10])
    acc_FL_bs = np.mean(np.sort(out_FL)[::-1][0:10])
    acc_FL_var = np.var(out_FL) * 10000
    print("Method: FairCL-RRW")
    print("Mean test accuracy:   %.2f %%" % (acc_FL_avg * 100))
    print("Worst 10%% accuracy:  %.2f %%" % (acc_FL_ws * 100))
    print("Best 10%% accuracy:   %.2f %%" % (acc_FL_bs * 100))
    print("Variance:             %.2f" % acc_FL_var)

    logh.close()


