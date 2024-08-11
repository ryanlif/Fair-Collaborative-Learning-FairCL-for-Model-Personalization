import warnings
warnings.filterwarnings('ignore')
import pickle
import sys
from IL import *
from BIL import *
from fairCL import *
from fairCLIF import *
if __name__ == '__main__':
    with open("../data/SSI/trainX_ssi", "rb") as fp:
        trainX = pickle.load(fp)
    with open("../data/SSI/testX_ssi", "rb") as fp:
        testX = pickle.load(fp)
    with open("../data/SSI/trainY_ssi", "rb") as fp:
        trainY = pickle.load(fp)
    with open("../data/SSI/testY_ssi", "rb") as fp:
        testY = pickle.load(fp)

    nsamples = []
    for ind in trainX:
        nsamples.append(len(ind))

    logh = open("../results/table6_real_ssi_res.txt", "w")
    sys.stdout = logh

    # Individualized learning
    IL = IndividualLearning(nsamples, trainX, trainY, testX, testY)
    IL.train()
    out_IL = IL.test()
    acc_IL_avg = np.mean(out_IL)
    acc_IL_ws = np.mean(np.sort(out_IL)[0:10])
    acc_IL_bs = np.mean(np.sort(out_IL)[::-1][0:10])
    acc_IL_var = np.var(out_IL) * 10000
    print("##############################################")
    print("Method: IL")
    print("Mean test accuracy:   %.2f %%" % (acc_IL_avg * 100))
    print("Worst 10%% accuracy:  %.2f %%" % (acc_IL_ws * 100))
    print("Best 10%% accuracy:   %.2f %%" % (acc_IL_bs * 100))
    print("Variance:             %.2f" % acc_IL_var)

    ## Collaborative learning (CL)
    ## This is a special case of FairCL (p=0, lambda=0)
    CL = FairCL(nsamples, 2, 0, trainX, trainY, testX, testY)
    CL.train(0, 10)
    out_CL = CL.test()
    acc_CL_avg = np.mean(out_CL)
    acc_CL_ws = np.mean(np.sort(out_CL)[0:10])
    acc_CL_bs = np.mean(np.sort(out_CL)[::-1][0:10])
    acc_CL_var = np.var(out_CL) * 10000
    print("##############################################")
    print("Method: CL")
    print("Mean test accuracy:   %.2f %%" % (acc_CL_avg * 100))
    print("Worst 10%% accuracy:  %.2f %%" % (acc_CL_ws * 100))
    print("Best 10%% accuracy:   %.2f %%" % (acc_CL_bs * 100))
    print("Variance:             %.2f" % acc_CL_var)

    ##  Fair Collaborative learning with Bounded Individual Loss (FairCL-BIL)
    max_BIL_avg = 0
    Z = [0.01] * len(trainX)
    BIL = BILCollaborativeLearning(nsamples, 2, 50, Z, 0.1, 0.05, trainX, trainY, testX, testY)
    BIL.train(maxIter=10)
    out_BIL = BIL.test()
    acc_BIL_avg = np.mean(out_BIL)
    acc_BIL_ws = np.mean(np.sort(out_BIL)[0:10])
    acc_BIL_bs = np.mean(np.sort(out_BIL)[::-1][0:10])
    acc_BIL_var = np.var(out_BIL) * 10000
    print("##############################################")
    print("Method: FairCL-BIL")
    print("Mean test accuracy:   %.2f %%" % (acc_BIL_avg * 100))
    print("Worst 10%% accuracy:  %.2f %%" % (acc_BIL_ws * 100))
    print("Best 10%% accuracy:   %.2f %%" % (acc_BIL_bs * 100))
    print("Variance:             %.2f" % acc_BIL_var)

    p = 0.8
    lbd = 0.05
    FL = FairCL(nsamples, 2, p, trainX, trainY, testX, testY)
    FL.train(lbd, 10)
    out_FL = FL.test()
    acc_FL_avg = np.mean(out_FL)
    acc_FL_ws = np.mean(np.sort(out_FL)[0:10])
    acc_FL_bs = np.mean(np.sort(out_FL)[::-1][0:10])
    acc_FL_var = np.var(out_FL) * 10000
    print("##############################################")
    print("Method: FairCL-RRW")
    print("Mean test accuracy:   %.2f %%" % (acc_FL_avg * 100))
    print("Worst 10%% accuracy:  %.2f %%" % (acc_FL_ws * 100))
    print("Best 10%% accuracy:   %.2f %%" % (acc_FL_bs * 100))
    print("Variance:             %.2f" % acc_FL_var)

    logh.close()


