import warnings
warnings.filterwarnings('ignore')
import pickle
import sys
import pandas as pd
from IL import *
from BIL import *
from fairCL import *
from fairCLIF import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

if __name__ == '__main__':
    N = 100
    epochs = {3: 10, 5: 10, 10: 3}
    # # The parameters selected by cross validation are directly used to save the running time.
    # params_BIL = {5:{3:5, 5:6, 10:7}, 10:{3:5, 5:10, 10:10}, 100:{3:3, 5:8, 10:8}}
    # params_RRW = {5:{3:{"p":0.7, "lambda":0.01}, 5:{"p": 1, "lambda": 0.0005}, 10:{"p": 0.8, "lambda": 0.005}}, 10:{3:{"p": 0.5, "lambda": 0.005}, 5:{"p":0.7, "lambda":0.0005}, 10:{"p":0.5, "lambda":0.01}}, 100:{3:{"p": 0.7, "lambda":0.005 }, 5:{"p":0.8, "lambda": 0.01 }, 10:{"p": 0.8, "lambda": 0.01}}}
    # params_RRWIF = {5:{3:{"p":0.7, "lambda":0.01}, 5:{"p": 1, "lambda": 0.0001}, 10:{"p": 0.1, "lambda": 0}}, 10:{3:{"p": 0, "lambda": 0.0001}, 5:{"p":0.7, "lambda":0.0005}, 10:{"p":0.9, "lambda":0.0005}},100:{3:{"p":0.2, "lambda": 0}, 5:{"p":1, "lambda": 0.005}, 10:{"p": 0.8, "lambda": 0.01}}}

    # The parameters selected by cross validation are directly used to save the running time.
    params_BIL = {5: {3: 5, 5: 6, 10: 15}, 10: {3: 5, 5: 7, 10: 2}, 100: {3: 8, 5: 8, 10: 5}}
    params_RRW = {5: {3: {"p": 0.7, "lambda": 0.01}, 5: {"p": 1, "lambda": 0.0005}, 10: {"p": 0.8, "lambda": 0.005}},
                  10: {3: {"p": 0.5, "lambda": 0.005}, 5: {"p": 0.7, "lambda": 0.0005}, 10: {"p": 0.5, "lambda": 0.01}},
                  100: {3: {"p": 0.3, "lambda": 0.001}, 5: {"p": 0.7, "lambda": 0.01}, 10: {"p": 1.0, "lambda": 0.01}}}
    params_RRWIF = {5: {3: {"p": 0.4, "lambda": 0.005}, 5: {"p": 0, "lambda": 0.0005}, 10: {"p": 0.2, "lambda": 0.0005}},
                    10: {3: {"p": 0, "lambda": 0.0005}, 5: {"p": 0.7, "lambda": 0.0005},
                         10: {"p": 0.9, "lambda": 0.0005}},
                    100: {3: {"p": 0.2, "lambda": 0}, 5: {"p": 1, "lambda": 0.005}, 10: {"p": 0.7, "lambda": 0.01}}}
    res_save = {5:{3:{}, 5:{}, 10:{}}, 10:{3:{},5:{},10:{}}, 100:{3:{},5:{},10:{}}}

    tbcnt = 0
    for d in [5, 10, 100]:
        tbcnt += 1
        filename = "../results/table"+str(tbcnt)+"_d_"+str(d)+"_simulation_res.txt"
        logg = open(filename, "a")
        sys.stdout = logg

        with open("../data/simulation/trainX_d" + str(d) + "_sim", "rb") as fp:
            trainX = pickle.load(fp)
        with open("../data/simulation/trainY_d" + str(d) + "_sim", "rb") as fp:
            trainY = pickle.load(fp)
        with open("../data/simulation/testX_d" + str(d) + "_sim", "rb") as fp:
            testX = pickle.load(fp)
        with open("../data/simulation/testY_d" + str(d) + "_sim", "rb") as fp:
            testY = pickle.load(fp)
        with open("../data/simulation/similarity_d" + str(d) + "_sim", "rb") as fp:
            similarity = pickle.load(fp)
        nsamples = []
        for ind in trainX:
            nsamples.append(len(ind))

        # Individiualized learning
        IL = IndividualLearning(nsamples, trainX, trainY, testX, testY)
        IL.train()
        out_IL = IL.test()
        acc_IL_avg = np.mean(out_IL)
        acc_IL_ws = np.mean(np.sort(out_IL)[0:10])
        acc_IL_bs = np.mean(np.sort(out_IL)[::-1][0:10])
        acc_IL_var = np.var(out_IL) * 10000
        res_save[d][3]["IL"] = out_IL
        res_save[d][5]["IL"] = out_IL
        res_save[d][10]["IL"] = out_IL
        print("##############################################")
        print("Method: IL")
        print("Mean test accuracy:   %.2f %%" % (acc_IL_avg * 100))
        print("Worst 10%% accuracy:  %.2f %%" % (acc_IL_ws * 100))
        print("Best 10%% accuracy:   %.2f %%" % (acc_IL_bs * 100))
        print("Variance:             %.2f" % (acc_IL_var))

        for K in [3, 5, 10]:
            print("################################################")
            print("#################### K = %d ####################" % K)
            print("################################################")
            maxit = epochs[K]

            # Collaborative learning (CL)
            ## This is a special case of FairCL (p=0, lambda=0)
            CL = FairCL(nsamples, K, 0, trainX, trainY, testX, testY)
            CL.train(0, maxit)
            out_CL = CL.test()
            acc_CL_avg = np.mean(out_CL)
            acc_CL_ws = np.mean(np.sort(out_CL)[0:10])
            acc_CL_bs = np.mean(np.sort(out_CL)[::-1][0:10])
            acc_CL_var = np.var(out_CL) * 10000
            res_save[d][K]["CL"] = out_CL

            ##  Fair Collaborative learning with Bounded Individual Loss (FairCL-BIL)
            ## Hyperparameters which need tuning
            ## We set the bounds to be the same, and only select its value from {5, 10}.
            ## If it is too small, the performance can largely degrade, and the problem can easily become infeasible.
            Bd = 100
            nu = 0.1
            eta = 0.05
            max_BIL_avg = 0
            zbstar = params_BIL[d][K]
            Z = [zbstar] * 100
            BIL = BILCollaborativeLearning(nsamples, K, Bd, Z, nu, eta, trainX, trainY, testX, testY)
            BIL.train(maxit)
            out_BIL = BIL.test()
            acc_BIL_avg = np.mean(out_BIL)
            acc_BIL_ws = np.mean(np.sort(out_BIL)[0:10])
            acc_BIL_bs = np.mean(np.sort(out_BIL)[::-1][0:10])
            acc_BIL_var = np.var(out_BIL) * 10000
            res_save[d][K]["BIL"] = out_BIL

            # Fair Collaborative learning - RRW
            ## These are hyperparameters which needs tuning
            ## When p=0, lbd=0, FairCL-RRW reduces to CL.
            max_FL_avg = 0
            pstar = params_RRW[d][K]["p"]
            lbdstar = params_RRW[d][K]["lambda"]
            FL = FairCL(nsamples, K, pstar, trainX, trainY, testX, testY)
            FL.train(lbdstar, maxit)
            out_FL = FL.test()
            acc_FL_avg = np.mean(out_FL)
            acc_FL_ws = np.mean(np.sort(out_FL)[0:10])
            acc_FL_bs = np.mean(np.sort(out_FL)[::-1][0:10])
            acc_FL_var = np.var(out_FL) * 10000
            res_save[d][K]["RRW"] = out_FL

            ## FairCL-IF
            ## This is a special case of FairCL-RRW+IF
            IFCL = FairCLIF(nsamples, similarity, K, 0, trainX, trainY, testX, testY)
            IFCL.train(0, maxit)
            out_IFCL = IFCL.test()
            acc_IFCL_avg = np.mean(out_IFCL)
            acc_IFCL_ws = np.mean(np.sort(out_IFCL)[0:10])
            acc_IFCL_bs = np.mean(np.sort(out_IFCL)[::-1][0:10])
            acc_IFCL_var = np.var(out_IFCL) * 10000
            res_save[d][K]["IF"] = out_IFCL

            # FairCL-RRW+IF
            # These are hyperparameters which needs tuning
            max_FLIF_avg = 0
            pifstar = params_RRWIF[d][K]["p"]
            lbdifstar = params_RRWIF[d][K]["lambda"]
            FLIF = FairCLIF(nsamples, similarity, K, pifstar, trainX, trainY, testX, testY)
            FLIF.train(lbdifstar, maxit)
            out_FLIF = FLIF.test()
            acc_FLIF_avg = np.mean(out_FLIF)
            acc_FLIF_ws = np.mean(np.sort(out_FLIF)[0:10])
            acc_FLIF_bs = np.mean(np.sort(out_FLIF)[::-1][0:10])
            acc_FLIF_var = np.var(out_FLIF) * 10000
            res_save[d][K]["RRWIF"] = out_FLIF

            print("##############################################")
            print("Method: CL")
            print("Mean test accuracy:   %.2f %%" % (acc_CL_avg * 100))
            print("Worst 10%% accuracy:  %.2f %%" % (acc_CL_ws * 100))
            print("Best 10%% accuracy:   %.2f %%" % (acc_CL_bs * 100))
            print("Variance:             %.2f" % (acc_CL_var))
            print("##############################################")
            print("Method: FairCL-BIL")
            print("Mean test accuracy:   %.2f %%" % (acc_BIL_avg * 100))
            print("Worst 10%% accuracy:  %.2f %%" % (acc_BIL_ws * 100))
            print("Best 10%% accuracy:   %.2f %%" % (acc_BIL_bs * 100))
            print("Variance:             %.2f" % (acc_BIL_var))
            print("##############################################")
            print("Method: FairCL-RRW")
            print("Mean test accuracy:   %.2f %%" % (acc_FL_avg * 100))
            print("Worst 10%% accuracy:  %.2f %%" % (acc_FL_ws * 100))
            print("Best 10%% accuracy:   %.2f %%" % (acc_FL_bs * 100))
            print("Variance:             %.2f" % (acc_FL_var))
            print("##############################################")
            print("Method: FairCL-IF")
            print("Mean test accuracy:   %.2f %%" % (acc_IFCL_avg * 100))
            print("Worst 10%% accuracy:  %.2f %%" % (acc_IFCL_ws * 100))
            print("Best 10%% accuracy:   %.2f %%" % (acc_IFCL_bs * 100))
            print("Variance:             %.2f" % (acc_IFCL_var))
            print("##############################################")
            print("Method: FairCL-RRW+IF")
            print("Mean test accuracy:   %.2f %%" % (acc_FLIF_avg * 100))
            print("Worst 10%% accuracy:  %.2f %%" % (acc_FLIF_ws * 100))
            print("Best 10%% accuracy:   %.2f %%" % (acc_FLIF_bs * 100))
            print("Variance:             %.2f" % (acc_FLIF_var))
        logg.close()

    ## Figure 2
    K = 3
    d = 10
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    df = pd.DataFrame({'Method': [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100 + [5] * 100})
    df["Accuracy"] = res_save[d][K]["IL"] + res_save[d][K]["CL"] + res_save[d][K]["IF"] + res_save[d][K]["BIL"] + \
                     res_save[d][K]["RRW"] + res_save[d][K]["RRWIF"]
    df["Method"] = df['Method'].map(
        {0: 'IL', 1: 'CL', 2: 'FairCL-IF', 3: 'FairCL-BIL', 4: 'FairCL-RRW', 5: 'FairCL-RRW+IF'})
    colors = ["#ABB2B9", "#50C878", "#5CB3FF", "#A3E4D7", "#EC7063", "#FCD299"]
    sns.set_palette(sns.color_palette(colors))
    sns.boxplot(x="Method", y="Accuracy", data=df, showfliers=False,
                medianprops=dict(color="black", linewidth=2, linestyle='-'))
    sns.stripplot(x="Method", y="Accuracy", hue="Method", data=df, jitter=0.2, linewidth=0.5, edgecolor='black', size=5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tick_params(bottom=False, labelbottom=False)
    plt.ylim(bottom=0.2)
    plt.savefig("../results/fig2" + "_comparison_d10.pdf", bbox_inches='tight')
    plt.show()


    ## Figure 3
    d = 10
    for K in [3, 5, 10]:
        df = pd.DataFrame({'Method': [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100 + [5] * 100})
        df["Accuracy"] = res_save[d][K]["IL"] + res_save[d][K]["CL"] + res_save[d][K]["IF"] + res_save[d][K]["BIL"] + \
                         res_save[d][K]["RRW"] + res_save[d][K]["RRWIF"]
        df["Method"] = df['Method'].map(
            {0: 'IL', 1: 'CL', 2: 'FairCL-IF', 3: 'FairCL-BIL', 4: 'FairCL-RRW', 5: 'FairCL-RRW+IF'})
        df["Accuracy"] = df["Accuracy"] * 100
        sns.set_style("white")
        fig, ax = plt.subplots(1, 1)
        colors = ["#F39C12", "#3498DB"]
        sns.set_palette(sns.color_palette(colors))
        ndf = df[df["Method"].isin(['IL', 'CL'])]
        bins = np.linspace(0, 100, 100)
        g = sns.histplot(data=ndf, x="Accuracy", kde=True, bins=40, alpha=0.5, hue="Method", legend=True)
        sns.despine(top=True, right=True)
        sns.move_legend(ax, "upper left")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Number of Individuals")
        plt.savefig("../results/fig3_K" + str(K) + ".pdf", bbox_inches='tight')
        plt.xlim(30, 100)
        plt.show()

    ## Figure 4
    K = 3
    d = 100

    sns.set_style("white")
    fig, ax = plt.subplots(1, 1)
    colors = ["#F39C12", "#3498DB"]
    sns.set_palette(sns.color_palette(colors))
    ndf = df[df["Method"].isin(['IL', 'CL'])]
    bins = np.linspace(0, 100, 100)
    g = sns.histplot(data=ndf, x="Accuracy", kde=True, bins=40, alpha=0.5, hue="Method", legend=True)
    sns.despine(top=True, right=True)
    sns.move_legend(ax, "upper left")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Number of Individuals")
    plt.savefig("../results/fig4a.pdf", bbox_inches='tight')
    plt.xlim(30, 100)
    plt.show()

    sns.set_style("white")
    fig, ax = plt.subplots(1, 1)
    colors = ["#3498DB", "#E380D6"]
    sns.set_palette(sns.color_palette(colors))
    ndf = df[df["Method"].isin(['CL', 'FairCL-BIL'])]
    bins = np.linspace(0, 100, 100)
    g = sns.histplot(data=ndf, x="Accuracy", kde=True, bins=40, alpha=0.6, hue="Method", legend=True)
    sns.despine(top=True, right=True)
    sns.move_legend(ax, "upper left")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Number of Individuals")
    plt.savefig("../results/fig4b.pdf", bbox_inches='tight')
    plt.xlim(30, 100)
    plt.show()

    sns.set_style("white")
    fig, ax = plt.subplots(1, 1)
    colors = ["#3498DB", "#9B421B"]
    sns.set_palette(sns.color_palette(colors))
    ndf = df[df["Method"].isin(['CL', 'FairCL-RRW'])]
    bins = np.linspace(0, 100, 100)
    g = sns.histplot(data=ndf, x="Accuracy", kde=True, bins=40, alpha=0.6, hue="Method", legend=True)
    sns.despine(top=True, right=True)
    sns.move_legend(ax, "upper left")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Number of Individuals")
    plt.savefig("../results/fig4c.pdf", bbox_inches='tight')
    plt.xlim(30, 100)
    plt.show()
