import numpy as np
import cvxpy as cp
import math
from statistics import mean
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from numpy.linalg import pinv
from scipy.linalg import block_diag


class BILCollaborativeLearning:
    def __init__(self, nsample, K, Bd, Z, nu, eta, trainX, trainY, testX, testY):
        #         super().__init__(X, Y)
        #         self.Lambda = Lambda
        self.nsample = nsample
        self.N = len(self.nsample)
        self.K = K
        self.Bd = Bd
        self.Z = Z
        self.nu = nu
        self.eta = eta
        self.Xtr = trainX
        #         self.Utr = u_train
        self.Ytr = trainY
        self.Xte = testX
        #         self.Ute = u_test
        self.Yte = testY
        self.d = len(trainX[0][0])
        self.Q0, self.B0 = self.InitializeQ(self.Xtr, self.Ytr)
        #         self.C0 = self.InitializeC(self.Q0)
        # method="uniform": using uniform distribution
        # method="mindis": minimizing \sum_i||\beta_i-Qc_i||^2 with |c_i|==1
        # method="fit": find c_i for beta_i=Qc_i with |c_i|==1
        self.C0 = self.InitializeC(self.Q0, self.B0, "mindis")
        self.theta0 = [0] * self.N
        self.params = {"Q": self.Q0, "C": self.C0, "L": []}

    def InitializeQ(self, X, Y):
        initb = []
        coefps = []
        for xdata, ydata in zip(X, Y):
            Xnp = np.array(xdata)
            Ynp = np.array(ydata)
            initmodel = LogisticRegression(solver='liblinear', random_state=0)
            initmodel.fit(Xnp, Ynp)
            coef = np.append(initmodel.coef_, initmodel.intercept_)
            coef = coef.tolist()
            coefps.append(coef)
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(np.array(coefps))
        Qt = kmeans.cluster_centers_
        return np.array(Qt.T), np.array(coefps)

    def InitializeC(self, Q, B, method="mindis"):
        if method == "uniform":
            C0 = 1.0 / self.K * np.ones((self.K, self.N))
        elif method == "mindis":
            ci = cp.Variable(self.N * self.K, nonneg=True)
            Qi = np.kron(np.eye(self.N), Q)
            bi = B.flatten()
            onem = np.kron(np.eye(self.N), np.ones(self.K))
            cost = cp.sum_squares(Qi @ ci - bi)
            prob = cp.Problem(cp.Minimize(cost), [onem @ ci == np.ones(self.N)])
            prob.solve()
            civ = ci.value
            C0 = civ.reshape(self.N, self.K).T
        else:
            Cinv = np.matmul(pinv(Q), B.T)
            C0 = normalize(Cinv, axis=1, norm='l1')
        return C0

    def updateQ(self, C, L):
        Xc = []
        Yc = []
        coef = []
        qv = cp.Variable((self.d + 1) * self.K)
        for i in range(self.N):
            Uxtr = self.Xtr[i]
            Uytr = self.Ytr[i]
            ni = self.nsample[i]
            for j, xij in enumerate(Uxtr):
                xijnew = np.kron(np.eye(self.K), xij + [1.0])
                Xc.append(list(np.matmul(xijnew.T, C[:, i])))
                Yc.append(Uytr[j])
                coef.append((L[i] + 1 / self.N) / ni)
        Xc = np.array(Xc)
        Yc = np.array(Yc)
        log_likelihood = cp.sum(cp.multiply(coef, cp.multiply(Yc, Xc @ qv) - cp.logistic(Xc @ qv)))
        problem = cp.Problem(cp.Maximize(log_likelihood / self.N))
        problem.solve(solver="SCS", max_iters=500)
        qval = qv.value
        Q = qval.reshape(self.K, (self.d + 1)).T
        self.params["Q"] = Q
        return Q

    def updateC(self, Q, L):
        Bxq = []
        Yq = []
        coef = []
        cv = cp.Variable(self.N * self.K, nonneg=True)
        for i in range(self.N):
            Uxtr = self.Xtr[i]
            Uytr = self.Ytr[i]
            ni = self.nsample[i]
            Uxq = []
            for j, xij in enumerate(Uxtr):
                Uxq.append(list(np.matmul(xij + [1.0], Q)))
                Yq.append(Uytr[j])
                coef.append((L[i] + 1 / self.N) / ni)
            Bxq.append(Uxq)
        Xq = np.array(block_diag(*Bxq))
        log_likelihood = cp.sum(cp.multiply(coef, cp.multiply(Yq, Xq @ cv) - cp.logistic(Xq @ cv)))
        onem = np.kron(np.eye(self.N, dtype=int), np.ones(self.K))
        problem = cp.Problem(cp.Maximize(log_likelihood / self.N), [onem @ cv == np.ones(self.N)])
        problem.solve(solver="SCS", max_iters=500)
        cval = cv.value
        C = cval.reshape(self.N, self.K).T
        self.params["C"] = C
        return C

    def updateL(self, Q, C, B, zeta):
        coef = []
        lv = cp.Variable(self.N, nonneg=True)
        for i in range(self.N):
            Uxtr = self.Xtr[i]
            Uytr = self.Ytr[i]
            ni = self.nsample[i]
            ci = np.matmul(Q, C[:, i])
            coefi = 0
            for j, xij in enumerate(Uxtr):
                xQC = np.matmul(xij + [1.0], ci)
                yj = Uytr[j]
                if xQC > 8:   # Avoid numerical issues
                    coefi += xQC - yj * xQC
                else:
                    coefi += np.log(1 + np.exp(xQC)) - yj * xQC
            coefi = coefi / ni - zeta[i]
            coef.append(coefi)
        coef = np.array(coef)
        lossfunc = cp.sum(cp.multiply(coef, lv))
        onem = np.kron(np.eye(self.N, dtype=int), np.ones(self.K))
        problem = cp.Problem(cp.Maximize(lossfunc / self.N), [cp.norm(lv, 1) <= B])
        problem.solve(solver="SCS", max_iters=500)
        L = lv.value
        Llist = self.params["L"]
        Llist.append(L)
        self.params["L"] = Llist
        return L

    def ComputeULoss(self, Q, C):
        Lossvec = []
        for i in range(self.N):
            Uxtr = self.Xtr[i]
            Uytr = self.Ytr[i]
            ni = self.nsample[i]
            ci = np.matmul(Q, C[:, i])
            coefu = 0
            for j, xij in enumerate(Uxtr):
                xQC = np.matmul(xij + [1.0], ci)
                yj = Uytr[j]
                if xQC > 8:
                    coefu += xQC - yj * xQC
                else:
                    coefu += np.log(1 + np.exp(xQC)) - yj * xQC
            lossu = coefu / ni
            Lossvec.append(lossu)
        return Lossvec

    def ComputeLoss(self, Lossvec, L, zeta):
        lossval = 0
        for i in range(self.N):
            ni = self.nsample[i]
            lossval += (L[i] + 1 / self.N) * Lossvec[i]
        lossval -= np.dot(L, zeta)
        return lossval

    def getacc(self, ypred, ytrue):
        assert len(ypred) == len(ytrue)
        err = np.abs(np.array(ypred) - np.array(ytrue))
        acc = 1 - np.sum(err) / len(ypred)
        return acc

    def basicTest(self, X, Y):
        Cp = self.params["C"]
        Qp = self.params["Q"]
        acc = []
        for i in range(self.N):
            c = Cp[:, i]
            beta = np.matmul(Qp, c)
            Xi = X[i]
            ytrue = np.array(Y[i])
            ypred = []
            for xij in Xi:
                ex = np.dot(xij + [1.0], beta)
                if ex < 0:
                    prob = 1 - 1 / (1 + math.exp(ex))
                else:
                    prob = 1 - math.exp(-ex) / (1 + math.exp(-ex))
                if prob >= 0.5:
                    ypred.append(1)
                else:
                    ypred.append(0)
            acci = self.getacc(ypred, ytrue)
            acc.append(acci)
        return acc

    def train(self, maxIter=10):
        C = self.C0
        theta = np.array(self.theta0)
        Ls = []
        Qs = []
        Cs = []
        prev_train = 0
        for i in range(maxIter):
            prevQ = self.params["Q"]
            prevC = self.params["C"]
            Lt = self.Bd * np.exp(theta) / (1 + np.exp(theta).sum())
            Ls.append(Lt)
            Lm = np.add.reduce(Ls) / (i + 1)
            Q = self.updateQ(C, Lt)
            C = self.updateC(Q, Lt)
            Qs.append(Q)
            Cs.append(C)
            Qm = np.add.reduce(Qs) / (i + 1)
            Cm = np.add.reduce(Cs) / (i + 1)
            L = self.updateL(Qm, Cm, self.Bd, self.Z)
            Qstar = self.updateQ(C, Lm)
            Cstar = self.updateC(Q, Lm)
            Lossvec1 = self.ComputeULoss(Qm, Cm)
            Loss1 = self.ComputeLoss(Lossvec1, L, self.Z)
            Loss0 = self.ComputeLoss(Lossvec1, Lm, self.Z)
            nu1 = Loss1 - Loss0
            Lossvec0 = self.ComputeULoss(Qstar, Cstar)
            Loss2 = self.ComputeLoss(Lossvec0, Lm, self.Z)
            nu2 = Loss0 - Loss2
            if (nu1 <= self.nu) and (nu2 <= self.nu):
                flag = True
                for j in range(self.N):
                    if Lossvec0[j] > self.Z[j]:
                        flag = False
                        break
                if flag:
                    self.params["Q"] = Qm
                    self.params["C"] = Cm
                    trainacc = self.basicTest(self.Xtr, self.Ytr)
                    return
                else:
                    self.params["Q"] = Qm
                    self.params["C"] = Cm
            Lossvec_update = self.ComputeULoss(Q, C)
            del_theta = np.array(Lossvec_update) - np.array(self.Z)
            theta = theta + del_theta
            trainacc = self.basicTest(self.Xtr, self.Ytr)
            if (mean(trainacc) - prev_train) < -0.01:
                self.params["Q"] = prevQ
                self.params["C"] = prevC
                break
            else:
                # prev_train = mean(trainacc)
                if mean(trainacc) > prev_train:
                    prev_train = mean(trainacc)
        return

    def test(self):
        Xte = self.Xte
        Yte = self.Yte
        testacc = self.basicTest(Xte, Yte)
        # print("Average test accuracy is " + str(mean(testacc) * 100) + "%")
        return testacc

