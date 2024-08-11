import numpy as np
import cvxpy as cp
import math
from statistics import mean
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from numpy.linalg import pinv
from scipy.linalg import block_diag

class FairCL:
    def __init__(self, nsample, K, p, trainX, trainY, testX, testY):
        #         super().__init__(X, Y)
        #         self.Lambda = Lambda
        self.nsample = nsample
        self.N = len(self.nsample)
        self.K = K
        self.Xtr = trainX
        #         self.Utr = u_train
        self.Ytr = trainY
        self.Xte = testX
        #         self.Ute = u_test
        self.Yte = testY
        self.d = len(trainX[0][0])
        self.p = p
        self.Q0, self.B0 = self.InitializeQ(self.Xtr, self.Ytr)
        # method="uniform": using uniform distribution
        # method="mindis": minimizing \sum_i||\beta_i-Qc_i||^2 with |c_i|==1
        self.C0 = self.InitializeC(self.Q0, self.B0, "mindis")
        self.params = {"Q": self.Q0, "C": self.C0, "w": []}

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

    def updateQ(self, C, w, Qt, lbd):
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
                coef.append(w[i] / ni)
        coef = np.array(coef)
        coef = coef / np.sum(coef)
        Xc = np.array(Xc)
        Yc = np.array(Yc)
        log_likelihood = cp.sum(cp.multiply(coef, cp.multiply(Yc, Xc @ qv) - cp.logistic(Xc @ qv)))
        qt = Qt.flatten()
        problem = cp.Problem(cp.Maximize(log_likelihood - lbd * cp.norm(qv - qt, 2)))
        problem.solve(solver="SCS", max_iters=1000)
        qval = qv.value
        Q = qval.reshape(self.K, (self.d + 1)).T
        self.params["Q"] = Q
        return Q

    def updateC(self, Q, w, Ct, lbd):
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
                coef.append(w[i] / ni)
            Bxq.append(Uxq)
        coef = np.array(coef)
        #         coef = coef / np.sum(coef)
        Xq = np.array(block_diag(*Bxq))
        log_likelihood = cp.sum(cp.multiply(coef, cp.multiply(Yq, Xq @ cv) - cp.logistic(Xq @ cv)))
        onem = np.kron(np.eye(self.N, dtype=int), np.ones(self.K))
        ct = Ct.flatten()
        problem = cp.Problem(cp.Maximize(log_likelihood - lbd * cp.norm(cv - ct, 2)), [onem @ cv == np.ones(self.N)])
        problem.solve(solver="SCS", max_iters=1000)
        cval = cv.value
        C = cval.reshape(self.N, self.K).T
        self.params["C"] = C
        return C

    def updateW(self, Q, C):
        coef = []
        for i in range(self.N):
            Uxtr = self.Xtr[i]
            Uytr = self.Ytr[i]
            ni = self.nsample[i]
            ci = np.matmul(Q, C[:, i])
            coefi = 0
            for j, xij in enumerate(Uxtr):
                xQC = np.matmul(xij + [1.0], ci)
                yj = Uytr[j]
                coefi += np.log(1 + np.exp(xQC)) - yj * xQC
            coef.append(coefi)
        coef = np.array(coef)
        w = coef / np.sum(coef)
        for i in range(self.N):
            if w[i] > 0.5:
                w[i] = 0.5
        w = w / np.sum(w)
        w = w ** self.p
        Wlist = self.params["w"]
        Wlist.append(w)
        self.params["w"] = Wlist
        return w, coef

    def getacc(self, ypred, ytrue):
        assert len(ypred) == len(ytrue)
        err = np.abs(np.array(ypred) - np.array(ytrue))
        acc = 1 - np.sum(err) / len(ypred)
        return acc

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

    def train(self, lbd, maxIter=10):
        Q = self.Q0
        C = self.C0
        prev_train = 0
        for i in range(maxIter):
            prevQ = self.params["Q"]
            prevC = self.params["C"]
            w, indloss = self.updateW(Q, C)
            Q = self.updateQ(C, w, Q, lbd)
            C = self.updateC(Q, w, C, lbd)
            self.params["Q"] = Q
            self.params["C"] = C
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
        return testacc
