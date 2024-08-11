import numpy as np
from sklearn.linear_model import LogisticRegression

class IndividualLearning:
    def __init__(self, nsample, trainX, trainY, testX, testY):
        #         super().__init__(X, Y)
        #         self.Lambda = Lambda
        self.nsample = nsample
        self.N = len(self.nsample)
        self.Xtr = trainX
        self.Ytr = trainY
        self.Xte = testX
        self.Yte = testY
        self.d = len(trainX[0][0])
        self.models = []

    def train(self):
        coefps = []
        trainacc = []
        models = []
        X = self.Xtr
        Y = self.Ytr
        for xdata, ydata in zip(X, Y):
            Xnp = np.array(xdata)
            Ynp = np.array(ydata)
            initmodel = LogisticRegression(solver='liblinear', random_state=0)
            initmodel.fit(Xnp, Ynp)
            coef = np.append(initmodel.coef_, initmodel.intercept_)
            coef = coef.tolist()
            coefps.append(coef)
            models.append(initmodel)
            Y_pred = initmodel.predict(Xnp)
            trainacc.append(np.sum(Y_pred == Ynp) / len(Ynp))
        self.models = models
        return

    def test(self):
        X = self.Xte
        Y = self.Yte
        i = 0
        testacc = []
        for xdata, ydata in zip(X, Y):
            modeli = self.models[i]
            Xnp = np.array(xdata)
            Ynp = np.array(ydata)
            Y_pred = modeli.predict(Xnp)
            testacc.append(np.sum(Y_pred == Ynp) / len(Ynp))
            i += 1
        return testacc