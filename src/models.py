import numpy as np
import torch as th


class EpidemicModel(th.nn.Module):
    def __init__(self):

        super(EpidemicModel, self).__init__()
        self.alpha = th.nn.Parameter(th.tensor(0.0, requires_grad=True))
        self.beta = th.nn.Parameter(th.tensor(0.0, requires_grad=True))
        self.gamma = th.nn.Parameter(th.tensor(0.0, requires_grad=True))

    def forward(self, x, y):

        self.omega = self.alpha / (1 - self.beta)
        self.score = 0
        log_lams = []
        omegas = []
        for t in range(x.shape[0]):
            self.omega = self.alpha + self.beta * self.omega + self.gamma * self.score
            log_lam = th.log(x[t]) + self.omega
            self.score = (y[t] - th.exp(log_lam)) / th.sqrt(th.exp(log_lam))
            log_lams.append(log_lam)
            omegas.append(self.omega.detach().numpy())

        return th.stack(log_lams), omegas

    def predict(self, x, y, horizon):

        log_lam, omegas = self.forward(x, y)
        omega = omegas[-1]
        x_last = x[-1]
        y_pred = []
        for h in th.arange(0, horizon):
            y_pred_t = x_last * np.exp(omega)
            omega = self.alpha.detach().numpy() + self.beta.detach().numpy() * omega
            x_last = x_last + y_pred_t
            y_pred.append(y_pred_t)

        return np.append(np.exp(log_lam.detach().numpy()), np.array(y_pred))


class EpidemicModelUnitRoot(th.nn.Module):
    def __init__(self):

        super(EpidemicModelUnitRoot, self).__init__()
        self.omega0 = th.nn.Parameter(th.tensor(0.0, requires_grad=True))
        self.gamma = th.nn.Parameter(th.tensor(0.0, requires_grad=True))

    def forward(self, x, y):

        omega = self.omega0
        self.score = 0
        log_lams = []
        omegas = []
        for t in range(x.shape[0]):
            omega = omega + self.gamma * self.score
            log_lam = th.log(x[t]) + omega
            self.score = (y[t] - th.exp(log_lam)) / th.sqrt(th.exp(log_lam))
            log_lams.append(log_lam)
            omegas.append(omega.detach().numpy())

        return th.stack(log_lams), omegas

    def predict(self, x, y, horizon):

        log_lam, omegas = self.forward(x, y)
        omega = omegas[-1]
        x_last = x[-1]
        y_pred = []
        for h in th.arange(0, horizon):
            y_pred_t = x_last * np.exp(omega)
            x_last = x_last + y_pred_t
            y_pred.append(y_pred_t)

        return np.append(np.exp(log_lam.detach().numpy()), np.array(y_pred))


class PoissonLogLikelihood(th.nn.Module):
    def __init__(self):
        super(PoissonLogLikelihood, self).__init__()

    def forward(self, log_lam, target, max_val=1e6):
        objective = th.exp(log_lam) - target * log_lam
        objective = th.where(
            th.isnan(objective), th.full_like(objective, max_val), objective
        )
        return th.mean(objective)
