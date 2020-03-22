import numpy as np
import torch as th


class EpidemicModel(th.nn.Module):
    """Score driven epidemic model."""
    def __init__(self):

        super(EpidemicModel, self).__init__()
        self.alpha = th.nn.Parameter(th.tensor(0.0, requires_grad=True))
        self.beta = th.nn.Parameter(th.tensor(0.0, requires_grad=True))
        self.gamma = th.nn.Parameter(th.tensor(0.0, requires_grad=True))

    def forward(self, x, y):
        """Run forward step.

        Parameters
        ----------
        x : torch.float
            Cumulative number of cases.
        y : torch.float
            New cases.

        Returns
        -------
        log_lams, omegas
            Time varying parameters.
        """

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
        """Predict from model.

        Predictions will be made for `horizon` steps after the input data.

        Parameters
        ----------
        x : torch.float
            Cumulative number of cases.
        y : torch.float
            New cases.
        horizon : int
            Number of periods to predict.

        Returns
        -------
        pred : numpy.array
            Predictions.
        """

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
    """Score driven epidemic model with unit root dynamics."""
    def __init__(self):

        super(EpidemicModelUnitRoot, self).__init__()
        self.omega0 = th.nn.Parameter(th.tensor(0.0, requires_grad=True))
        self.gamma = th.nn.Parameter(th.tensor(0.0, requires_grad=True))

    def forward(self, x, y):
        """Run forward step.

        Parameters
        ----------
        x : torch.float
            Cumulative number of cases.
        y : torch.float
            New cases.

        Returns
        -------
        log_lams, omegas
            Time varying parameters.
        """

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
        """Predict from model.

        Predictions will be made for `horizon` steps after the input data.

        Parameters
        ----------
        x : torch.float
            Cumulative number of cases.
        y : torch.float
            New cases.
        horizon : int
            Number of periods to predict.

        Returns
        -------
        pred : numpy.array
            Predictions.
        """

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
    """Compute the average Poisson log likelihood."""
    def __init__(self):
        super(PoissonLogLikelihood, self).__init__()

    def forward(self, log_lam, target, max_val=1e6):
        """Run forward step.

        Parameters
        ----------
        log_lam : torch.float
            Log of predicted new cases.
        target : torch.float
            Actual new cases.
        max_val : int
            Number to replace missing values in objective by.

        Returns
        -------
        objective
            Average objective.
        """
        objective = th.exp(log_lam) - target * log_lam
        objective = th.where(
            th.isnan(objective), th.full_like(objective, max_val), objective
        )
        return th.mean(objective)
