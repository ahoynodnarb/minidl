import math

import minidiff as md


class Optimizer:
    learning_rate: float

    def __init__(self, learning_rate=1e-3):
        raise NotImplementedError

    def update(self, params, grad, l2_lambda=0):
        raise NotImplementedError


# pretty basic SGD implementation
class SGD(Optimizer):
    def __init__(self, learning_rate=0.0005, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = None

    def update(self, params, grad, l2_lambda=0):
        if self.velocity is None:
            self.velocity = md.zeros_like(params)

        batch_size = grad.shape[0]
        regularized_grad = grad + l2_lambda * params
        self.velocity = (
            self.beta * self.velocity + self.learning_rate * regularized_grad
        )
        return params - self.velocity / batch_size


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.momentum = None
        self.velocity = None
        self.t = 0

    def update(self, params, grad, l2_lambda=0):
        self.t += 1
        if self.momentum is None:
            self.momentum = md.zeros_like(grad)
            self.velocity = md.zeros_like(grad)

        regularized_grad = grad + l2_lambda * params

        self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * regularized_grad
        self.velocity = self.beta2 * self.velocity + (1 - self.beta2) * (
            regularized_grad**2
        )

        m_hat = self.momentum / (1 - self.beta1**self.t)
        v_hat = self.velocity / (1 - self.beta2**self.t)

        w_updt = self.learning_rate * m_hat / (md.sqrt(v_hat) + self.epsilon)
        return params - w_updt


class AdamW(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.momentum = None
        self.velocity = None
        self.t = 0

    def update(self, params, grad, l2_lambda=0):
        self.t += 1
        if self.momentum is None:
            self.momentum = md.zeros_like(grad)
            self.velocity = md.zeros_like(grad)

        self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * grad
        self.velocity = self.beta2 * self.velocity + (1 - self.beta2) * (grad**2)

        m_hat = self.momentum / (1 - self.beta1**self.t)
        v_hat = self.velocity / (1 - self.beta2**self.t)

        # this part decouples weight decay from the step calculation
        # https://paperswithcode.com/method/adamw
        return params - self.learning_rate * (
            m_hat / (md.sqrt(v_hat) + self.epsilon) + l2_lambda * params
        )


class LRScheduler(Optimizer):
    stopped: bool = False
    optimizer: Optimizer

    def __init__(self, optimizer: Optimizer):
        raise NotImplementedError

    def update_state(self, current_epoch, current_metric):
        raise NotImplementedError

    @property
    def learning_rate(self):
        return self.optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate


# inspired by ReduceLROnPlateau PyTorch
class ReduceLROnPlateau(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        mode="min",
        factor=0.1,
        patience=5,
        threshold=1e-3,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        epsilon=1e-8,
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor  # how we actually scale the learning rate when metric falls below threshold
        self.patience = patience  # number of epochs we are willing to tolerate while performance decreases
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown  # number of epochs to wait before resuming normally after reducing lr
        self.min_lr = min_lr  # minimum possible lr we are willing to go to
        self.epsilon = epsilon

        self.best = (
            float("-inf") if self.mode == "max" else float("inf")
        )  # lowest previous metric
        self.prev_epochs = 0  # update this whenever we change the lr
        self.intolerable_epochs = 0  # number of epochs performance has decreased
        self.last_updated_epoch = (
            -1
        )  # this keeps track of the last time we updated the optimizer
        self.update_lr = False

    def dynamic_thresh(self):
        thresh_term = self.threshold if self.mode == "max" else -self.threshold
        if self.threshold_mode == "rel":
            return self.best * (1 + thresh_term)
        else:
            return self.best + thresh_term

    def update_state(self, current_epoch, current_metric):
        # update the worst metric we've seen before computing our dynamic threshold
        currently_performing_well = (
            current_metric > self.best
            if self.mode == "max"
            else current_metric < self.best
        )
        if currently_performing_well:
            self.best = current_metric
            self.intolerable_epochs = 0
            return

        thresh = self.dynamic_thresh()

        surpasses_threshold = (
            current_metric < thresh if self.mode == "max" else current_metric > thresh
        )
        # if our metric at this point is too high, add 1 to the amount of epochs performance has decreased
        if surpasses_threshold:
            self.intolerable_epochs += 1

        passed_epochs = current_epoch - self.last_updated_epoch
        in_cooldown = self.last_updated_epoch != -1 and passed_epochs <= self.cooldown
        # finally, if performance has deteriorated for enough epochs, we can update the learning rate
        if not in_cooldown and self.intolerable_epochs >= self.patience:
            # flag for update, reset values and set when we last updated
            self.update_lr = True
            self.intolerable_epochs = 0
            self.last_updated_epoch = current_epoch

    def update(self, params, grad, **kwargs):
        if self.update_lr:
            self.update_lr = False
            cur_lr = self.optimizer.learning_rate
            new_lr = max(cur_lr * self.factor, self.min_lr)
            # if the difference between the lr's is lower than epsilon (decay), then we don't do anything
            if abs(new_lr - cur_lr) >= self.epsilon:
                self.optimizer.learning_rate = new_lr
                print("updating learning rate: " + str(new_lr))
        # always need to at least update
        return self.optimizer.update(params, grad, **kwargs)


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, max_epochs=20):
        self.optimizer = optimizer
        self.max_epochs = max_epochs

        self.current_epoch = 0
        self.stop = False
        self.initial_lr = self.optimizer.learning_rate

    def update_state(self, current_epoch, current_metric):
        if current_epoch > self.max_epochs:
            self.stopped = True
        self.current_epoch = current_epoch

    def update(self, params, grad, **kwargs):
        if not self.stopped:
            self.optimizer.learning_rate = (
                self.initial_lr
                * 0.5
                * (1 + math.cos(math.pi * self.current_epoch / self.max_epochs))
            )
        return self.optimizer.update(params, grad, **kwargs)


class LinearLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iterations=5,
    ):
        self.optimizer = optimizer
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iterations = total_iterations

        self.initial_lr = optimizer.learning_rate
        self.stopped = False
        self.iterations = 0

    def update_state(self, current_epoch, current_metric):
        if self.iterations >= self.total_iterations:
            self.stopped = True
        else:
            self.iterations += 1

    def update(self, params, grad, **kwargs):
        if not self.stopped:
            self.optimizer.learning_rate = self.initial_lr * (
                (1 - self.iterations / self.total_iterations) * self.start_factor
                + (self.iterations / self.total_iterations) * self.end_factor
            )
        else:
            self.optimizer.learning_rate = self.initial_lr * self.end_factor
        return self.optimizer.update(params, grad, **kwargs)

    @property
    def learning_rate(self):
        return self.optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate


class SequentialLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, schedulers, milestones):
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.milestones = milestones

        self.iterations = 0
        self.idx = 0

    def update_state(self, current_epoch, current_metric):
        if self.idx >= len(self.milestones):
            return

        cur_milestone = self.milestones[self.idx]
        if self.iterations >= cur_milestone:
            self.iterations = 0
            self.idx += 1
        else:
            self.iterations += 1
        self.schedulers[self.idx].update_state(current_epoch, current_metric)

    def update(self, params, grad, **kwargs):
        cur_scheduler = self.schedulers[self.idx]
        return cur_scheduler.update(params, grad, **kwargs)
