class EarlyStopping:
    def __init__(self, patience=25, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def step(self, val_loss):
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False  # Not stopped
        else:
            self.counter += 1
            return self.counter >= self.patience