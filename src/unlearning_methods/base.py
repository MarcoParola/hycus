

class BaseUnlearningMethod():
    def __init__(self, model, **kwargs):
        self.model = model

    def unlearn(self, data, **kwargs):
        raise NotImplementedError