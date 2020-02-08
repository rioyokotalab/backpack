from backpack.extensions.module_extension import ModuleExtension


class LossHessianBaseModule(ModuleExtension):

    def __init__(self, derivatives):
        self.derivatives = derivatives
        super().__init__()
