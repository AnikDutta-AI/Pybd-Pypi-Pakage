class invalidexception(Exception):
    def __init__(self, message:str="Dataset is not valid"):
        self.message = message
        super().__init__(self.message)