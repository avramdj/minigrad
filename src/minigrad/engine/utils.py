NO_GRAD = False

def is_grad():
    return NO_GRAD

class no_grad:
    def __enter__(self):
        global NO_GRAD
        NO_GRAD = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        global NO_GRAD
        NO_GRAD = False