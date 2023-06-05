import os


def cuda_is_available():
    if os.system("nvcc --version") == 0:
        return True
    return False
