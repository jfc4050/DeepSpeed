import subprocess
import os
from collections import defaultdict
import numpy as np

# test deepspeed locally
file_path = os.path.abspath(__file__)
file_dir = os.path.dirname(file_path)

def main():
    """"""
    stages = [0, 1, 2, 3]
    test_model_script = os.path.join(file_dir, 'test_model.py')
    for s in stages:
        cmd = ['deepspeed', test_model_script, '--zero', str(s)]
        subprocess.run(cmd)
    
    # read data
    default_name_format = "/tmp/loss_log_stage{}.h16.cgFalse.rcTrue.txt"
    losses = dict()
    for s in stages:
        f = default_name_format.format(s)
        vals = np.genfromtxt(f, delimiter=',')
        losses[s] = vals

    for i in stages[1:]:
        allclose = np.allclose(losses[0], losses[i], rtol=1e-4, atol=1e-4)
        print(f'stage{0}, stage{i}, losses all close {allclose}')
        if not allclose:
            for row1, row2 in zip(losses[0], losses[i]):
                if not np.allclose(row1, row2):
                    print(f"loss diff {row1[1] - row2[1]}:: stage0: step{row1[0]}, loss {row1[1]}"
                        f" stage{i}: step{row2[0]}, loss {row2[1]}")

if __name__ == "__main__":
    main()