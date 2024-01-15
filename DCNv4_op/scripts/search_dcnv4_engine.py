import os

def factors(N):
    res = []
    for i in range(1, N+1):
        if N % i == 0:
            res.append(i)
    return res

if __name__ == '__main__':
    BATCH=64
    for group_channel in [16, 32, 64]:
        for group in [4, 5, 6, 7, 8]:
            for N, Hin, Win in [(BATCH, 56, 56), (BATCH, 28, 28), (BATCH, 14, 14), (BATCH, 7, 7), 
                                (1, 200, 320), (1, 100, 160), (1, 50, 80), (1, 25, 40), (1, 64, 64)]:
                for d_stride in [2, 4, 8, 16]:
                    for m in factors(N*Hin*Win):
                        if m > 64:
                            break
                        block_thread = group * (group_channel//d_stride) * m
                        if block_thread > 1024:
                            break
                        cmd = f"python search_dcnv4.py --n {N} --h {Hin} --w {Win} --g {group} --c {group_channel} --dstride {d_stride} --blockthread {block_thread} --multiplier {m}"
                        os.system(cmd)