import os
import math
import argparse
import numpy as np
from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-dir', type=str, default='', help='Path to the results')
    parser.add_argument('--shot-list', type=int, nargs='+', default=[10], help='')
    args = parser.parse_args()

    wf = open(os.path.join(args.res_dir, 'results.txt'), 'w')

    for shot in args.shot_list:

        file_paths = []
        for fid, fname in enumerate(os.listdir(args.res_dir)):
            if fname.split('_')[0] != '{}shot'.format(shot):
                continue
            _dir = os.path.join(args.res_dir, fname)
            if not os.path.isdir(_dir):
                continue
            file_paths.append(os.path.join(_dir, 'log.txt'))

        header, results = [], []
        for fid, fpath in enumerate(sorted(file_paths)):
            lineinfos = open(fpath).readlines()
            if fid == 0:
                res_info = lineinfos[-2].strip()
                header = res_info.split(':')[-1].split(',')
            res_info = lineinfos[-1].strip()
            res_list = []
            for x in res_info.split(':')[-1].split(','):
                print("x ========== "+x+" "+ x.split('.')[0])
                if x.split('.')[0] not in [0,1,2,3,4,5,6,7,8,9]:
                    print("x ========== "+x)
                    res_list.append(float(x))
                else:
                    res_list.append(0.0)
            results.append([fid] + res_list)
#             results.append([fid] + [float(x) for x in res_info.split(':')[-1].split(',')])
        print("results ============= "+str(results))
        results_np = np.array(results)
        avg = np.mean(results_np, axis=0).tolist()
        cid = [1.96 * s / math.sqrt(results_np.shape[0]) for s in np.std(results_np, axis=0)]
        results.append(['μ'] + avg[1:])
        results.append(['c'] + cid[1:])

        table = tabulate(
            results,
            tablefmt="pipe",
            floatfmt=".2f",
            headers=[''] + header,
            numalign="left",
        )
        
        
        res = []
#         str(results[0][0]).encode('utf-8')
        for i in range(len(results)):
            res.append([])
#             print(res)      
            for j in range(len(results[0])):
#                 print(results[i][j])
#                 print(type(results[i][j]))
                if type(results[i][j]) != str  :
                    res[i].append(float(str(results[i][j]).encode('utf-8')))
                else:
                    res[i].append(float(str(0).encode('utf-8')))
        table = tabulate(
            res,
            tablefmt="pipe",
            floatfmt=".2f",
            headers=[''] + header,
            numalign="left",
        )
        
        print(table)      
        wf.write('--> {}-shot\n'.format(shot))
        wf.write('{}\n\n'.format(table))
        wf.flush()
    wf.close()

    print('Reformat all results -> {}'.format(os.path.join(args.res_dir, 'results.txt')))


if __name__ == '__main__':
    main()
