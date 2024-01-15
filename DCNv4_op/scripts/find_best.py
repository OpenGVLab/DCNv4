import json
import argparse
class LineParser:
    def __init__(self) -> None:
        self.data = {}

    def parse(self, line):
        def startswith(line, lst):
            for ele in lst:
                if line.startswith(ele):
                    return True
            return False

        if not startswith(line, ['1', '2', '3', '4', '5', '6', '7', '8', '9']):
            return

        eles = line.strip().split()
        key = eles[0]
        if key not in self.data:
            self.data[key] = []
        
        self.data[key].append([eles[1], float(eles[2])])
    
    def sort(self):
        for k, v in self.data.items():
            nv = sorted(v, key=lambda x: x[1])
            self.data[k] = nv
    
    def display_best(self):
        for k, v in self.data.items():
            print(f'{k} \t {v[0][0]} \t {v[0][1]:.4f} \t {v[1][0]} \t {v[1][1]:.4f}')
    
    def display_best_python(self, output):
        res = {}
        def parse(spec):
            d_stride = int(spec.split('/')[0])
            thread = int(spec.split('/')[1].split('(')[0])
            m = int(spec.split('(')[1].split(')')[0])
            return d_stride, thread, m

        for k, v in self.data.items():
            res[k] = parse(v[0][0])

        with open(output, "w") as f:
            json.dump(res, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    with open(args.input) as f:
        lines = f.readlines()

    lineparser = LineParser()
    for line in lines:
        lineparser.parse(line)
    lineparser.sort()
    lineparser.display_best()
    lineparser.display_best_python(args.output)