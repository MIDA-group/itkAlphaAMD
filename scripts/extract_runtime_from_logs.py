
# Small script for extracting and parsing the runtimes from the logfiles of the registration tools

import numpy as np

def parse_log_runtime(path):
    keyword = "Registration time elapsed:"
    with open(path, 'r') as myfile:
        data = myfile.read()
        pos = data.find(keyword)
        if pos < 0:
            raise 'Keyword not found.'
        else:
            pos = pos + len(keyword)
            while pos < len(data) and data[pos] == ' ':
                pos = pos + 1
            datatail = data[pos:]
            return float(datatail)

def parse_runtimes(paths):
    times = np.zeros(len(paths))
    for (index, pth) in enumerate(paths):
        times[index] = parse_log_runtime(pth)
    return times

def parse_runtimes_for_experiment(path_prefix, path_postfix, N, metric, transformation_size, noise_level):
    log_file = "register_affine_out.txt"
    path = path_prefix + transformation_size + "/" + noise_level + "/" + metric + "/" + path_postfix
    paths = [path + ("registration_%d/" % i) + log_file for i in xrange(1, N+1)]
    times = parse_runtimes(paths)

    return times


if __name__ == "__main__":
    # Cilia 
    metrics = ["alpha_smd", "ssd", "ncc", "mi"]
    transformation_sizes = ["all"]
    noise_levels = ["large"]

    cilia_prefix = "/home/johof680/work/itkAlphaCut-4j/cilia_random_6/"

    print("--- CILIA ---")

    for t in transformation_sizes:
        for n in noise_levels:
            for m in metrics:
                print(t + ", " + n + ", " + m)
                times = parse_runtimes_for_experiment(cilia_prefix, "", 1000, m, t, n)   
                print(times)
                print("Mean:    %.3f" % np.mean(times))
                print("Std-dev: %.3f" % np.std(times))
    
    # LPBA40
    metrics = ["alpha_smd", "ssd", "ncc", "mi"]
    transformation_sizes = ["all"]
    noise_levels = ["large"]

    lpba40_prefix = "/home/johof680/work/itkAlphaCut-4j/lpba40_random_5/"

    print("--- LPBA40 ---")

    for t in transformation_sizes:
        for n in noise_levels:
            for m in metrics:
                print(t + ", " + n + ", " + m)
                times = parse_runtimes_for_experiment(lpba40_prefix, "w_pyramid/", 200, m, t, n)   
                print(times)
                print("Mean:    %.3f" % np.mean(times))
                print("Std-dev: %.3f" % np.std(times))
