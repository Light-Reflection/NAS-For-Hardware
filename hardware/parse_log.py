
import re
import csv
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
def parse_log(path):
    pat = r'mean of time: *.*,'
    strings = []
    i = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            res = re.search(pat, line)
            if res:
                nums = res.group(0).split(":")[-1].split(',')[0].strip()
                # print(nums)
                strings.append(eval(nums))
    return strings

# def to_csv(path):
#     with open(path, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         writer

def parse_model():
    # model_b = parse_log('/media/dm/d/Projects/modelb-gpu.log')
    # model_a = parse_log('/media/dm/d/Projects/modela-gpu.log')
    # model_d = parse_log('/media/dm/d/Projects/modeld-gpu.log')
    # model_c = parse_log('/media/dm/d/Projects/modelc-gpu.log')
    # model_a.sort()
    # model_b.sort()
    # model_d.sort()
    # model_c.sort()
    # t = 30

    # with open('./model.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow(["Model", "Cost Time /s"])
    #     for num in model_a[:t]:
    #         print(num)
    #         writer.writerow(["A", num])
    #     for num in model_b[:t]:
    #         writer.writerow(["B", num])
    #     for num in model_c[:t]:
    #         writer.writerow(["C", num])

    #     for num in model_d[:t]:
            # writer.writerow(["D", num])
    df = pd.read_csv("./model.csv")

    sbn.boxplot(x=df['Model'], y=df['Cost Time /s'],  palette="pastel")
    sbn.swarmplot(x=df['Model'], y=df['Cost Time /s'], color='.1', size=1)
    plt.show()
    # model_e = 


if __name__ == '__main__':
    mbv1_g = parse_log('/media/dm/d/Projects/dmmo/hardware/C-MBV1.log')
    mbv2_g = parse_log('/media/dm/d/Projects/dmmo/hardware/C-MBV2.log')

    mbv1_c = parse_log('/media/dm/d/Projects/dmmo/hardware/C-MBV1-CPU.log')
    mbv2_c = parse_log('/media/dm/d/Projects/dmmo/hardware/C-MBV2-CPU.log')

    print(sum(mbv1_g[10:]) / len(mbv1_g[10:]))
    print(sum(mbv2_g) / len(mbv2_g))
    print(sum(mbv1_c) / len(mbv1_c))
    print(sum(mbv2_c) / len(mbv2_c))

    # # mbv1_r = parse_log('/media/dm/d/Projects/dmmo/hardware/C-MBV1-RAS.log')
    # # mbv2_r = parse_log('/media/dm/d/Projects/dmmo/hardware/C-MBV1-RAS.log')
    # t = 30
    # with open('./mbv1v2_c.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow(["Model", "Cost Time /s", "Platform"])
    #     for num in mbv1_c[:t]:
    #         # print(num)
    #         writer.writerow(["MobileNet V1", num, "GPU"])        
    #     for num in mbv2_c[:t]:
    #         # print(num)
    #         writer.writerow(["MobileNet V2", num, "GPU"])

    # #     for num in mbv1_c[:t]:
    # #         # print(num)
    # #         writer.writerow(["MobileNet V1", num, "CPU"])
    # #     for num in mbv2_c[:t]:  
    # #         # print(num)
    # #         writer.writerow(["MobileNet V2", num, "CPU"])

    # df = pd.read_csv("./mbv1v2_c.csv")
    # # df[""]

    # sbn.boxplot(x='Model', y='Cost Time /s', data=df) #,  palette="")
    # sbn.swarmplot(x='Model', y='Cost Time /s', data=df,  color='.1', size=2)
    # plt.show()












