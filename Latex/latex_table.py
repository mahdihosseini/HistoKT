import pandas as pd

avg = pd.read_excel("/Users/JZ/Downloads/norm_ImageNet-pretrained_on ImageNet_post_without_aug.xlsx", sheet_name = ['avg'])
std = pd.read_excel("/Users/JZ/Downloads/norm_ImageNet-pretrained_on ImageNet_post_without_aug.xlsx", sheet_name = ['std'])

# \textbf{ADP} & \baseline{93.56$_{0.40}$} & {97.57$_{0.04}$}\\

out_str = {}
keys = list(avg['avg'].keys())[1:]
for i in range(0, len(keys)):
    key = keys[i]
    out_str[i] = r"\textbf{" + key + "}"

for i in range(0, len(keys)):
    key = keys[i]
    baseline = avg['avg'][key][i]
    diff = max(avg['avg'][key]) - min(avg['avg'][key])
    for j in range(0, len(avg['avg'][key])):
        if i == j:
            out_str[j] = out_str[j] + r" & {" + str(round(avg['avg'][key][j], 2)) + r"$_{" + str(round(std['std'][key][j], 2))+"}$}"
        else:
            if avg['avg'][key][j] > baseline:
                amount = 20 + (avg['avg'][key][j] - baseline)/(max(avg['avg'][key])-baseline)*(100-20)
                out_str[j] = out_str[j] + r" & \cellGreenBG{" + str(round(amount, 2)) + "}{" + str(round(avg['avg'][key][j], 2)) + r"$_{" + str(round(std['std'][key][j], 2))+"}$}"
            else:
                amount = 10 + (baseline-avg['avg'][key][j])/(baseline-min(avg['avg'][key]))*(60-15)
                out_str[j] = out_str[j] + r" & \cellRedBG{" + str(round(amount, 2)) + "}{" + str(round(avg['avg'][key][j], 2)) + r"$_{" + str(round(std['std'][key][j], 2))+"}$}"

for i in range(0, len(keys)):
    key = keys[i]
    out_str[i] = out_str[i] + r"\\"

#print(out_str)

file = open("out.txt","w")
for key in out_str.keys():
    file.write(out_str[key]+ "\n")
file.close()