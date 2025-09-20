import pandas
import matplotlib.pyplot as plt

import seaborn as sns

loss = {
    "Round": [0, 1, 2, 3, 4],
    "float16": [
        2.542977809906006,
        1.785860538482666,
        1.7768937349319458,
        1.7765766382217407,
        1.7799557447433472,
    ],
    "float4": [
        2.880049467086792,
        1.9248440265655518,
        1.9041017293930054,
        1.8973793983459473,
        1.894836664199829,
    ],
    "normalized float4": [
        2.707158088684082,
        1.8648680448532104,
        1.8497788906097412,
        1.8457902669906616,
        1.8462167978286743,
    ],
    "adaquant": [
        3.9254233837127686,
        2.0198545455932617,
        1.994623064994812,
        1.9854929447174072,
        1.955665946006775,
    ],
    "blockwise8": [
        2.5443713665008545,
        1.7876044511795044,
        1.7765042781829834,
        1.775207757949829,
        1.7779940366744995,
    ],
}
df = pandas.DataFrame(loss)
dfl = pandas.melt(df, ["Round"], value_name="Test Loss", var_name="mode")


ax = sns.lineplot(data=dfl, x="Round", y="Test Loss", hue="mode")

# plt.title("Test loss of different quantization schemes")
plt.xlabel("Round")
plt.show()
fig = ax.get_figure()
fig.savefig("output.png")
