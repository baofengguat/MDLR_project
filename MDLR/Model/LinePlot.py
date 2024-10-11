import os

import matplotlib.pyplot as plt

def excelplot_acc(epoch,acc1,acc2,savefile):
    #
    # with plt.style.context(['science', 'ieee', 'no-latex']):
    #     plt.rcParams['font.family'] = 'Times New Roman'
    #     plt.rcParams['axes.linewidth'] = 0.2

        fig, ax = plt.subplots(figsize=(6,4),dpi=200)
        # plt.title("loss line")
        plt.xlim(1, len(epoch))
        plt.ylim(0.0, 1.0)
        x = plt.MultipleLocator(10)
        y = plt.MultipleLocator(0.1)  

        # ax = plt.gca()
        ax.xaxis.set_major_locator(x)
        ax.yaxis.set_major_locator(y)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.yaxis.set_major_locator(y)

        ax.plot(epoch, acc1, "b--", label="Train acc")
        ax.plot(epoch, acc2, "g--", label="Test  acc")

        ax.legend(fontsize=7)
        plt.savefig(os.path.join(savefile,"acc.png"))

        if not plt.rcParams["text.usetex"]:
            plt.savefig(os.path.join(savefile,"acc.eps"), dpi=1000)
        plt.show()



