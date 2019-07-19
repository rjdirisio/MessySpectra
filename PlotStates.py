import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.linalg as sla
FundamentalShell = "Fundamentals_"
exptlPeaks = "redH//allHTesting/spectraTesting/ExperimentalPeaksJCPL"
VCIPeaks = "redH//allHTesting/spectraTesting/VCIPeaksJCPL"
ComboShell = "combinationOverrtone_"
#7       2259.8322404027153
#6       1869.632661124042   0.051675331356500445
#7       1869.810029038624   0.051695242922013974
allD=2050
allH=2600
def plotSpec(pltdata):
    interest = np.loadtxt(FundamentalShell + pltdata)
    interestfreqs = interest[:, 1]
    interest2 = np.loadtxt(ComboShell + pltdata + ".data2")
    interestfreqs2 = interest2[:, 0]
    totalFreqs = np.concatenate((interestfreqs, interestfreqs2))
    plt.stem(totalFreqs,np.ones(len(totalFreqs)),'k',markerfmt='')
    qual = (totalFreqs>(allH-50.)) * (totalFreqs<(allH+50.))
    asdf = np.where(qual)
    print 'allH'
    print len(asdf[0])
    qual = (totalFreqs>(allD-50.)) * (totalFreqs<(allD+50.))
    asdf = np.where(qual)
    print 'allD'
    print len(asdf[0])

    if 'allH' in pltdata:
        plt.xlim([800,4000])
    else:
        plt.xlim([800,3500])
    plt.savefig('densityOfStates/'+pltdata+'states.png')
plotSpec('fSymtet_allHrnspc_justOThenCartOEck')
plotSpec('fSymtet_allDrnspc_justOThenCartOEck')