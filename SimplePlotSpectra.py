import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.linalg as sla
import itertools
import csv
FundamentalShell = "Fundamentals_"
exptlPeaks = "redH//allHTesting/spectraTesting/ExperimentalPeaksJCPL"
VCIPeaks = "redH//allHTesting/spectraTesting/VCIPeaksJCPL"
ComboShell = "combinationOverrtone_"


class myPlot:
    def __init__(self, cfg, pltdata,mix=True, pltVCI=False,stix=False,OC=False):
        self.cfg = cfg
        self.pltdata = pltdata  # intending to be a file with the format I like, with freq       intensity
        print self.pltdata
        self.mix = mix
        self.pltVCI = pltVCI  # true or false
        self.pltVCI = pltVCI  # true or false
        self.stix = stix
        self.OC=OC

        if 'tet' in self.pltdata:
            self.nvibs = 33
            if '1He'in self.pltdata:
                self.eShift = 400 #300 #600 #250
            elif 'allH'in self.pltdata:
                self.eShift = 600 #600 looks basically the same as 400
            elif 'allD' in self.pltdata:
                self.eShift = 600 #300
            else:
                self.eShift = 300
        else:
            self.nvibs = 24
            if 'allH' in self.pltdata:
                self.eShift = 500
            elif 'allD' in self.pltdata:
                self.eShift = 400
            elif '1Hw' in self.pltdata or '1Dw' in self.pltdata:
                self.eShift = 400
            elif '1He' in self.pltdata or '1De' in self.pltdata:
                self.eShift = 400
            elif '1Hh' in self.pltdata or '1Dh' in self.pltdata:
                self.eShift = 400


        self.mx = None
        self.mn = None
        self.mnE=None
        self.mxE=None
        self.centerE=None

    def normAll(self,freqs,ints,num,assign=[]):
        fundFreqG800Index = np.where(freqs >= num)
        len1 = len(freqs)
        freqs2 = freqs[fundFreqG800Index]
        ints2 = ints[fundFreqG800Index]
        if len(assign) != 0:
            assign2 = assign[fundFreqG800Index]
            lost = len1-len(freqs2)
            print 'less',lost
            ints2 = ints2 / np.amax(ints2)
            return freqs2, ints2,lost,assign2
        else:
            lost = len1 - len(freqs2)
            print 'less', lost
            ints2 = ints2 / np.amax(ints2)
            return freqs2, ints2, lost
    def normAll2(self,freqs,ints,num,assign=[]):
        peaksE = np.where(freqs > num)
        topInts = np.amax(ints[peaksE])
        ints /= topInts
        return ints

    def getLargeCouplings(self,chunkE,chunkAssign,threshold):
        # Get biggest couplings and their assignments
        bigCouple = np.argwhere(np.absolute(chunkE) > threshold)
        bigCouple = bigCouple[bigCouple[:, 0] != bigCouple[:, 1]]
        new_pairs = set([])
        for ra in bigCouple:
            x = ra[0]
            y = ra[1]
            if x < y:
                new_pairs.add((x, y))
            else:
                new_pairs.add((y, x))
        bigCouple = np.array(list(new_pairs))
        if len(bigCouple) > 0:
            energies = chunkE[bigCouple[:, 0], bigCouple[:, 1]]
            rowInds = bigCouple[:, 0]
            colInds = bigCouple[:,1]  # These are the rows and columns that will correspond to a certain assignment, since we know that the indices in the H matrix correspond to.
            bigCouplings = np.hstack((energies[:, np.newaxis], chunkAssign[rowInds], chunkAssign[colInds]))
            if threshold > 1:
                np.savetxt("bigCouplings/bigCoupleAssignments" + self.pltdata,
                           bigCouplings[bigCouplings[:,0].argsort()], fmt='%i')
            else:
                np.savetxt("bigCouplings/bigOverlapAssignments" + self.pltdata,
                           bigCouplings[bigCouplings[:, 0].argsort()], fmt='%f')
    def setOffDiags(self,num,ov,cou,killall):
        if not killall:
            ov[num,num+1:]=0
            ov[num+1:,num]=0
            cou[num,num+1:]=0
            cou[num+1:,num]=0
        else:
            idxO=np.triu_indices_from(ov,k=1)
            idxC = np.triu_indices_from(cou,k=1)
            ov[idxO]=0.0
            cou[idxC] = 0.0

            idxO = np.tril_indices_from(ov,k=-1)
            idxC = np.tril_indices_from(cou,k=-1)
            ov[idxO] = 0.0
            cou[idxC] = 0.0
        return ov,cou

    def plotOverlapAndCouplings(self,chunkO,chunkE):

        print "max overlap in matrix",np.amax(chunkO - np.eye(len(chunkO)))
        plt.matshow(chunkO - np.eye(len(chunkO)))
        plt.colorbar()
        # plt.clim(-1,1)
        plt.savefig('Matrices/sortedOverlaps_' + self.pltdata,dpi=500)
        plt.close()

        plt.matshow(np.abs(chunkO - np.eye(len(chunkO))))
        plt.colorbar()
        # plt.clim(-1,1)
        plt.savefig('Matrices/sortedOverlaps_abs_' + self.pltdata, dpi=500)
        plt.close()

        plt.matshow(chunkE)
        plt.colorbar()
        plt.savefig('Matrices/sortedCouplings_' + self.pltdata,dpi=500)
        plt.close()

        plt.matshow(np.abs(chunkE))
        plt.colorbar()
        plt.savefig('Matrices/sortedCouplings_abs_' + self.pltdata,dpi=500)
        plt.close()

        chunkCopy = np.copy(chunkE)
        chunkCopy[chunkCopy >= 2000.0] = 2000.0
        plt.matshow(chunkCopy-np.diag(np.diag(chunkCopy)))
        plt.colorbar()
        plt.savefig('Matrices/sortedCouplingsNoDiag_' + self.pltdata,dpi=500)
        plt.close()

        plt.matshow(np.abs(chunkCopy-np.diag(np.diag(chunkCopy))))
        plt.colorbar()
        plt.savefig('Matrices/sortedCouplingsNoDiag_abs_' + self.pltdata,dpi=500)
        plt.close()

    def getScatterCombos_all(self,chunkE,freqOversCombos,evecs,evals): #symm/anti + low freq modes
        if 'tet' in self.pltdata:
            fund1 = 6 #Symm
            fund2 = 7 #Anti 1
            fund3 = 8 #Anti 2
        else:
            fund1 = 5
            fund2 = 6
        #trimer
            #allH - 5 = symm 6 = anti
            #allD - 5 = symm 6 = anti
            #1He - 5 OH stretch 6 = OD stretch
            #1hw - 5  = symm 6 = anti
            #1hh - 5  = symm 6 = anti
        # tetramer
            # allH - 6 = symm, 7,8 = anti
            # allD - 6 = symm, 7,8 = anti
            # 1He - 6 = shared H, 7,8 = symm IDB, anti IDB
            # 1hw - 6 = symm, 7,8 = anti(2,1), and anti(1,1)
        # assignz = {}
        # k=0
        # with open("TransformationMatrix/newest/csv/TransformationMatrix"+self.pltdata+".csv",'rb') as csvfile:
        #     reader = csv.reader(csvfile)
        #     for row in reader:
        #         if k != 0:
        #             assignz[k-1] = row[1]
        #         k+=1
        whereMatch = np.in1d(freqOversCombos , np.diag(chunkE))
        whereMatch2 = np.where(whereMatch)[0] #index where freqCombosOvers matches chunkE
        evecInd = []
        for state in whereMatch2:
            evecInd.append(np.where(np.diag(chunkE)==freqOversCombos[state])[0][0])
        evecInd = np.array(evecInd)
        assignList = []
        # overtones THEN combinations?
        for x in np.arange(self.nvibs):
            assignList.append([x, x])
        for (x,y) in itertools.combinations(np.arange(self.nvibs),2):
            assignList.append([x,y])
        assignList = np.array(assignList) #full combo and overtone indices

        finalAssign = assignList[whereMatch2] #get combo numbers, same size as evecInd and in the same order (?)
        evecFin = [[],[]]
        evecFin[0] = evecInd[np.where(finalAssign[:,0] == fund1)]
        evecFin[1] = evecInd[np.where(finalAssign[:,0] == fund2)]

        indList1 = finalAssign[np.where(finalAssign[:,0] == fund1)] #for which mode
        indList2 = finalAssign[np.where(finalAssign[:, 0] == fund2)]  # for which mode
        # indListAssign = [[],[]]
        Assign = ["Symmetric + Low Freq Mode","Antisymmetric + Low Freq Mode"]
        col = ['#951cf3','#f1a10c']
        if 'tet' in self.pltdata:
            evecFin = [[],[], []]
            evecFin[0] = evecInd[np.where(finalAssign[:, 0] == fund1)]
            evecFin[1] = evecInd[np.where(finalAssign[:, 0] == fund2)]
            evecFin[2] = evecInd[np.where(finalAssign[:, 0] == fund3)]
            indList3 = finalAssign[np.where(finalAssign[:, 0] == fund3)]  # for which mode
            Assign = ["Symmetric + Low Freq Mode", "Antisymmetric 1 + Low Freq Mode", "Antisymmetric 2 + Low Freq Mode"]
            col.append("m")
        #     indListAssign = [[],[],[]]
        # for indListN in range(len(indListList)):
        #     for q in indListList[indListN]:
        #         indListAssign[indListN].append(assignz[q[1]]) #assign
        fig = plt.figure()
        ax = plt.subplot(111)
        for x in range(len(Assign)):
            for ghnum,gh in enumerate(evecFin[x]):
                for nevs in range(len(evals)):
                    if nevs == 0 and ghnum == 0:
                        # plt.scatter(evals[nevs], np.square(evecs[indList[gh], nevs]), c=col,label=indListAssign[gh])
                        ax.scatter(evals[nevs], np.square(evecs[gh, nevs]), facecolor='none',edgecolors=col[x], label=Assign[x])
                    else:
                        # plt.scatter(evals[nevs], np.square(evecs[indList[gh], nevs]), c=col)
                        ax.scatter(evals[nevs], np.square(evecs[gh, nevs]), facecolor='none',edgecolors=col[x])
        if 'allD' in self.pltdata:
            ax.set_xlim([800,3000])
        else:
            ax.set_xlim([800, 4000])
        ax.set_ylim([-0.01,1.0])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),scatterpoints=1)
        # plt.legend(scatterpoints=1, fontsize=8)
        plt.savefig("ScatterCoefPlots/TotalContribs" + self.pltdata + "_shift"+str(self.eShift),dpi=500)
        plt.close()


    # def getScatterCombos(self,chunkE,freqCombosOvers, mode, evecs,evals):
    #     """CUURENTLY BAD BECAUSE IT GRABS THE EVEC IND OF EVERYTHING NOT JUST THE MODE IT WANTS"""
    #     if 'tet' in self.pltdata:
    #         if mode == "Symm":
    #             fund = 6
    #         elif mode == "Anti":
    #             fund = 7
    #         elif mode == "Anti2":
    #             fund = 8
    #     else:
    #         if mode == "Symm":
    #             fund = 5
    #         elif mode == "Anti":
    #             fund = 6
    #     assignz = {}
    #     k=0
    #     with open("TransformationMatrix/newest/csv/TransformationMatrix"+self.pltdata+".csv",'rb') as csvfile:
    #         reader = csv.reader(csvfile)
    #         for row in reader:
    #             if k != 0:
    #                 assignz[k-1] = row[1]
    #             k+=1
    #
    #     whereMatch = np.in1d(freqCombosOvers , np.diag(chunkE))
    #     whereMatch2 = np.where(whereMatch)[0] #index where freqCombosOvers matches chunkE
    #     evecInd = []
    #     for state in whereMatch2:
    #         evecInd.append(np.where(np.diag(chunkE)==freqCombosOvers[state])[0][0]) #where there is a state in redH that is part of the combo & overtone
    #     evecInd = np.array(evecInd)
    #     assignList = []
    #     for (x,y) in itertools.combinations(np.arange(self.nvibs),2):
    #         assignList.append([x,y])
    #     for x in np.arange(self.nvibs):
    #         assignList.append([x,x])
    #     assignList = np.array(assignList) #full combo and overtone indices
    #
    #     finalAssign = assignList[whereMatch2] #get combo numbers
    #     indList = finalAssign[np.where(finalAssign[:,0] == fund)] #for which mode
    #     indListAssign = []
    #     for q in indList:
    #         indListAssign.append(assignz[q[1]]) #assign
    #
    #     colors = iter(cm.rainbow(np.linspace(0, 1, len(indList))))
    #     fig = plt.figure()
    #     ax = plt.subplot(111)
    #     for gh in range(len(indList)):
    #         # if gh == 0:
    #         #     col=[0,0,0,0]
    #         # elif gh == 1:
    #         #     col = [0, 0, 0, 0.5]
    #         # else:
    #         col = next(colors)
    #         for nevs in range(len(evals)):
    #             if nevs == 0:
    #                 # plt.scatter(evals[nevs], np.square(evecs[indList[gh], nevs]), c=col,label=indListAssign[gh])
    #                 ax.scatter(evals[nevs], np.square(evecs[evecInd[gh], nevs]), c=col, label=indListAssign[gh])
    #             else:
    #                 # plt.scatter(evals[nevs], np.square(evecs[indList[gh], nevs]), c=col)
    #                 ax.scatter(evals[nevs], np.square(evecs[evecInd[gh], nevs]), c=col)
    #     if 'allD' in self.pltdata:
    #         ax.set_xlim([800,3000])
    #     else:
    #         ax.set_xlim([800, 4000])
    #     ax.set_ylim([-0.001,1.0])
    #     ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),scatterpoints=1)
    #     # plt.legend(scatterpoints=1, fontsize=8)
    #     plt.savefig("ScatterCoefPlots/LowFreqContributions" + self.pltdata + assignz[indList[0,0]].replace(" ","")+"_shift"+str(self.eShift),dpi=500)
    #     plt.close()

    def getContributionsAndPlot(self,chunkE,freqFunds,freqOversCombos,imat,evals,evecs,chunkAssign):
        topC = open("topContributions/contr_"+self.pltdata,"w+")
        topC.write('E     I      State (pair)   Coef^2\n')
        for nevs in range(len(evals)):
            stAs=chunkAssign[np.argmax(np.square(evecs[:, nevs]))]
            topC.write('%5.1f, %5.4f, %2.0f %2.0f , %5.3f\n' % (evals[nevs],imat[nevs]/np.amax(imat),stAs[0],stAs[1],np.amax(np.square(evecs[:,nevs]))))
        topC.close()
        # for nevs in range(len(evals)):
        #     stAs=chunkAssign[np.argmax(np.square(evecs[nevs]))]
        #     topC.write('%5.1f, %5.4f, %2.0f %2.0f , %5.3f\n' % (evals[nevs],imat[nevs],stAs[0],stAs[1],np.square(np.amax(evecs[nevs,:]))))
        # topC.close()
        allC = open("allContributions/contr_" + self.pltdata, "w+")
        allC.write('E     I\n')
        evecsSorted = np.zeros((len(evecs),len(evecs)))
        for nevs in range(len(evals)):
            stAs = chunkAssign[np.flip(np.argsort(np.square(evecs[:, nevs])))]
            evecsSorted[:,nevs] = evecs[np.flip(np.argsort(np.square(evecs[:, nevs]))),nevs]
            allC.write('%5.1f, %5.4f\n '% (evals[nevs],imat[nevs]/np.amax(imat)))
            for istate in range(10): #top 10 states
                allC.write("%2.0f % 2.0f, %5.6f; " % (
                    stAs[istate][0],
                    stAs[istate][1],
                    np.square(evecsSorted[istate,nevs]))
                           )
            allC.write("\n")
        allC.close()

        contrib = open('sharedProtonCoefs/contributionOfOH_'+self.pltdata,"w+")
        if 'tet' in self.pltdata:
            if '1He' not in self.pltdata:
                someInd =  np.argwhere(chunkE == freqFunds[6])[0][0]
                otherInd = np.argwhere(chunkE == freqFunds[7])[0][0]
                thirdInd = np.argwhere(chunkE == freqFunds[8])[0][0]
            else:
                someInd = np.argwhere(chunkE == freqFunds[6])[0][0]
                otherInd = np.argwhere(chunkE == freqFunds[6])[0][0]
                thirdInd = np.argwhere(chunkE == freqFunds[6])[0][0]
            contrib.write('E     I      OH-1   OH-2  OH-3\n')
            for nevs in range(len(evals)):
                contrib.write('%5.1f %5.4f %5.3f %5.3f %5.3f\n' % (
                evals[nevs],
                imat[nevs] / np.amax(imat),
                np.square(evecs[someInd, nevs]),
                np.square(evecs[otherInd, nevs]),
                np.square(evecs[thirdInd, nevs]))
                              )
            fig = plt.figure()
            ax = plt.subplot(111)
            for nevs in range(len(evals)):
                # print 'hi'
                if nevs == 0:
                    #plt.scatter(x, y, s=80, facecolors='none', edgecolors='r')
                    ax.scatter(evals[nevs],np.square(evecs[someInd,nevs]),facecolor='none',edgecolors='r',label="Totally Symmetric Stretch")
                    ax.scatter(evals[nevs], np.square(evecs[otherInd,nevs]),facecolor='none',edgecolors='b',label="Antisymmetric Stretch 1")
                    ax.scatter(evals[nevs], np.square(evecs[thirdInd,nevs]),facecolor='none',edgecolors='g',label="Antisymmetric Stretch 2")
                else:
                    ax.scatter(evals[nevs], np.square(evecs[someInd, nevs]), facecolor='none',edgecolors='r')
                    ax.scatter(evals[nevs], np.square(evecs[otherInd, nevs]), facecolor='none',edgecolors='b')
                    ax.scatter(evals[nevs], np.square(evecs[thirdInd, nevs]), facecolor='none',edgecolors='g')


        elif 'final' in self.pltdata:
            # print 'hihi'
            if '1Hh' in self.pltdata:
                someInd = np.argwhere(chunkE == freqFunds[5])[0][0]
                otherInd = np.argwhere(chunkE == freqFunds[6])[0][0]
            elif 'all' in self.pltdata:
                someInd = np.argwhere(chunkE == freqFunds[5])[0][0]
                otherInd = np.argwhere(chunkE == freqFunds[6])[0][0]  # good
                # someInd = np.argwhere(chunkE == freqFunds[6])[0][0]
                # otherInd = np.argwhere(chunkE == freqFunds[6])[0][0] #testing

                # otherInd = np.argwhere(chunkE == freqFunds[6])[0][0]
            elif '1Hw' in self.pltdata:
                someInd = np.argwhere(chunkE == freqFunds[5])[0][0]
                otherInd = np.argwhere(chunkE == freqFunds[6])[0][0]
            else:
                someInd = np.argwhere(chunkE == freqFunds[5])[0][0]
                otherInd = np.argwhere(chunkE == freqFunds[5])[0][0]
            contrib.write('E      I      OH-1  OH-2\n')
            for nevs in range(len(evals)):
                # print 'hi'
                contrib.write('%5.1f %5.4f %5.3f %5.3f\n' %
                              (
                                  evals[nevs],
                                  imat[nevs] / np.amax(imat),
                                  np.square(evecs[someInd, nevs]),
                                  np.square(evecs[otherInd, nevs])
                              )
                              )
            ax = plt.subplot(111)
            ax.scatter(evals, np.square(evecs[someInd]), facecolor='none', edgecolors='r',
                       label="Symmetric Stretch")
            ax.scatter(evals, np.square(evecs[otherInd]), facecolor='none', edgecolors='b',
                       label="Antisymmetric Stretch")
            # for nevs in range(len(evals)):
            #     # print 'hi'
            #     if nevs == 0:
            #         ax.scatter(evals[nevs], np.square(evecs[someInd, nevs]), facecolor='none',edgecolors='r', label="Symmetric Stretch")
            #         ax.scatter(evals[nevs], np.square(evecs[otherInd, nevs]), facecolor='none',edgecolors='b', label="Antisymmetric Stretch")
            #     else:
            #         ax.scatter(evals[nevs], np.square(evecs[someInd, nevs]), facecolor='none',edgecolors='r')
            #         ax.scatter(evals[nevs], np.square(evecs[otherInd, nevs]), facecolor='none',edgecolors='b')

        contrib.close()
        if 'allD' in self.pltdata:
            ax.set_xlim([800, 3000])
        else:
            ax.set_xlim([800, 4000])
        ax.set_ylim([-0.01,1.0])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),scatterpoints=1)
        plt.savefig("ScatterCoefPlots/sharedProtonContributions" + self.pltdata + "_" + str(self.eShift), dpi=500)
        plt.close()


        if 'final' in self.pltdata:
            someIndG = 7
        else:
            someIndG = 9
        listInds = []
        for i in range(len(chunkAssign)):
            if chunkAssign[i][0] >= someIndG and chunkAssign[i][0] >= someIndG:
                listInds.append(i)
        fig = plt.figure()
        ax = plt.subplot(111)
        for nevs in range(len(evals)):
            for a in range(len(listInds)):
                if a == 0 and nevs == 0:
                    ax.scatter(evals[nevs], np.square(evecs[listInds[a], nevs]), facecolor='none', edgecolors='b', label="Low Freq + Low Freq")
                else:
                    ax.scatter(evals[nevs], np.square(evecs[listInds[a], nevs]), facecolor='none', edgecolors='b')
        if 'allD' in self.pltdata:
            ax.set_xlim([800,3000])
        else:
            ax.set_xlim([800, 4000])
        ax.set_ylim([-0.01, 1.0])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),scatterpoints=1)
        plt.savefig("ScatterCoefPlots/lowFreqCombos" + self.pltdata + "_" + str(self.eShift), dpi=500)
        plt.close()

        oneColor = True
        if oneColor:
            self.getScatterCombos_all(chunkE, freqOversCombos,evecs, evals)
        else:

            self.getScatterCombos(chunkE, freqCombosOvers, 'Symm', evecs, evals)
            self.getScatterCombos(chunkE, freqCombosOvers, 'Anti', evecs, evals)
            if 'tet' in self.pltdata:
                self.getScatterCombos(chunkE, freqCombosOvers, 'Anti2', evecs, evals)


    def DiagonalizeHamiltonian(self,chunkE,chunkO,chunkMyMu,freqFunds):
        msk = np.copy(chunkO)
        if 'tet' in self.pltdata:
            v0 = np.load("redH/v_0"+self.pltdata+'.npy')
            msk = msk*v0
        else:
            if 'allH' in self.pltdata:
                msk = msk * 9791.10322743
            elif 'allD' in self.pltdata:
                msk = msk * 7399.02055935
            elif '1Hh' in self.pltdata:
                msk = msk * 7786.35258766
            elif '1He' in self.pltdata:
                msk = msk * 7764.31198935
            elif '1Hw' in self.pltdata:
                msk = msk * 7782.6285036
            elif '1Dh' in self.pltdata or '1Dw' in self.pltdata or '1De' in self.pltdata:
                v0 = np.load("redH/v_0" + self.pltdata + '.npy')
                msk = msk * v0
        np.fill_diagonal(msk,np.zeros(len(msk)))
        chunkE = chunkE-msk

        #add in 2x2 for the allD stuff
        #9    10   1621.9677985 0.0249839
        # twoByTwoO = np.zeros((2,2))
        # twoByTwoH = np.zeros((2,2))
        # twoByTwoO[0,0] = 1.0
        # twoByTwoO[1,1] = 1.0
        # twoByTwoO[0,1] = -0.05296268688541153
        # twoByTwoO[1,0] = -0.05296268688541153
        # twoByTwoH[0,0] = 1623.508163398968
        # twoByTwoH[1,1] = 1621.9677985
        # twoByTwoH[0,1] = -64.64814888561853
        # twoByTwoH[1,0] = -64.64814888561853
        # np.savetxt("twoByTwoO",twoByTwoO)
        # np.savetxt("twoByTwoH",twoByTwoH)

        # print('shloop')
        # tOmat = np.eye(len(chunkO))
        # whereBig0,whereBig1 = np.where(np.abs(chunkO-np.eye(len(chunkO))) > 0.1)
        # whereBig0 = np.array(whereBig0)[len(whereBig0)/2:]
        # whereBig1 = np.array(whereBig1)[len(whereBig1)/2:]
        # tOmat[whereBig0,whereBig1] = np.negative(chunkO[whereBig0,whereBig1])
        # chunkO = tOmat.T.dot(chunkO.dot(tOmat))
        # chunkE = tOmat.T.dot(chunkE.dot(tOmat))
        # # chunkMyMu = tOmat.dot(chunkMyMu)
        # dov = np.copy(np.diag(chunkO))
        # chunkO = chunkO / np.sqrt(dov)
        # chunkO = chunkO / np.sqrt(dov[:,None])
        # chunkE = chunkE / np.sqrt(dov)
        # chunkE = chunkE / np.sqrt(dov[:, None])

        print('floopshloop')
        #1He - special
        #just anti with anti+OO
        # chunkO[9,29] = 0.
        # chunkO[29,9] = 0.
        #weird 9 10 / 9 13
        # chunkO[10, 42] = 0.
        # chunkO[42, 10] = 0.

        #1Hh
        # #####weird#####
        # chunkO[4,3]=0.
        # chunkO[3,4]=0.
        # ################
        #
        # chunkO[10,42]=0.
        # chunkO[42,10]=0.
        # chunkO[23,37] = 0.
        # chunkO[37,23] = 0.
        #1Hw
        # chunkO[19,51]=0.
        # chunkO[51,19]=0.
        # chunkO[29,43] = 0.
        # chunkO[43, 29] = 0.
        #allH
        # chunkO[7, 39] = 0
        # chunkO[39, 7] = 0.
        # chunkO[20,29]=0 #beeg boi, symm sps with anti ihb + anti OO
        # chunkO[29,20]=0 #beeg boi, symm sps with anti ihb + anti OO
        #
        #allD
        # chunkO[10, 41] = 0
        # chunkO[41, 10] = 0.
        # chunkO[35,21]=0 #beeg boi, symm sps with anti ihb + anti OO
        # chunkO[21,35]=0 #beeg boi, symm sps with anti ihb + anti OO


        ####tetramer###
        #all H - 300 cm-1
        # chunkO[16,45]=0.0
        # chunkO[16, 46]=0.0
        # chunkO[44,45]=0.0
        # chunkO[44,46]=0.0
        # chunkO[45,16]=0.0
        # chunkO[46,16]=0.0
        # chunkO[45,44]=0.0
        # chunkO[46,44]=0.0
        #all H - 400 cm-1
        # print('b4')
        # chunkO[20,49]=0.0
        # chunkO[20,50]= 0.0
        # chunkO[21,61]= 0.0
        # chunkO[21,62]= 0.0
        # chunkO[48,49] = 0.0
        # chunkO[48,50] = 0.0
        # #-
        # chunkO[49,20] = 0.0
        # chunkO[49,48] = 0.0
        # chunkO[50,20] = 0.0
        # chunkO[50,48] = 0.0
        # chunkO[61,21] = 0.0
        # chunkO[62,21] = 0.0
        #allH - 600 cm-1
        # print('asdf')
        # idxs=np.where(0.06 <= np.abs(chunkO-np.eye(len(chunkO))))
        # for idcs in range(len(idxs[0])):
        #     chunkO[idxs[0][idcs],idxs[1][idcs]]=0.0
        #allD - 300 cm-1
        if 'tet' in self.pltdata:
            idxss = (0.05 <= np.abs(chunkO-np.eye(len(chunkO)))) * ((0.07 > np.abs(chunkO-np.eye(len(chunkO)))))
            idxs=np.where(idxss)
            for idcs in range(len(idxs[0])):
                chunkO[idxs[0][idcs],idxs[1][idcs]]=0.0
        else:
            if 'allH' in self.pltdata:
                chunkO[7, 39] = 0
                chunkO[39, 7] = 0.
                chunkO[20,29]=0 #beeg boi, symm sps with anti ihb + anti OO
                chunkO[29,20]=0 #beeg boi, symm sps with anti ihb + anti OO
            elif 'allD'  in self.pltdata:
                chunkO[10, 41] = 0
                chunkO[41, 10] = 0.
                chunkO[35,21]=0. #beeg boi, symm sps with anti ihb + anti OO
                chunkO[21,35]=0. #beeg boi, symm sps with anti ihb + anti OO
            else:
                print('fix!!!!!!!!!')
        # chunkO[26,55]=0.0
        # chunkO[26,56] = 0.0
        # chunkO[26,72] = 0.0
        # chunkO[27,67] = 0.0
        # chunkO[27,68] = 0.0
        # chunkO[27,73] = 0.0
        # chunkO[54,55] = 0.0
        # chunkO[55,26] = 0.0
        #
        # chunkO[55,26] = 0.0
        # chunkO[56,26] = 0.0
        # chunkO[72,26] = 0.0
        # chunkO[67,27] = 0.0
        # chunkO[68,27] = 0.0
        # chunkO[73,27] = 0.0
        # chunkO[55,54] = 0.0
        # chunkO[26,55] = 0.0
        #IDENTITY
        # chunkO = np.eye(len(chunkO))

        ev,Q = la.eigh(chunkO)
        # test = la.inv(Q)
        ev12 = np.diag(1/np.sqrt(ev))
        SLAV,SLAEV=sla.eigh(chunkE,chunkO)
        # seval,sevecL,sevecR = sla.eig(chunkE,chunkO,right=True,left=True) #left and right eigenvectors are equal
        SLAEV /= la.norm(SLAEV,axis=0)
        # Hpsi1 = chunkE.dot(SLAEV)
        # EPsi1 = np.dot(np.dot(chunkO,SLAEV),np.diag(SLAV))
        # SLAV2,SLAEV2=sla.eig(chunkE,chunkO)
        S12 = np.matmul(Q,np.matmul(ev12,Q.T)) #Same thing Lindsey does. Project back to "chunkE/chunkO" basis via sim. transform
        fmat = np.dot(S12, np.dot(chunkE, S12)) #S12 is symmetric
        # fmat2[np.where(~np.eye(fmat.shape[0],dtype=bool))] = fmat2[np.where(~np.eye(fmat.shape[0],dtype=bool))]+9791.10322743

        # fmat = self.MartinCouplings(fmat)
        evalz, eveccs = la.eigh(fmat)
        evecsTr = np.dot(S12, eveccs)

        # evecsTr2 = np.dot(eveccs,S12)
        evecsTr /= la.norm(evecsTr,axis=0) 
        # evecsTr = evecsTr.T #test, this used to make stuff work well.

        mumat = evecsTr.T.dot(chunkMyMu)
        # mumat2 = np.copy(mumat)
        # for i in range(len(mumat)):
        #     mumat2[i] = np.dot(evecsTr[:,i],chunkMyMu) #same.

        imat = np.sum(np.square(mumat),axis=1) #square whole thing then sum, since ux uy uz is effectively a norm then.
        # imat2 = la.norm(mumat,axis=1)**2 #same as imat


        # plt.stem(evalz,imat)
        # plt.show()
        # # #######testing####
        # hamVals,hamVecs = la.eigh(chunkE)
        # # hamVecs = hamVecs.T
        # imat = np.sum(np.square(hamVecs.T.dot(chunkMyMu)),axis=1)
        # np.savetxt("chunkMyMu"+self.pltdata,chunkMyMu)
        # evalz = hamVals
        # evecsTr = hamVecs
        # # np.savetxt("HamMat.txt",chunkE)
        # # np.savetxt("dipoles.txt", testMu3)
        # # np.savetxt("FreqInten.txt", zip(evals,imat))
        # # np.savetxt("Evecs.txt", evecsTr)
        # #######testing####

        return evalz,evecsTr,imat

    def getEChunk(self,sortedCouplings, sortedOverlap, sortedMyMus, sortedAssign):
        energy = sortedCouplings[self.center,self.center]
        qual = (np.diag(sortedCouplings) < (energy+self.eShift)) * (np.diag(sortedCouplings) > (energy-self.eShift))
        self.mx = np.where(qual)[0][-1] #for indexing stuff
        self.mn = np.where(qual)[0][0]  #indexing - no zero order state in regular freqs
        self.mnE=sortedCouplings[self.mn,self.mn]
        self.mxE = sortedCouplings[self.mx, self.mx]
        chunkE = sortedCouplings[self.mn:self.mx+1,self.mn:self.mx+1]
        chunkO = sortedOverlap[self.mn:self.mx+1,self.mn:self.mx+1]
        chunkMyMu = sortedMyMus[self.mn:self.mx+1]
        chunkAssign = sortedAssign[self.mn:self.mx+1]
        return chunkE,chunkO,chunkMyMu,chunkAssign


    def participationRatio(self,evals, evecs):
        y = []
        for ecc in range(len(evecs)):
            invrati = 1./(np.sum(np.power(evecs[:,ecc],4)))
            y.append(invrati)
            # plt.scatter(evals[ecc],invrati)
        plt.stem(evals,y,linefmt='k',markerfmt='ko')
        if 'allD' in self.pltdata:
            plt.xlim([800,3000])
        else:
            plt.xlim([800,4000])
        plt.savefig("invParRat/Rats"+self.pltdata+"_"+str(self.eShift))
        plt.close()

    #
    def plotStates(self,totalFreqs):
        if 'allH' in self.pltdata:
            rng = 4000
        else:
            rng = 3000
        rng=4000
        theLen,xx = np.histogram(totalFreqs,bins=(rng-800)/100,range=(800,rng))
        cen = 0.5*(xx[1:]+xx[:-1])
        plt.bar(cen,theLen,width=100,facecolor='grey')
        plt.xlabel("Energy")
        plt.ylabel("Bin Count")

        # plt.hist(totalFreqs,bins=(rng-800)/100,range=(800,rng))
        # plt.stem(totalFreqs, np.ones(len(totalFreqs)), 'k', markerfmt='')
        # # Shared proton?
        # # qual = (totalFreqs > ( - 50.)) * (totalFreqs < (allH + 50.))
        # # asdf = np.where(qual)
        # # print 'allH'
        # # print len(asdf[0])
        # plt.xlim([800, rng])
        plt.savefig('densityOfStates/' + self.pltdata + '_states.png')
        plt.close()
    def weightedHist(self,sortedCoups,freqzF,sortOv):
        sortOv = sortOv[1:,1:]
        sortedCoups = sortedCoups[1:, 1:]
        dgg = np.diag(sortedCoups)
        qual = np.where(((dgg > 800.) * (dgg < 3000)))[0]
        sortedCoups = sortedCoups[qual[0]:qual[-1],qual[0]:qual[-1]]
        sortOv = sortOv[qual[0]:qual[-1],qual[0]:qual[-1]]
        msk = np.copy(sortOv)
        if 'tet' in self.pltdata:
            v0 = np.load("redH/v_0" + self.pltdata + '.npy')
            msk = msk * v0
        else:
            if 'allH' in self.pltdata:
                msk = msk * 9791.10322743
            elif 'allD' in self.pltdata:
                msk = msk * 7399.02055935
            elif '1Hh' in self.pltdata:
                msk = msk * 7786.35258766
            elif '1He' in self.pltdata:
                msk = msk * 7764.31198935
            elif '1Hw' in self.pltdata:
                msk = msk * 7782.6285036

        np.fill_diagonal(msk, np.zeros(len(msk)))
        sortedCoupsp = sortedCoups - msk
        if 'tet' not in self.pltdata:
            symmStr = np.argwhere(sortedCoupsp == freqzF[5])[0, 0]
            antiStr = np.argwhere(sortedCoupsp == freqzF[6])[0, 0]
        else:
            symmStr = np.argwhere(sortedCoupsp == freqzF[6])[0, 0]
            antiStr = np.argwhere(sortedCoupsp == freqzF[7])[0, 0]
            antiStr2 = np.argwhere(sortedCoupsp == freqzF[8])[0, 0]
            coupStrAnti2 = sortedCoupsp[:,antiStr2]
            ovAnti2 = sortOv[:,antiStr2]
        zeroOrd = np.diag(sortedCoupsp)
        coupStrSymm = sortedCoupsp[:,symmStr]
        coupStrAnti = sortedCoupsp[:,antiStr]
        ovSymm = sortOv[:,symmStr]
        ovAnti = sortOv[:,antiStr]

        #symm
        # delIdx = np.where(ovSymm == 1.0)[0]
        # coupStrSymm = np.delete(coupStrSymm,delIdx)
        # ovSymm = np.delete(ovSymm,delIdx)
        # zeroOrdP = np.delete(np.copy(zeroOrd),delIdx)
        ###################################
        # fig, ax1 = plt.subplots()
        #
        # color = 'blue'
        # ax1.set_xlabel('Zero-order Frequency')
        # ax1.set_ylabel('Coupling Strength', color=color)
        # markerline,stemlines,baseline=ax1.stem(zeroOrdP, np.abs(coupStrSymm))
        # for stem in stemlines:
        #     stem.set_linewidth(3)
        # ax1.tick_params(axis='y', labelcolor=color)
        #
        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        # color = 'red'
        # ax2.set_ylabel('Overlap', color=color)  # we already handled the x-label with ax1
        # markerline,stemlines,baseline=ax2.stem(zeroOrdP, np.abs(ovSymm), linefmt='r--', markerfmt='ro')
        # for stem in stemlines:
        #     stem.set_linewidth(3)
        # ax2.tick_params(axis='y', labelcolor=color)
        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.show()
        ####################################
        # plt.stem(zeroOrdP,np.abs(coupStrSymm)/np.amax(np.abs(coupStrSymm)))
        # ovSymm[ovSymm > 0.0] *= -1.0
        # plt.stem(zeroOrdP, ovSymm,linefmt='r-',markerfmt='ro')
        # plt.xlim([800,4000])
        # plt.ylim([-0.01,1.0])
        # plt.show()
        #Anti

        delIdx = np.where(np.around(ovAnti,7) == 1.0)[0]
        coupStrAnti = np.delete(coupStrAnti,delIdx)
        ovAnti = np.delete(ovAnti,delIdx)
        zeroOrdP = np.delete(np.copy(zeroOrd),delIdx)
        # plt.scatter(np.abs(ovAnti),np.abs(coupStrAnti))
        # plt.show()
        # plt.close()
        ###################################
        fig, ax1 = plt.subplots()

        color = 'blue'
        ax1.set_xlabel('Zero-order Frequency')
        ax1.set_ylabel('Coupling Strength', color=color)
        markerline,stemlines,baseline=ax1.stem(zeroOrdP, np.abs(coupStrAnti),markerfmt=' ')
        for stem in stemlines:
            stem.set_linewidth(3)
        ax1.tick_params(axis='y', labelcolor=color)
        if 'tet' in self.pltdata:
            ax1.stem([freqzF[7]],[100],linefmt='k',markerfmt='k')
        else:
            ax1.stem([freqzF[6]], [100], linefmt='k', markerfmt='k')
        ax1.set_ylim([0,100])
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'red'
        ax2.set_ylabel('Overlap', color=color,rotation=270)  # we already handled the x-label with ax1
        markerline,stemlines,baseline=ax2.stem(zeroOrdP, np.abs(ovAnti), linefmt='r--', markerfmt=' ')
        for stem in stemlines:
            stem.set_linewidth(3)
        if 'tet' in self.pltdata:
            ax2.set_ylim([0,0.1])
        else:
            ax2.set_ylim([0,0.2])
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig('CoupsAndOverlaps'+self.pltdata+'.png',dpi=400)
        plt.show()
        ####################################

        # ovAnti[ovAnti < 0] *= -1.0
        # plt.stem(zeroOrdP, ovAnti*np.abs(coupStrAnti / np.amax(np.abs(coupStrAnti))))
        # # plt.stem(zeroOrdP, ovAnti,linefmt='r-',markerfmt='ro')
        # plt.xlim([800,3000])
        # plt.ylim([0,0.15])
        # plt.savefig("coupStrength_"+self.pltdata+".png",dpi=400)
        close
    def includeCouplings(self, freqFunds, freqOvsC, assignments):
        #put in format of overlap matrix and H matrix
        freqOvers=freqOvsC[-self.nvibs:]
        freqCombos= freqOvsC[:-self.nvibs]
        assignmentsp = np.column_stack((np.array([-1,-1]),assignments[:self.nvibs].T)).T
        assignmentsq = np.column_stack((assignmentsp.T,assignments[-self.nvibs:].T)).T
        assignmentsTot = np.column_stack((assignmentsq.T,assignments[self.nvibs:-self.nvibs].T)).T
        assignments = assignmentsTot

        overlap = np.loadtxt(
            'redH/overlapMatrix2_' + self.pltdata + '.dat')
        couplings = np.loadtxt(
            'redH/offDiagonalCouplingsInPotential2_' + self.pltdata+".dat")

        # overlap = np.load('redH/overlap2NOTTRANSPOSED_notNormedffinal_allHrnspc_finalVersion_NegZForYInAllCds.dat.npy')
        # couplings = np.load("redH/ham2NOTTRANSPOSED_notNormedffinal_allHrnspc_finalVersion_NegZForYInAllCds.dat.npy")
        # dgO = np.diag(np.diag(overlap))
        # overlap = overlap+overlap.T-dgO
        # couplings = couplings+couplings.T

        # diagonalFreq = np.concatenate((np.array([0]), freqFunds, freqOvers, freqCombos))
        # np.fill_diagonal(couplings, diagonalFreq)  # lay excitations on diagonal
        # red_overlapOrig = overlapOrig[np.ix_([7,163],[7,163])]
        # red_couplingsOrig = couplingsOrig[np.ix_([7,163],[7,163])]
        # red_overlap = overlap[np.ix_([7,163],[7,163])]
        # red_couplings = couplings[np.ix_([7,163],[7,163])]
        # np.fill_diagonal(red_couplingsOrig,diagonalFreq[[7,163]])
        # print('h')
        #########TESTING############
        # overlap, couplings = self.setOffDiags(6 + 1, overlap, couplings,killall=True)
        # overlap, couplings = self.setOffDiags(6+1, overlap, couplings)
        # overlap, couplings = self.setOffDiags(7+1, overlap, couplings)
        # overlap,couplings=self.setOffDiags(8+1,overlap,couplings)
        ############################
        fundMu = np.loadtxt('redH/1mu0'+ self.pltdata)
        overMu = np.loadtxt('redH/2mu0'+ self.pltdata)
        comboMu = np.loadtxt('redH/11mu0p'+ self.pltdata)
        # print fundMu.shape,overMu.shape,comboMu.shape
        diagonalFreq = np.concatenate((np.array([0]),freqFunds, freqOvers,freqCombos))
        np.fill_diagonal(couplings, diagonalFreq) #lay excitations on diagonal
        # mus = np.zeros((len(fundMu)+len(overMu)+len(comboMu)+1,len(fundMu)+len(overMu)+len(comboMu)+1,3))
        myMus = np.vstack((np.array([0, 0, 0]),fundMu, overMu,comboMu))
        # myMus = np.vstack((np.array([0, 0, 0]), fundMu, np.zeros((len(overMu),3)), np.zeros((len(comboMu),3))))
        if 'tet' not in self.pltdata:
            nonZ = fundMu[6]
            fundMu = np.zeros(fundMu.shape)
            fundMu[6] = nonZ
        else:
            nonZs = [fundMu[7],fundMu[8]]
            fundMu = np.zeros(fundMu.shape)
            fundMu[7] = nonZs[0]
            fundMu[8] = nonZs[1]

        myMus = np.vstack((np.array([0, 0, 0]), fundMu, np.zeros(overMu.shape), np.zeros(comboMu.shape)))

        np.savetxt("myMus"+self.pltdata,myMus)
        intz = np.sum(np.square(myMus),axis=1)
        sumIntz = np.sum(intz)
        print "SUMINTZ - PREMIX",sumIntz
        # for p in range(mus.shape[0]):
        #     mus[p,p,:] = myMus[p]
        idx = np.argsort(np.diag(couplings))
        # idxprime = diagonalFreq.argsort() #good
        sortedAssign = assignments[idx]
        sortedOverlap = overlap[idx,:][:,idx]
        # sortedMus = mus[idx,:,:][:,idx,:]
        sortedCouplings = couplings[idx, :][:, idx]
        sortedMyMus = myMus[idx]
        if 'tet' in self.pltdata: #tetramer
            if 'all' in self.pltdata:
                self.center=np.argwhere(sortedCouplings==freqFunds[7])[0,0]
                if 'allD' in self.pltdata:
                    self.center = np.argwhere(sortedCouplings == freqFunds[6])[0, 0]
            elif '1He' in self.pltdata or '1De' in self.pltdata:  # tetramer
                # self.center = np.argwhere(sortedCouplings == freqFunds[3])[0, 0] #center to blobby mes
                self.center = np.argwhere(sortedCouplings == freqFunds[0])[0,0]

                # self.center = np.argwhere(sortedCouplings == freqCombos[110])[0, 0]
                print 'hi'
                # self.center = np.argwhere(sortedCouplings == freqFunds[6])[0, 0]


            else:
                #Mix about 2600 - 300/350 mix
                # self.center = np.argwhere(sortedCouplings == freqFunds[6])[0, 0]
                #2658
                self.center = np.argwhere(sortedCouplings == freqFunds[5])[0, 0]
        else:
            if '1He' in self.pltdata:
                self.center = np.argwhere(sortedCouplings == freqCombos[132])[0, 0]  # good one
            else:
                self.center = np.argwhere(sortedCouplings == freqFunds[5])[0, 0]  # good one

        #
        # self.weightedHist(sortedCouplings,freqFunds,sortedOverlap)
        #

        self.centerE = sortedCouplings[self.center,self.center]
        chunkE,chunkO,chunkMyMu,chunkAssign = self.getEChunk(sortedCouplings,sortedOverlap,sortedMyMus,sortedAssign)
        flll = open("Matrices/chunkAssign_"+str(self.eShift)+"_"+self.pltdata,'w+')
        freqsForWrite = np.diag(chunkE)
        chunkCopy = np.copy(chunkE)
        chunkForWrite = (chunkCopy - np.diag(np.diag(chunkCopy)))
        flll.write("Matrix index / Freq / State assign / Highest off diag / Highest Index\n")
        for i in range(len(chunkE)):
            print(i)
            k = np.where(np.abs(chunkForWrite) == np.amax(np.abs(chunkForWrite[i:, i])))[0][1]
            flll.write("%d %5.12f %d %d  %5.5f %d \n" % (i,freqsForWrite[i],chunkAssign[i][0],chunkAssign[i][1],np.amax(np.abs(chunkForWrite[i:,i])), k))
        flll.close()

        self.plotOverlapAndCouplings(chunkO, chunkE)
        self.getLargeCouplings(chunkE,chunkAssign,200)
        self.getLargeCouplings(chunkO-np.eye(len(chunkO)), chunkAssign, 0.05)

        #old
        # self.analyzeHmat(chunkE,chunkAssign,freqFunds)
        # chunkE=self.MartinCouplings(chunkE)
 
        evals,evecs,imat = self.DiagonalizeHamiltonian(chunkE,chunkO,chunkMyMu,freqFunds)
        self.participationRatio(evals,evecs)
        if 'tet' in self.pltdata and  ('1Hw' in self.pltdata or '1Dw' in self.pltdata):
                print('skipping contribs')
        else:
            self.getContributionsAndPlot(chunkE, freqFunds, np.concatenate((freqOvers,freqCombos)),imat, evals, evecs, chunkAssign)
        self.mn-=1
        self.mx-=1 #for regular freqs, no "ground state"
        return evals, imat

    def MartinCouplings(self,chunkE):
        strong = np.zeros((len(chunkE),len(chunkE)))
        np.fill_diagonal(strong,np.diag(chunkE))
        for i in range(len(chunkE)-1):
            gamma=np.power(chunkE[i,(i+1):],4)/np.power(chunkE[i,i]-np.diag(chunkE)[(i+1):],4)
            if np.amin(gamma) < 0.0:
                stop
            good= gamma>1.0
            good = np.array([False for _ in range(i+1)]+list(good))
            strong[i,good]=chunkE[i,good]
            print 'good_'+str(good.sum())
        stronk = strong+strong.T-np.diag(np.diag(strong))
        return strong+strong.T-np.diag(np.diag(strong))

    def gauss(self, E, freq, inten,newFreqq,overtone=False):
        # inten*=2
        #plt.stem(freq,inten)
        #plt.savefig('prelim')
        if 'allH' in self.pltdata:
            broad=25
        elif 'allD' or 'final' in self.pltdata:
            broad = 20
        if overtone:
            broad= 10
        g = np.zeros(len(E))
        artShift = False
        if artShift:
            print 'ADDING ARTIFICIAL SHIFT'
            # self.pltdata=self.pltdata+'shifted'
            # arbsh = (2600-1750-250-150-50) #400
            # if 'allH' and 'tet' in self.pltdata:
            #     idd=np.argwhere(np.logical_and(freq>=1750, freq<=2700))
            #     freq[idd]+=arbsh
            # elif 'allD' and 'tet' in self.pltdata:
            #     idd=np.argwhere(np.logical_and(freq>=1375, freq<=2325))
            #     freq[idd]+=(1/np.sqrt(2))*arbsh #282

        for i in range(len(freq)):
            g += inten[i] * np.exp(-np.square(E - freq[i]) / (2.0*np.square(broad)))
        if np.amax(g) > 1:
            g/=np.amax(g)
        plt.rc('text', usetex=True)
        plt.rcParams.update({'font.size': 16})
        plt.plot(E, g, color='k',label='Ground State Approx.')
        # plt.show()
        # plt.fill_between(E, g, color='r', alpha=0.5)
        # plt.show()
        if overtone:
            plt.ylim([0,1])

            # plt.xlim([3400,4800])
            # broad= 10
            # plt.xticks(np.arange(3400,4800, 400))


            # plt.xlim([4800, 8000])
            # broad= 20
            # plt.xticks(np.arange(4800, 8000, 400))

            #allH
            plt.xlim([5600, 8000])
            broad = 15
            plt.xticks(np.arange(5600, 8000, 400))
        else:
            if '1He' in self.pltdata and 'tet' in self.pltdata:
                plt.xlim([2200,4000])
            else:
                plt.xlim([np.amin(E), np.amax(E)+1])
                plt.xticks(np.arange(np.amin(E),np.amax(E)+10,400))
            plt.ylim([0, 2])
        #     #plt.title(self.pltdata)
        plt.ylabel(r'Rel. Intensity')
        plt.xlabel(r'Energy (cm$^{-1}$)')
        plt.gca().tick_params(axis='x',pad=10)
        if self.mix:
            if self.stix:
                idx = np.in1d(freq,newFreqq)
                plt.stem(freq[idx],inten[idx],'r',markerfmt=" ",basefmt=" ")
                plt.stem(freq[np.invert(idx)],inten[np.invert(idx)],'k',markerfmt=" ",basefmt=" ")
            plt.plot([self.mnE,self.mxE],[1.1,1.1],'r')
            if self.pltVCI:
                if 'allH' in self.pltdata:
                    cf = 'allH'
                elif 'allD' in self.pltdata:
                    cf = 'allD'
                else:
                    noChinhdataYo
                ctrim = np.loadtxt("ChinhTrimer/Chinh_freqI_Trimer_" + cf)
                cfreq = ctrim[:, 0]
                cInt = ctrim[:, 1] / np.amax(ctrim[:, 1])
                ch = np.zeros(len(E))
                for c in range(len(cfreq)):
                    ch += cInt[c] * np.exp(-np.square(E - cfreq[c]) / (np.square(broad)))
                plt.plot(E, ch / np.amax(ch), 'k', label='VCI/VSCF')
                # plt.fill(E, ch/np.amax(ch), color='b', alpha=0.5)

                plt.legend()
                # plt.stem(cfreq,cInt, 'g', markerfmt=" ", basefmt=" ")
                plt.savefig('SpectraPix/' + self.pltdata + '_' + self.pltdata+'_MIX_stem_chinh_newnewnew.png',dpi=500)
            else:
                # plt.fill_between(E, g, color='blue', alpha=0.5)
                plt.savefig('SpectraPix/' + self.pltdata + '_' + self.pltdata +'_MIX_shift'+str(self.eShift)+''+'.png',dpi=500)
        else:
            if self.stix:
                plt.stem(freq,inten, 'k', markerfmt=" ", basefmt=" ",label='_nolegend_')

            if self.pltVCI:
                if 'allH' in self.pltdata:
                    cf = 'allH'
                elif 'allD' in self.pltdata:
                    cf = 'allD'
                else:
                    noChunhDatyO
                ctrim = np.loadtxt("ChinhTrimer/Chinh_freqI_Trimer_" + cf)
                cfreq = ctrim[:, 0]
                cInt = ctrim[:, 1] / np.amax(ctrim[:, 1])
                ch = np.zeros(len(E))
                for c in range(len(cfreq)):
                    ch += cInt[c] * np.exp(-np.square(E - cfreq[c]) / (np.square(broad)))
                plt.plot(E, ch / np.amax(ch), 'k', label='VCI/VSCF')
                # plt.fill(E, ch/np.amax(ch), color='b', alpha=0.5)

                plt.legend()
                # plt.stem(cfreq,cInt, 'g', markerfmt=" ", basefmt=" ")
                plt.savefig('SpectraPix/' + self.cfg+ '_' + self.pltdata+'_NoMIX_stem_chinh.png',dpi=500)
            else:
                if not overtone:
                    plt.savefig('SpectraPix/' + self.cfg + '_' + self.pltdata+'.png',dpi=500)
                else:
                    plt.savefig('SpectraPix/' + self.cfg + '_' + self.pltdata + '_overtone5600.png', dpi=500)
                # plt.xlim([3400,4000])
                # plt.ylim([0,0.4])
                # plt.savefig('SpectraPix/' + self.cfg + '_' + self.pltdata+'_highFreq.png',dpi=500)
        plt.close()

    # def gauss_OC(self, E, freq, inten,less):
    #     g = np.zeros(len(E))
    #     for i in range(len(freq)):
    #         g += inten[i] * np.exp(-np.square(E - freq[i]) / (np.square(10)))
    #     if np.amax(g) > 1:
    #         g/=np.amax(g)
    #
    #     #plt.ylim([0,.2])
    #     plt.ylim([0,.006])
    #     #plt.xlim([3400,4600])
    #     plt.xlim([4800,8000])
    #     # plt.xticks(np.arange(3400,4600, 400))
    #     plt.xticks(np.arange(4800,8000, 400))
    #
    #         #plt.title(self.pltdata)
    #     plt.ylabel('Rel. Intensity')
    #     plt.xlabel('$cm^{-1} $')
    #     plt.gca().tick_params(axis='x',pad=10)
    #     plt.plot(E, g, color='k',label='Ground State Approx.',linewidth=2)
    #     #plt.fill(E, g, color='r', alpha=0.5)
    #     #plt.stem(np.append(freq[:(self.center - self.shift-less)], freq[(self.center + self.shift-less):]),np.append(inten[:(self.center - self.shift-less)], inten[(self.center + self.shift-less):]), 'b', markerfmt=" ", basefmt=" ",label='_nolegend_')
    #     #plt.stem(freq[self.center - self.shift-less:self.center + self.shift-less], inten[self.center - self.shift-less:self.center + self.shift-less], 'r', markerfmt=" ", basefmt=" ",label='_nolegend_')
    #     plt.rcParams.update({'font.size':20})
    #     if self.mix:
    #         if self.stix:
    #             plt.stem(np.append(freq[:(self.center - self.shift-less)], freq[(self.center + self.shift-less):]),np.append(inten[:(self.center - self.shift-less)], inten[(self.center + self.shift-less):]), 'b', markerfmt=" ", basefmt=" ",label='_nolegend_')
    #             plt.stem(freq[self.center - self.shift-less:self.center + self.shift-less], inten[self.center - self.shift-less:self.center + self.shift-less], 'g', markerfmt=" ", basefmt=" ",label='_nolegend_')
    #         if self.pltVCI:
    #             if 'allH' in self.pltdata:
    #                 cf = 'allH'
    #             else:
    #                 cf = 'allD'
    #             ctrim = np.loadtxt("ChinhTrimer/Chinh_freqI_Trimer_" + cf)
    #             cfreq = ctrim[:, 0]
    #             cInt = ctrim[:, 1] / np.amax(ctrim[:, 1])
    #             ch = np.zeros(len(E))
    #             for c in range(len(cfreq)):
    #                 ch += cInt[c] * np.exp(-np.square(E - cfreq[c]) / (np.square(10)))
    #             plt.plot(E, ch / np.amax(ch), 'k', label='VCI/VSCF')
    #             plt.fill(E, ch/np.amax(ch), color='b', alpha=0.5)
    #
    #             plt.legend()
    #             # plt.stem(cfreq,cInt, 'g', markerfmt=" ", basefmt=" ")
    #             plt.savefig('SpectraPix/SmoothedNew/HighFreq/hf__' + self.pltdata + '_' + self.pltdata+'_MIX_stem_chinh_newnewnew.png',dpi=500)
    #         else:
    #             if self.stix:
    #                 plt.stem(freq,inten, 'g', markerfmt=" ", basefmt=" ",label='_nolegend_')
    #                 plt.savefig('SpectraPix/SmoothedNew/HighFreq/hf__' + self.pltdata + '_' + self.pltdata +'_MIX_.png',dpi=500)
    #     else:
    #         plt.savefig('SpectraPix/SmoothedNew/HighFreq/hf__' + self.pltdata + '_' + self.pltdata+'.png',dpi=500)
    #     plt.close()

    def writeAssignSorts(self,asn,fre,inte):
        fl = open("assignmentsPreMix/assignSorted"+self.cfg+self.pltdata,"w+")
        for k in reversed(range(len(asn))):
            fl.write("%-4d %-4d %-7.7f %-7.7f\n" % (asn[k,0],asn[k,1],fre[k],inte[k]))
        fl.close()

    def plotSpec(self):
        interest = np.loadtxt(FundamentalShell + self.pltdata) #load in fundamental frequencies
        interestfreqs = interest[:, 1] #load in frequencies themselves
        interestInts = interest[:, 2] #load in intensities themselves
        interestAssign = np.column_stack((interest[:,0],np.array(np.repeat(999,len(interest[:,0]))))) #load in fundamental assignments
        interest2 = np.loadtxt(ComboShell + self.pltdata + ".data2") #load in combos AND overtones at the end
        interestfreqs2 = interest2[:, 0] #load in frequencies
        interestInts2 = interest2[:, 1] #load in intensities combinations then overtones
        interestAssign2 = interest2[:, 2:] # load in assignments
        np.set_printoptions(threshold=np.nan)
        if not self.OC:
            if 'allD' in self.pltdata:
                print 'exception activated'
                normParam=600
                E = np.linspace(600, 3000,4800)
            else:
                normParam=800
                E = np.linspace(800, 4000, 6400)
            if 'tet' in self.pltdata and ('1H' in self.pltdata or '1D' in self.pltdata):
                normParam = 2200
                E = np.linspace(2200, 4000, 3600)
        else:
            # E = np.linspace(3400, 4800, 8800)
            # normParam = 3400
            # E = np.linspace(4800, 8000, 8800)
            # normParam = 4800

            #allH
            E = np.linspace(5600, 8000, 8800)
            normParam = 5600
        totalInts = np.concatenate((interestInts, interestInts2))
        totalFreqs = np.concatenate((interestfreqs, interestfreqs2))
        self.plotStates(totalFreqs)
        totalAssign = np.concatenate((interestAssign,interestAssign2))
        sortInd = totalInts.argsort()
        sortedTotalFreqs = totalFreqs[sortInd]
        sortedTotalInts = totalInts[sortInd]
        sortedTotalAssign = totalAssign[sortInd]
        if self.mix:
            newFreqs, newInts = self.includeCouplings(interestfreqs, interestfreqs2,totalAssign)

            # totalInts = np.concatenate((interestInts, interestInts2))
            # totalFreqs = np.concatenate((interestfreqs, interestfreqs2))
            sortInd = totalFreqs.argsort()
            sortedTotalFreqs = totalFreqs[sortInd]
            sortedTotalInts = totalInts[sortInd]
            sortedTotalFreqs[self.mn:self.mx+1] = newFreqs
            sortedTotalInts[self.mn:self.mx+1] = newInts #no longer sorted
            print "SUMINTZ - POST MIX", np.sum(sortedTotalInts)
            sortedTotalInts = self.normAll2(sortedTotalFreqs,sortedTotalInts,normParam)
            # interestfreqsp, interestIntsp, lose = self.normAll(sortedTotalFreqs, sortedTotalInts, normParam)
        else:
            interestfreqsp, interestIntsp, lose, sortedTotalAssignp = self.normAll(sortedTotalFreqs, sortedTotalInts,normParam, sortedTotalAssign)
            self.writeAssignSorts(sortedTotalAssignp,interestfreqsp,interestIntsp)

            sortedTotalInts = self.normAll2(sortedTotalFreqs,sortedTotalInts,normParam)
            newFreqs = None

        self.gauss(E, sortedTotalFreqs, sortedTotalInts, newFreqs,self.OC)
# before = np.loadtxt("chunkMyMufinal_allHrnspc_finalVersion_SPC_BasedOnOs")
# after_full = np.loadtxt("chunkMyMufinal_allHrnspc_finalVersion_SPC_BasedOnOs_PrincAx_fullEckDips")
# after_justO = np.loadtxt("chunkMyMufinal_allHrnspc_finalVersion_SPC_BasedOnOs_PrincAx_justODips")
# before = np.loadtxt("myMusfinal_allHrnspc_finalVersion_SPC_BasedOnOs")[1:]
# after_full = np.loadtxt("myMusfinal_allHrnspc_finalVersion_SPC_BasedOnOs_PrincAx_fullEckDips")[1:]
# after_justO = np.loadtxt("myMusfinal_allHrnspc_finalVersion_SPC_BasedOnOs_PrincAx_justODips")[1:]
# before = np.loadtxt("myMusfinal_allHrnspc_finalVersion_SPC_BasedOnOs")[1:]
# after = np.loadtxt("myMusxTfinal_allHrnspc_finalVersion")[1:]

print('hi')
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1,projection='3d')
# ax.scatter(before[:,0],before[:,1],before[:,2])
# # ax.scatter(after_justO[:,0],after_justO[:,1],after_justO[:,2],c='r')
# ax.set_xlabel("Mu X")
# ax.set_ylabel("Mu y")
# ax.set_zlabel("Mu z")
# plt.show()
# plt.close()
#
# fig = plt.figure(figsize=plt.figaspect(0.3))
#
# ax = fig.add_subplot(1, 3, 1, projection='3d')
# ax.scatter(before[:,0],before[:,1],before[:,2])
# ax.set_xlabel("Mu X")
# ax.set_ylabel("Mu y")
# ax.set_zlabel("Mu z")
# ax = fig.add_subplot(1, 3, 2, projection='3d')
# ax.scatter(after_full[:,0],after_full[:,1],after_full[:,2])
# ax.set_xlabel("Mu X")
# ax.set_ylabel("Mu y")
# ax.set_zlabel("Mu z")
# ax = fig.add_subplot(1, 3, 3, projection='3d')
# ax.scatter(after_justO[:,0],after_justO[:,1],after_justO[:,2])
# ax.set_xlabel("Mu X")
# ax.set_ylabel("Mu y")
# ax.set_zlabel("Mu z")
# plt.show()
# print('hi')
"""def __init__(self, cfg, pltdata, pltVCI=True,stix=False):"""
#
# mp = myPlot('Trimer_fBest','ffinal_allHrnspc_ffinalVersion',mix=False,stix=False,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('Trimer_fBest','ffinal_allHrnspc_ffinalVersion',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('Trimer_fBest','ffinal_allDrnspc_ffinalVersion',mix=False,stix=False,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('Trimer_fBest','ffinal_allDrnspc_ffinalVersion',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()
# #
# mp = myPlot('Trimer_fBest','ffinal__1Hernspc_ffinalVersion',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('Trimer_fBest','ffinal__1Hernspc_ffinalVersion',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('Trimer_fBest','ffinal__1Hwrnspc_ffinalVersion',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('Trimer_fBest','ffinal__1Hwrnspc_ffinalVersion',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('Trimer_fBest','ffinal__1Hhrnspc_ffinalVersion',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('Trimer_fBest','ffinal__1Hhrnspc_ffinalVersion',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()
#
mp = myPlot('Trimer_fBest','ffinal__1Dhrnspc_ffinalVersion',mix=False,stix=True,pltVCI=False)
mp.plotSpec()

# mp = myPlot('Trimer_fBest','ffinal__1Dhrnspc_ffinalVersion',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()

mp = myPlot('Trimer_fBest','ffinal__1Dernspc_ffinalVersion',mix=False,stix=True,pltVCI=False)
mp.plotSpec()

mp = myPlot('Trimer_fBest','ffinal__1Dernspc_ffinalVersion',mix=True,stix=True,pltVCI=False)
mp.plotSpec()

mp = myPlot('Trimer_fBest','ffinal__1Dwrnspc_ffinalVersion',mix=False,stix=True,pltVCI=False)
mp.plotSpec()

mp = myPlot('Trimer_fBest','ffinal__1Dwrnspc_ffinalVersion',mix=True,stix=True,pltVCI=False)
mp.plotSpec()



#
# mp = myPlot('Tetramer_fBest','tetffinal_allHrnspc_f_finalVersion',mix=False,stix=False,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('Tetramer_fBest','tetffinal_allHrnspc_f_finalVersion',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('Tetramer_fBest','tetffinal_allDrnspc_f_finalVersion',mix=False,stix=False,pltVCI=False)
# mp.plotSpec()
# #
# mp = myPlot('Tetramer_fBest','tetffinal_allDrnspc_f_finalVersion',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()



#
# mp = myPlot('Tetramer_fBest','tetffinal__1Hernspc_f_finalVersion',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('Tetramer_fBest','tetffinal__1Hernspc_f_finalVersion',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('Tetramer_fBest','tetffinal__1Hwrnspc_f_finalVersion',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('Tetramer_fBest','tetffinal__1Hwrnspc_f_finalVersion',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()

###########Overtone###################
# mp = myPlot('Tetramer_fBest','tetffinal_allHrnspc_f_finalVersion',mix=False,stix=False,pltVCI=False,OC=True)
# mp.plotSpec()

# mp = myPlot('Trimer_fBest','ffinal_allHrnspc_ffinalVersion',mix=False,stix=True,pltVCI=False,OC=True)
# mp.plotSpec()
