import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import numpy.linalg as la
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
        print self.cfg
        self.pltdata = pltdata  # intending to be a file with the format I like, with freq       intensity
        self.mix = mix
        self.pltVCI = pltVCI  # true or false
        self.pltVCI = pltVCI  # true or false
        self.stix = stix
        self.OC=OC

        if 'tet' in self.pltdata:
            self.nvibs = 33
            if '1He'in self.pltdata:
                self.eShift = 300 #350
            elif 'allH'in self.pltdata:
                self.eShift = 600 #350
            elif 'allD' in self.pltdata:
                self.eShift = 300 #300
            else:
                self.eShift = 100
        elif 'final' in self.pltdata:
            self.nvibs = 24
            if 'allH' in self.pltdata:
                self.eShift = 600 #600 for newest?
                # if 'Then' in self.pltdata:
                #     self.eShift = 500  # 450
            elif 'allD' in self.pltdata:
                self.eShift = 350
            elif '1Hw' in self.pltdata or '1Dw' in self.pltdata:
                self.eShift = 400 #300
            elif '1He' in self.pltdata or '1De' in self.pltdata:
                self.eShift = 300
            elif '1Hh' in self.pltdata or '1Dh' in self.pltdata:
                self.eShift = 300 #400


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
            np.savetxt("bigCouplings/bigCoupleAssignments" + self.pltdata, bigCouplings[bigCouplings[:,0].argsort()], fmt='%i')

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
        print np.amax(chunkO - np.eye(len(chunkO)))
        plt.matshow(chunkO - np.eye(len(chunkO)))
        plt.colorbar()
        # plt.clim(-1,1)
        plt.savefig('Matrices/sortedOverlaps_' + self.pltdata,dpi=500)
        plt.close()
        plt.matshow(chunkE)
        plt.colorbar()
        plt.savefig('Matrices/sortedCouplings_' + self.pltdata,dpi=500)
        plt.close()

    def getScatterCombos(self,chunkE,freqCombosOvers,mode,evecs,evals):
        if 'tet' in self.pltdata:
            if mode == "Symm":
                fund = 6
            elif mode == "Anti":
                fund = 7
            elif mode == "Anti2":
                fund = 8
        else:
            if mode == "Symm":
                fund = 5
            elif mode == "Anti":
                fund = 6
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
        assignz = {}
        k=0
        with open("TransformationMatrix/newest/csv/TransformationMatrix"+self.pltdata+".csv",'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if k != 0:
                    assignz[k-1] = row[1]
                k+=1

        whereMatch = np.in1d(freqCombosOvers , np.diag(chunkE))
        whereMatch2 = np.where(whereMatch)[0] #index where freqCombosOvers matches chunkE
        evecInd = []
        for state in whereMatch2:
            evecInd.append(np.where(np.diag(chunkE)==freqCombosOvers[state])[0][0])
        evecInd = np.array(evecInd)
        assignList = []
        for (x,y) in itertools.combinations(np.arange(self.nvibs),2):
            assignList.append([x,y])
        for x in np.arange(self.nvibs):
            assignList.append([x,x])
        assignList = np.array(assignList) #full combo and overtone indices

        finalAssign = assignList[whereMatch2] #get combo numbers
        indList = finalAssign[np.where(finalAssign[:,0] == fund)] #for which mode
        indListAssign = []
        for q in indList:
            indListAssign.append(assignz[q[1]]) #assign

        colors = iter(cm.rainbow(np.linspace(0, 1, len(indList))))
        fig = plt.figure()
        ax = plt.subplot(111)
        for gh in range(len(indList)):
            # if gh == 0:
            #     col=[0,0,0,0]
            # elif gh == 1:
            #     col = [0, 0, 0, 0.5]
            # else:
            col = next(colors)
            for nevs in range(len(evals)):
                if nevs == 0:
                    # plt.scatter(evals[nevs], np.square(evecs[indList[gh], nevs]), c=col,label=indListAssign[gh])
                    ax.scatter(evals[nevs], np.square(evecs[evecInd[gh], nevs]), c=col, label=indListAssign[gh])
                else:
                    # plt.scatter(evals[nevs], np.square(evecs[indList[gh], nevs]), c=col)
                    ax.scatter(evals[nevs], np.square(evecs[evecInd[gh], nevs]), c=col)
        if 'allD' in self.pltdata:
            ax.set_xlim([800,3000])
        else:
            ax.set_xlim([800, 4000])
        ax.set_ylim([-0.001,1.0])
        box = ax.get_position()
        ax.set_position([box.x0,box.y0, box.width * 0.7,box.height])
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),scatterpoints=1, fontsize=8)
        # plt.legend(scatterpoints=1, fontsize=8)
        plt.savefig("ScatterCoefPlots/LowFreqContributions" + self.pltdata + assignz[indList[0,0]].replace(" ","")+"_shift"+str(self.eShift),dpi=500)
        plt.close()


    def getContributionsAndPlot(self,chunkE,freqFunds,freqCombosOvers,imat,evals,evecs,chunkAssign):
        self.getScatterCombos(chunkE,freqCombosOvers,'Symm',evecs,evals)
        self.getScatterCombos(chunkE, freqCombosOvers, 'Anti',evecs,evals)
        if 'tet' in self.pltdata:
            self.getScatterCombos(chunkE, freqCombosOvers, 'Anti2', evecs, evals)

        topC = open("topContributions/contr_"+self.pltdata,"w+")
        topC.write('E     I      State (pair)   Coef^2\n')
        for nevs in range(len(evals)):
            stAs=chunkAssign[np.argmax(np.square(evecs[:, nevs]))]
            topC.write('%5.1f, %5.4f, %2.0f %2.0f , %5.3f\n' % (evals[nevs],imat[nevs],stAs[0],stAs[1],np.square(np.amax(evecs[:,nevs]))))
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
            allC.write('%5.1f, %5.4f\n '% (evals[nevs],imat[nevs]))
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
            someInd = np.argwhere(chunkE == freqFunds[6])[0][0]
            if '1He' not in self.pltdata:
                otherInd = np.argwhere(chunkE == freqFunds[7])[0][0]
                thirdInd = np.argwhere(chunkE == freqFunds[8])[0][0]
            else:
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
                print 'hi'
                if nevs == 0:
                    ax.scatter(evals[nevs],np.square(evecs[someInd,nevs]),c='r',label="Totally Symmetric Stretch")
                    ax.scatter(evals[nevs], np.square(evecs[otherInd,nevs]),c='b',label="Antisymmetric Stretch 1")
                    ax.scatter(evals[nevs], np.square(evecs[thirdInd,nevs]),c='g',label="Antisymmetric Stretch 2")
                else:
                    ax.scatter(evals[nevs], np.square(evecs[someInd, nevs]), c='r')
                    ax.scatter(evals[nevs], np.square(evecs[otherInd, nevs]), c='b')
                    ax.scatter(evals[nevs], np.square(evecs[thirdInd, nevs]), c='g')


        elif 'final' in self.pltdata:
            print 'hihi'
            if '1Hh' in self.pltdata:
                someInd = np.argwhere(chunkE == freqFunds[5])[0][0]
                otherInd = np.argwhere(chunkE == freqFunds[6])[0][0]
            elif 'all' in self.pltdata:
                someInd = np.argwhere(chunkE == freqFunds[5])[0][0]
                otherInd = np.argwhere(chunkE == freqFunds[6])[0][0]
            elif '1Hw' in self.pltdata:
                someInd = np.argwhere(chunkE == freqFunds[5])[0][0]
                otherInd = np.argwhere(chunkE == freqFunds[6])[0][0]
            else:
                someInd = np.argwhere(chunkE == freqFunds[5])[0][0]
                otherInd = np.argwhere(chunkE == freqFunds[5])[0][0]
            contrib.write('E      I      OH-1  OH-2\n')
            for nevs in range(len(evals)):
                print 'hi'
                contrib.write('%5.1f %5.4f %5.3f %5.3f\n' %
                              (
                                  evals[nevs],
                                  imat[nevs] / np.amax(imat),
                                  np.square(evecs[someInd, nevs]),
                                  np.square(evecs[otherInd, nevs])
                              )
                              )
            fig = plt.figure()
            ax = plt.subplot(111)
            for nevs in range(len(evals)):
                print 'hi'
                if nevs == 0:
                    ax.scatter(evals[nevs], np.square(evecs[someInd, nevs]), c='r', label="Symmetric Stretch")
                    ax.scatter(evals[nevs], np.square(evecs[otherInd, nevs]), c='b', label="Antisymmetric Stretch")
                else:
                    ax.scatter(evals[nevs], np.square(evecs[someInd, nevs]), c='r')
                    ax.scatter(evals[nevs], np.square(evecs[otherInd, nevs]), c='b')

        contrib.close()
        if 'allD' in self.pltdata:
            ax.set_xlim([800, 3000])
        else:
            ax.set_xlim([800, 4000])
        ax.set_ylim([-0.001,1.0])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), scatterpoints=1, fontsize=8)
        plt.savefig("ScatterCoefPlots/sharedProtonContributions"+self.pltdata+"_"+str(self.eShift),dpi=500)
        plt.close()


    def DiagonalizeHamiltonian(self,chunkE,chunkO,chunkMu):
        print 'original Freqs\n', np.diagonal(chunkE)
        ev,Q = la.eigh(chunkO)
        # test = la.inv(Q)
        ev12 = np.diag(1/np.sqrt(ev))
        # SLAV,SLAEV=sla.eigh(chunkE,chunkO)
        #S12 = np.matmul(Q.T, np.matmul(E12, Q))  # !?!?!??!?!
        S12 = np.matmul(Q,np.matmul(ev12,Q.T)) #Same thing Lindsey does. Project back to "chunkE/chunkO" basis via sim. transform
        fmat = np.dot(S12, np.dot(chunkE, S12))
        # fmat = self.MartinCouplings(fmat)
        # plt.matshow(fmat)
        # plt.colorbar()
        # plt.savefig('OverlapHMatrix.png')
        # plt.matshow(chunkE)
        # plt.colorbar()
        # plt.savefig('RegularHMatrix.png')
        evals, evecs = la.eigh(fmat)

        # test = la.norm(evecs,axis=0)
        # test2 = la.norm(evecs,axis=1)
        evecsTr = np.dot(S12, evecs)
        evecsTr /= la.norm(evecsTr,axis=0)

        plt.matshow(np.square(evecsTr)) #22 and 33 for IHB a and s
        plt.colorbar()
        plt.savefig("evecsTr_"+self.pltdata+".png",dpi=500)
        plt.close()

        evals1,evecs1 = sla.eigh(chunkE,chunkO)
        # plt.matshow(np.square(evecs1))
        # plt.colorbar()
        # plt.show()

        # testMu2 = np.diagonal(chunkMu)
        testMu3 = np.diagonal(chunkMu).T
        mumat1 = np.dot(evecs, np.dot(S12, testMu3)) #CHECKECHECK
        mumat2 = np.dot(S12,evecs.dot(testMu3)) #My New guess as to what I should be doiong
        mumat = evecsTr.dot(testMu3)  # My New guess as to what I should be doiong - simple
        imat = np.sum(np.square(mumat),axis=1)
        return evals,evecs,imat,fmat

    def getShiftChunk(self,sortedCouplings, sortedOverlap, sortedMus, sortedMyMus, sortedAssign):
        chunkE = sortedCouplings[self.center - self.shift + 1:self.center + self.shift + 1,
                 self.center - self.shift + 1:self.center + self.shift + 1]

        # np.savetxt("FreqsBeingMixed_" + self.pltdata, np.diag(chunkE))

        chunkO = sortedOverlap[self.center - self.shift + 1:self.center + self.shift + 1,
                 self.center - self.shift + 1:self.center + self.shift + 1]
        chunkMu = sortedMus[self.center - self.shift + 1:self.center + self.shift + 1,
                  self.center - self.shift + 1:self.center + self.shift + 1]
        chunkMyMu = sortedMyMus[self.center - self.shift + 1:self.center + self.shift + 1]
        chunkAssign = sortedAssign[self.center - self.shift + 1:self.center + self.shift + 1]
        return chunkE,chunkO,chunkMu,chunkMyMu,chunkAssign


    def getEChunk(self,sortedCouplings, sortedOverlap, sortedMus, sortedMyMus, sortedAssign):
        energy = sortedCouplings[self.center,self.center]
        qual = (np.diag(sortedCouplings) < (energy+self.eShift)) * (np.diag(sortedCouplings) > (energy-self.eShift))
        self.mx = np.where(qual)[0][-1]+1 #for indexing stuff
        self.mn = np.where(qual)[0][0]
        self.mnE=sortedCouplings[self.mn,self.mn]
        self.mxE = sortedCouplings[self.mx, self.mx]
        chunkE = sortedCouplings[self.mn:self.mx,self.mn:self.mx]
        chunkO = sortedOverlap[self.mn:self.mx,self.mn:self.mx]
        chunkMu = sortedMus[self.mn:self.mx,self.mn:self.mx]
        chunkMyMu = sortedMyMus[self.mn:self.mx]
        chunkAssign = sortedAssign[self.mn:self.mx]
        return chunkE,chunkO,chunkMu,chunkMyMu,chunkAssign


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
    def includeCouplings(self, freqFunds, freqOvsC, assignments):
        freqOvers=freqOvsC[-self.nvibs:]
        freqCombos= freqOvsC[:-self.nvibs]

        print self.pltdata
        assignments=np.column_stack((np.array([-1,-1]),assignments.T)).T
        overlap = np.loadtxt(
            'redH/overlapMatrix2_' + self.pltdata + '.dat')
        couplings = np.loadtxt(
            'redH/offDiagonalCouplingsInPotential2_' + self.pltdata+".dat")
        #########TESTING############
        # overlap, couplings = self.setOffDiags(6 + 1, overlap, couplings,killall=True)
        # overlap, couplings = self.setOffDiags(6+1, overlap, couplings)
        # overlap, couplings = self.setOffDiags(7+1, overlap, couplings)
        # overlap,couplings=self.setOffDiags(8+1,overlap,couplings)
        ############################
        fundMu = np.loadtxt('redH/1mu0'+ self.pltdata)
        overMu = np.loadtxt('redH/2mu0'+ self.pltdata)
        comboMu = np.loadtxt('redH/11mu0p'+ self.pltdata)
        print fundMu.shape,overMu.shape,comboMu.shape
        diagonalFreq = np.concatenate((np.array([0]),freqFunds, freqOvers,freqCombos))
        np.fill_diagonal(couplings, diagonalFreq) #lay excitations on diagonal
        mus = np.zeros((len(fundMu)+len(overMu)+len(comboMu)+1,len(fundMu)+len(overMu)+len(comboMu)+1,3))
        myMus = np.vstack((np.array([0, 0, 0]),fundMu, overMu,comboMu))
        for p in range(mus.shape[0]):
            mus[p,p,:] = myMus[p]
        idx = np.argsort(np.diag(couplings))
        sortedAssign = assignments[idx]
        sortedOverlap = overlap[idx,:][:,idx]
        sortedMus = mus[idx,:,:][:,idx,:]
        sortedCouplings = couplings[idx, :][:, idx]
        sortedMyMus = myMus[idx]
        if 'tet' in self.pltdata: #tetramer
            if 'all' in self.pltdata:
                self.center=np.argwhere(sortedCouplings==freqFunds[7])[0,0]
            elif '1He' in self.pltdata or '1De' in self.pltdata:  # tetramer
                #Mix about 2800 - 350 mix
                # self.center = np.argwhere(sortedCouplings == freqFunds[6])[0, 0]
                #2759.159
                self.center = np.argwhere(sortedCouplings == freqFunds[0])[0, 0]
            else:
                #Mix about 2600 - 300/350 mix
                # self.center = np.argwhere(sortedCouplings == freqFunds[6])[0, 0]
                #2658
                self.center = np.argwhere(sortedCouplings == freqFunds[5])[0, 0]
        else:
            if 'allH' in self.pltdata or 'allD' in self.pltdata:
                # self.center = np.argwhere(sortedCouplings == freqCombos[134])[0, 0] #2069.3734
                self.center = np.argwhere(sortedCouplings == freqFunds[5])[0, 0]
                # if 'Then' in self.pltdata:
                #     self.center = np.argwhere(sortedCouplings == freqFunds[8])[0, 0]
            elif '1Hh' in self.pltdata or '1Dh' in self.pltdata:
                #around 1700
                # self.center = np.argwhere(sortedCouplings == freqFunds[5])[0, 0]
                self.center = np.argwhere(sortedCouplings == freqFunds[6])[0, 0]
            elif '1He' in self.pltdata:
                #1855.1322272585126   0.00010951344816488748       6 17
                # 6 is the antisymmetric shared proton OD Stretch
                self.center = np.argwhere(sortedCouplings == freqCombos[133])[0, 0]
                self.center = np.argwhere(sortedCouplings == freqFunds[6])[0, 0]

            else:
                self.center = np.argwhere(sortedCouplings == freqFunds[6])[0, 0]

        self.centerE = sortedCouplings[self.center,self.center]
        chunkE,chunkO,chunkMu,chunkMyMu,chunkAssign = self.getEChunk(sortedCouplings,sortedOverlap,sortedMus,sortedMyMus,sortedAssign)
        self.plotOverlapAndCouplings(chunkO, chunkE)
        self.getLargeCouplings(chunkE,chunkAssign,100)
        # newChunkE=self.MartinCouplings(chunkE)
        evals,evecs,imat,fmat = self.DiagonalizeHamiltonian(chunkE,chunkO,chunkMu)
        self.participationRatio(evals,evecs)
        # if 'tet' in self.pltdata and  ('1Hw' in self.pltdata or '1Dw' in self.pltdata):
        #         print('skipping contribs')
        # else:
        self.getContributionsAndPlot(chunkE, freqFunds, np.concatenate((freqCombos,freqOvers)),imat, evals, evecs, chunkAssign)
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

    def gauss(self, E, freq, inten,less):
        # inten*=2
        #plt.stem(freq,inten)
        #plt.savefig('prelim')
        if 'allH' in self.pltdata:
            broad=25
        elif 'allD' or 'final' in self.pltdata:
            broad = 20
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
            # (1/(broad*np.sqrt(2*np.pi)))*
            g += inten[i] * np.exp(-np.square(E - freq[i]) / (2.0*np.square(broad)))
        if np.amax(g) > 1:
            g/=np.amax(g)
        plt.rc('text', usetex=True)
        plt.rcParams.update({'font.size': 16})
        plt.plot(E, g, color='r',label='Ground State Approx.')

        if self.mix:
            print 'filling'
            # plt.fill_between(E, g, color='r', alpha=0.5,label='GSA')
        # else:
        #     plt.fill_between(E, g, color='b', alpha=0.5, label='GSA')

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
                # plt.stem(np.append(freq[:(self.center - self.shift-less)], freq[(self.center + self.shift-less):]),np.append(inten[:(self.center - self.shift-less)], inten[(self.center + self.shift-less):]), 'r', markerfmt=" ", basefmt=" ",label='_nolegend_')
                # plt.stem(freq[self.center - self.shift-less:self.center + self.shift-less], inten[self.center - self.shift-less:self.center + self.shift-less], 'g', markerfmt=" ", basefmt=" ",label='_nolegend_')
                if self.mn == 1 or self.mn < less:
                    print 'shabadadadababdadba'
                    plt.stem(freq,inten,'b', markerfmt=" ", basefmt=" ", label='_nolegend_')
                else:
                    plt.stem(np.append(freq[:(self.mn - less)], freq[(self.mx - less):]),
                             np.append(inten[:(self.mn - less)], inten[(self.mx - less):]),
                             'b', markerfmt=" ", basefmt=" ", label='_nolegend_')
                    plt.stem(freq[self.mn-less:self.mx-less],
                             inten[self.mn-less:self.mx-less], 'k', markerfmt=" ",
                             basefmt=" ", label='_nolegend_')
                    # plt.scatter([freq[self.center-less]], [1.0],c='g')
            plt.plot([self.mnE,self.mxE],[1.1,1.1])
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
                plt.fill(E, ch/np.amax(ch), color='b', alpha=0.5)

                plt.legend()
                # plt.stem(cfreq,cInt, 'g', markerfmt=" ", basefmt=" ")
                plt.savefig('SpectraPix/' + self.pltdata + '_' + self.pltdata+'_MIX_stem_chinh_newnewnew.png',dpi=500)
            else:
                plt.savefig('SpectraPix/' + self.pltdata + '_' + self.pltdata +'_MIX_shift'+str(self.eShift)+'.png',dpi=500)
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
                plt.fill(E, ch/np.amax(ch), color='b', alpha=0.5)

                plt.legend()
                # plt.stem(cfreq,cInt, 'g', markerfmt=" ", basefmt=" ")
                plt.savefig('SpectraPix/' + self.cfg+ '_' + self.pltdata+'_NoMIX_stem_chinh.png',dpi=500)
            else:
                plt.savefig('SpectraPix/' + self.cfg + '_' + self.pltdata+'.png',dpi=500)
        plt.close()

    def gauss_OC(self, E, freq, inten,less):
        g = np.zeros(len(E))
        for i in range(len(freq)):
            g += inten[i] * np.exp(-np.square(E - freq[i]) / (np.square(10)))
        if np.amax(g) > 1:
            g/=np.amax(g)

        #plt.ylim([0,.2])
        plt.ylim([0,.006])
        #plt.xlim([3400,4600])
        plt.xlim([4800,8000])
        # plt.xticks(np.arange(3400,4600, 400))
        plt.xticks(np.arange(4800,8000, 400))

            #plt.title(self.pltdata)
        plt.ylabel('Rel. Intensity')
        plt.xlabel('$cm^{-1} $')
        plt.gca().tick_params(axis='x',pad=10)
        plt.plot(E, g, color='r',label='Ground State Approx.',linewidth=2)
        #plt.fill(E, g, color='r', alpha=0.5)
        #plt.stem(np.append(freq[:(self.center - self.shift-less)], freq[(self.center + self.shift-less):]),np.append(inten[:(self.center - self.shift-less)], inten[(self.center + self.shift-less):]), 'b', markerfmt=" ", basefmt=" ",label='_nolegend_')
        #plt.stem(freq[self.center - self.shift-less:self.center + self.shift-less], inten[self.center - self.shift-less:self.center + self.shift-less], 'r', markerfmt=" ", basefmt=" ",label='_nolegend_')
        plt.rcParams.update({'font.size':16})
        if self.mix:
            if self.stix:
                plt.stem(np.append(freq[:(self.center - self.shift-less)], freq[(self.center + self.shift-less):]),np.append(inten[:(self.center - self.shift-less)], inten[(self.center + self.shift-less):]), 'b', markerfmt=" ", basefmt=" ",label='_nolegend_')
                plt.stem(freq[self.center - self.shift-less:self.center + self.shift-less], inten[self.center - self.shift-less:self.center + self.shift-less], 'g', markerfmt=" ", basefmt=" ",label='_nolegend_')
            if self.pltVCI:
                if 'allH' in self.pltdata:
                    cf = 'allH'
                else:
                    cf = 'allD'
                ctrim = np.loadtxt("ChinhTrimer/Chinh_freqI_Trimer_" + cf)
                cfreq = ctrim[:, 0]
                cInt = ctrim[:, 1] / np.amax(ctrim[:, 1])
                ch = np.zeros(len(E))
                for c in range(len(cfreq)):
                    ch += cInt[c] * np.exp(-np.square(E - cfreq[c]) / (np.square(10)))
                plt.plot(E, ch / np.amax(ch), 'k', label='VCI/VSCF')
                plt.fill(E, ch/np.amax(ch), color='b', alpha=0.5)

                plt.legend()
                # plt.stem(cfreq,cInt, 'g', markerfmt=" ", basefmt=" ")
                plt.savefig('SpectraPix/SmoothedNew/HighFreq/hf__' + self.pltdata + '_' + self.pltdata+'_MIX_stem_chinh_newnewnew.png',dpi=500)
            else:
                if self.stix:
                    plt.stem(freq,inten, 'g', markerfmt=" ", basefmt=" ",label='_nolegend_')
                    plt.savefig('SpectraPix/SmoothedNew/HighFreq/hf__' + self.pltdata + '_' + self.pltdata +'_MIX_.png',dpi=500)
        else:
            plt.savefig('SpectraPix/SmoothedNew/HighFreq/hf__' + self.pltdata + '_' + self.pltdata+'.png',dpi=500)
        plt.close()

    def writeAssignSorts(self,asn,fre,inte):
        fl = open("assignmentsPreMix/assignSorted"+self.cfg+self.pltdata,"w+")
        for k in reversed(range(len(asn))):
            fl.write("%-4d %-4d %-7.7f %-7.7f\n" % (asn[k,0],asn[k,1],fre[k],inte[k]))
        fl.close()

    def plotSpec(self):
        interest = np.loadtxt(FundamentalShell + self.pltdata)
        interestfreqs = interest[:, 1]
        interestInts = interest[:, 2]
        interestAssign = np.column_stack((interest[:,0],np.array(np.repeat(999,len(interest[:,0])))))
        interest2 = np.loadtxt(ComboShell + self.pltdata + ".data2")
        interestfreqs2 = interest2[:, 0]
        interestInts2 = interest2[:, 1] #combinations then overtones
        interestAssign2 = interest2[:, 2:]
        np.set_printoptions(threshold=np.nan)
        if 'allD' in self.pltdata:
            print 'exception activated'
            normParam=800
            E = np.linspace(800, 3000,4400)
        else:
            normParam=800
            E = np.linspace(800, 4000, 6400)
        if 'tet' in self.pltdata and ('1H' in self.pltdata or '1D' in self.pltdata):
            normParam = 2100
            E = np.linspace(2200, 4000, 3600)
        totalInts = np.concatenate((interestInts, interestInts2))
        totalFreqs = np.concatenate((interestfreqs, interestfreqs2))
        totalAssign = np.concatenate((interestAssign,interestAssign2))
        sortInd = totalInts.argsort()
        sortedTotalFreqs = totalFreqs[sortInd]
        sortedTotalInts = totalInts[sortInd]
        sortedTotalAssign = totalAssign[sortInd]
        if self.mix:
            newFreqs, newInts = self.includeCouplings(interestfreqs, interestfreqs2,totalAssign)
            totalInts = np.concatenate((interestInts, interestInts2))
            totalFreqs = np.concatenate((interestfreqs, interestfreqs2))
            sortInd = totalFreqs.argsort()
            sortedTotalFreqs = totalFreqs[sortInd]
            sortedTotalInts = totalInts[sortInd]
            sortedTotalFreqs[self.mn:self.mx] = newFreqs
            sortedTotalInts[self.mn:self.mx] = newInts #no longer sorted
            interestfreqsp, interestIntsp, lose = self.normAll(sortedTotalFreqs, sortedTotalInts, normParam)
        else:
            interestfreqsp, interestIntsp, lose, sortedTotalAssignp = self.normAll(sortedTotalFreqs, sortedTotalInts,normParam, sortedTotalAssign)
            self.writeAssignSorts(sortedTotalAssignp,interestfreqsp,interestIntsp)
        self.gauss(E, interestfreqsp, interestIntsp,lose)
        if self.OC:
            self.gauss_OC(E, interestfreqsp, interestIntsp,lose)


"""def __init__(self, cfg, pltdata, pltVCI=True,stix=False):"""

mp = myPlot('TrimerNewDefsEck','final_allHrnspc_finalVersion_SPC_BasedOnOs',mix=False,stix=True,pltVCI=False)
mp.plotSpec()
#
mp = myPlot('TrimerNewDefsEck','final_allHrnspc_finalVersion_SPC_BasedOnOs',mix=True,stix=True,pltVCI=False)
mp.plotSpec()

mp = myPlot('TrimerNewDefsEck','final_allDrnspc_finalVersion_SPC_BasedOnOs',mix=False,stix=True,pltVCI=False)
mp.plotSpec()

mp = myPlot('TrimerNewDefsEck','final_allDrnspc_finalVersion_SPC_BasedOnOs',mix=True,stix=True,pltVCI=False)
mp.plotSpec()
#
# mp = myPlot('TrimerNewDefsEck','final__1Hwrnspc_finalVersion_SPC_BasedOnOs',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('TrimerNewDefsEck','final__1Hwrnspc_finalVersion_SPC_BasedOnOs',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('TrimerNewDefsEck','final__1Dwrnspc_finalVersion_SPC_BasedOnOs',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('TrimerNewDefsEck','final__1Dwrnspc_finalVersion_SPC_BasedOnOs',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()
# #
# mp = myPlot('TrimerNewDefsEck','final__1Hernspc_finalVersion_SPC_BasedOnOs',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
# #
# mp = myPlot('TrimerNewDefsEck','final__1Hernspc_finalVersion_SPC_BasedOnOs',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()
#
# # mp = myPlot('TrimerNewDefsEck','final__1Dernspc_finalVersion_SPC_BasedOnOs',mix=False,stix=True,pltVCI=False)
# # mp.plotSpec()
# #
# # mp = myPlot('TrimerNewDefsEck','final__1Dernspc_finalVersion_SPC_BasedOnOs',mix=True,stix=True,pltVCI=False)
# # mp.plotSpec()
#
# mp = myPlot('TrimerNewDefsEck','final__1Hhrnspc_finalVersion_SPC_BasedOnOs_fix',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('TrimerNewDefsEck','final__1Hhrnspc_finalVersion_SPC_BasedOnOs_fix',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()

# mp = myPlot('TrimerNewDefsEck','final__1Dhrnspc_finalVersion_SPC_BasedOnOs',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('TrimerNewDefsEck','final__1Dhrnspc_finalVersion_SPC_BasedOnOs',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()

#
# mp = myPlot('TetramerNewDefsEck','fSymtet_allHrnspc_justOThenCartOEck',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('TetramerNewDefsEck','fSymtet_allHrnspc_justOThenCartOEck',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()
#
# #
# mp = myPlot('TetramerNewDefsEck','fSymtet_allDrnspc_justOThenCartOEck',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
# #
# mp = myPlot('TetramerNewDefsEck','fSymtet_allDrnspc_justOThenCartOEck',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('TetramerNewDefsEck','fSymtet_1Hernspc_finalVersion',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
# #
# mp = myPlot('TetramerNewDefsEck','fSymtet_1Hernspc_finalVersion',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()
# #
# # mp = myPlot('TetramerNewDefsEck','fSymtet_1Dernspc_finalVersion',mix=False,stix=True,pltVCI=False)
# # mp.plotSpec()
# #
# # mp = myPlot('TetramerNewDefsEck','fSymtet_1Dernspc_finalVersion',mix=True,stix=True,pltVCI=False)
# # mp.plotSpec()
# #
# mp = myPlot('TetramerNewDefsEck','fSymtet_1Hwrnspc_finalVersion',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
# #
# mp = myPlot('TetramerNewDefsEck','fSymtet_1Hwrnspc_finalVersion',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()

# mp = myPlot('TetramerNewDefsEck','fSymtet_1Dwrnspc_finalVersion',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('TetramerNewDefsEck','fSymtet_1Dwrnspc_finalVersion',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()












# #Ryans Wfn
# mp = myPlot('TetramerNewDefsEck','iSymtet_allHrnspc_finalVersion',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('TetramerNewDefsEck','iSymtet_allHrnspc_finalVersion',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()
