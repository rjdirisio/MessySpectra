import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.linalg as sla
import itertools
FundamentalShell = "Fundamentals_"
exptlPeaks = "redH//allHTesting/spectraTesting/ExperimentalPeaksJCPL"
VCIPeaks = "redH//allHTesting/spectraTesting/VCIPeaksJCPL"
ComboShell = "combinationOverrtone_"


class myPlot:
    def __init__(self, cfg, pltdata,mix=True, pltVCI=False,stix=False,anneRange=False,OC=False):
        self.cfg = cfg
        print self.cfg
        self.pltdata = pltdata  # intending to be a file with the format I like, with freq       intensity
        self.mix = mix
        self.pltVCI = pltVCI  # true or false
        self.stix = stix
        self.OC=OC

        if 'tet' in self.pltdata:
            self.nvibs = 33
            if '1He'in self.pltdata:
                self.shift = 20
            else:
                self.shift = 40
            self.eShift = 300
        elif 'final' in self.pltdata:
            self.nvibs = 24
            self.shift = 25
            self.eShift = 150

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
            np.savetxt("bigCoupleAssignments" + self.pltdata, bigCouplings[bigCouplings[:,0].argsort()], fmt='%i')



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
        plt.savefig('sortedOverlaps_' + self.pltdata)
        plt.close()
        plt.matshow(chunkE)
        plt.colorbar()
        plt.savefig('sortedCouplings_' + self.pltdata)
        plt.close()

    def getContributionsAndPlot(self,chunkE,freqFunds,imat,evals,evecs):
        contrib = open('contributionOfOH_'+self.pltdata,"w+")
        if 'tet' in self.pltdata:
            print np.diag(chunkE)
            someInd = np.argwhere(chunkE == freqFunds[7])[0][0]
            otherInd = np.argwhere(chunkE == freqFunds[8])[0][0]
            thirdInd = np.argwhere(chunkE==freqFunds[6])[0][0]
        elif 'final' in self.pltdata:
            print 'hihi'
            someInd = np.argwhere(chunkE == freqFunds[6])[0][0]
            otherInd = np.argwhere(chunkE == freqFunds[6])[0][0]
            # otherInd = np.argwhere(chunkE == freqFunds[5])[0][0]



        if 'Trimer' in self.cfg:
            contrib.write('E     I      OH-1   OH-2\n')
            # someInd = 5
            # otherInd = someInd+1 #other OH stretch fundamental
            for nevs in range(len(evals)):
                print 'hi'
                contrib.write('%5.1f %5.4f %5.3f %5.3f\n' % (
                evals[nevs], imat[nevs] / np.amax(imat), np.square(evecs[someInd, nevs]),
                np.square(evecs[otherInd, nevs])))

            sumsquares = np.zeros(2)
            for nevs in range(len(evals)):
                # print 'hi'
                evtest = np.sum(np.square(evecs[:, nevs]))
                sumsquares[0] += np.square(evecs[someInd, nevs])
                sumsquares[1] += np.square(evecs[otherInd, nevs])
                if nevs == 0:
                    plt.scatter(evals[nevs], np.square(evecs[someInd, nevs]), c='r', label="Totally Symmetric Stretch")
                    plt.scatter(evals[nevs], np.square(evecs[otherInd, nevs]), c='b', label="Antisymmetric Stretch 1")
                else:
                    plt.scatter(evals[nevs], np.square(evecs[someInd, nevs]), c='r')
                    plt.scatter(evals[nevs], np.square(evecs[otherInd, nevs]), c='b')
        else:
            contrib.write('E     I      OH-1   OH-2  OH-3\n')
            # someInd = 6
            # otherInd = someInd + 1  # other OH stretch fundamental
            # thirdInd = otherInd + 1  # other OH stretch fundamental
            for nevs in range(len(evals)):
                contrib.write('%5.1f %5.4f %5.3f %5.3f %5.3f\n' % (
                evals[nevs], imat[nevs] / np.amax(imat), np.square(evecs[someInd, nevs]), np.square(evecs[otherInd, nevs]),np.square(evecs[thirdInd, nevs])))
            sumsquares = np.zeros(3)
            for nevs in range(len(evals)):
                print 'hi'
                # evtest = np.sum(np.square(evecs[:,nevs]))
                sumsquares[0] += np.square(evecs[someInd,nevs])
                sumsquares[1] +=np.square(evecs[otherInd,nevs])
                sumsquares[2]+=np.square(evecs[thirdInd,nevs])
                if nevs == 0:
                    plt.scatter(evals[nevs],np.square(evecs[someInd,nevs]),c='r',label="Totally Symmetric Stretch")
                    plt.scatter(evals[nevs], np.square(evecs[otherInd,nevs]),c='b',label="Antisymmetric Stretch 1")
                    plt.scatter(evals[nevs], np.square(evecs[thirdInd,nevs]),c='g',label="Antisymmetric Stretch 2")
                else:
                    plt.scatter(evals[nevs], np.square(evecs[someInd, nevs]), c='r')
                    plt.scatter(evals[nevs], np.square(evecs[otherInd, nevs]), c='b')
                    plt.scatter(evals[nevs], np.square(evecs[thirdInd, nevs]), c='g')
        plt.legend(scatterpoints=1,fontsize=8)
        plt.xlim([800,4000])
        plt.ylim([0,0.5])
        plt.savefig("ScatterCoefPlots/sharedProtonContributions"+self.pltdata)
        plt.close()


    def DiagonalizeHamiltonian(self,chunkE,chunkO,chunkMu):
        print 'original Freqs\n', np.diagonal(chunkE)
        ev,Q = la.eigh(chunkO)
        E12 = np.diag(1/np.sqrt(ev))
        #S12 = np.matmul(Q.T, np.matmul(E12, Q))  # !?!?!??!?!
        S12 = np.matmul(Q,np.matmul(E12,Q.T)) #!?!?!??!?!
        fmat = np.dot(S12, np.dot(chunkE, S12))
        evals, evecs = la.eigh(fmat)
        #np.savetxt("evecsT_mixed",evecs.T)
        evals1,evecs1 = sla.eigh(chunkE,chunkO)
        np.savetxt("evecsT_mixed"+self.pltdata,evecs1.T)
        test=evecs1.T.dot(chunkO.T)
        testMu3 = np.diagonal(chunkMu).T
        # testMu4 = chunkMyMu
        # print chunkMyMu
        mumat = np.dot(evecs, np.dot(S12, testMu3))
        # print np.around(mumat,6) == np.around(testMu4,6)
        imat2 = np.square(la.norm(mumat,axis=1))
        imat = np.sum(np.square(mumat),axis=1)
        return evals,evecs,imat

    def getShiftChunk(self,sortedCouplings, sortedOverlap, sortedMus, sortedMyMus, sortedAssign):
        chunkE = sortedCouplings[self.center - self.shift + 1:self.center + self.shift + 1,
                 self.center - self.shift + 1:self.center + self.shift + 1]

        # print 'someind', someInd #get where SPS fundamental is in reduced H
        # print 'other', otherInd #get other SPS fundamental is in reduced H
        np.savetxt("FreqsBeingMixed_" + self.pltdata, np.diag(chunkE))

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
        mx = np.where(qual)[0][-1]
        mn = np.where(qual)[0][0]
        chunkE = sortedCouplings[mn:mx,mn:mx]
        chunkO = sortedOverlap[mn:mx,mn:mx]
        chunkMu = sortedMus[mn:mx,mn:mx]
        chunkMyMu = sortedMyMus[mn:mx]
        chunkAssign = sortedAssign[mn:mx]
        return chunkE,chunkO,chunkMu,chunkMyMu,chunkAssign

    def includeCouplings(self, freqFunds, freqOvsC,intFunds,ovInts, assignments):
        freqOvers=freqOvsC[-self.nvibs:]
        freqCombos= freqOvsC[:-self.nvibs]
        intOvers = ovInts[-self.nvibs:]
        intCombos= ovInts[:-self.nvibs]

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
        # ovp = np.copy(overlap[6:9, 6:25])
        # ovp[(np.around(ovp,7)==1.0)] = 0.0
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
        # sortedTints = testIntensities[idx, :][:, idx]
        sortedMyMus = myMus[idx]
        if 'tet' in self.pltdata: #tetramer
            self.center=np.argwhere(sortedCouplings==freqFunds[7])[0,0]
            if '1He' in self.pltdata:  # tetramer
                self.center = np.argwhere(sortedCouplings == freqFunds[6])[0, 0]

        else:
            self.center=np.argwhere(sortedCouplings==freqFunds[6])[0,0]



        chunkE,chunkO,chunkMu,chunkMyMu,chunkAssign = self.getEChunk(sortedCouplings,sortedOverlap,sortedMus,sortedMyMus,sortedAssign)
        chunkE,chunkO,chunkMu,chunkMyMu,chunkAssign = self.getShiftChunk(sortedCouplings, sortedOverlap, sortedMus, sortedMyMus, sortedAssign)

        # chunkE = sortedCouplings[self.center-self.shift+1:self.center+self.shift+1,self.center-self.shift+1:self.center+self.shift+1]
        # np.savetxt("FreqsBeingMixed_"+self.pltdata,np.diag(chunkE))
        # chunkO = sortedOverlap[self.center-self.shift+1:self.center+self.shift+1,self.center-self.shift+1:self.center+self.shift+1]
        # chunkMu = sortedMus[self.center-self.shift+1:self.center+self.shift+1,self.center-self.shift+1:self.center+self.shift+1]
        # chunkMyMu = sortedMyMus[self.center-self.shift+1:self.center+self.shift+1]
        # chunkAssign = sortedAssign[self.center-self.shift+1:self.center+self.shift+1]

        self.plotOverlapAndCouplings(chunkO, chunkE)
        self.getLargeCouplings(chunkE,chunkAssign,300)
        evals,evecs,imat = self.DiagonalizeHamiltonian(chunkE,chunkO,chunkMu)
        # plt.matshow(np.square(evecs.T))
        # plt.colorbar()
        # plt.show()
        self.getContributionsAndPlot(chunkE, freqFunds, imat, evals, evecs)

        # print 'sumsquares',sumsquares
        print 'hi'
        #contrib.write('%5.9f' % np.sum(np.square(evecs[:, someInd])))
        #contrib.write('%5.9f' % np.sum(np.square(evecs[:, otherInd])))
        #contrib.close()
        # print np.c_[evals,imat]
        return evals, imat


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
            g += inten[i] * np.exp(-np.square(E - freq[i]) / (np.square(broad)))
        if np.amax(g) > 1:
            g/=np.amax(g)
        plt.rc('text', usetex=True)
        plt.rcParams.update({'font.size': 16})
        plt.plot(E, g, color='k',label='Ground State Approx.')
        if self.mix:
            plt.fill_between(E, g, color='r', alpha=0.5,label='GSA')
        else:
            plt.fill_between(E, g, color='b', alpha=0.5, label='GSA')
        #plt.stem(freq,np.ones(len(freq)),linefmt='grey',markerfmt='')
        #plt.show()
        #plt.stem(np.append(freq[:(self.center - self.shift-less)], freq[(self.center + self.shift-less):]),np.append(inten[:(self.center - self.shift-less)], inten[(self.center + self.shift-less):]), 'b', markerfmt=" ", basefmt=" ",label='_nolegend_')
        #plt.stem(freq[self.center - self.shift-less:self.center + self.shift-less], inten[self.center - self.shift-less:self.center + self.shift-less], 'r', markerfmt=" ", basefmt=" ",label='_nolegend_')
        #plt.xlim([800,4600])
        plt.xlim([np.amin(E), np.amax(E)])
        # if 'allH' in self.pltdata:
        #     print 'asdf'
        # elif 'allD' in self.pltdata:
        #     plt.xlim([1000,3000])
        plt.ylim([0, 2])
        # if 'allD' in self.pltdata:
        #     plt.xticks(np.arange(600, 4000, 400))
        # elif '1He' in self.pltdata:
        #     plt.xticks(np.arange(1000, 4600, 800))
        # else:
        #     plt.xticks(np.arange(800, 4600, 400))
        #     #plt.title(self.pltdata)
        plt.ylabel(r'Rel. Intensity')
        plt.xlabel(r'Energy (cm$^{-1}$)')
        plt.gca().tick_params(axis='x',pad=10)
        if self.mix:
            if self.stix:
                plt.stem(np.append(freq[:(self.center - self.shift-less)], freq[(self.center + self.shift-less):]),np.append(inten[:(self.center - self.shift-less)], inten[(self.center + self.shift-less):]), 'r', markerfmt=" ", basefmt=" ",label='_nolegend_')
                plt.stem(freq[self.center - self.shift-less:self.center + self.shift-less], inten[self.center - self.shift-less:self.center + self.shift-less], 'g', markerfmt=" ", basefmt=" ",label='_nolegend_')

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
                plt.savefig('SpectraPix/' + self.pltdata + '_' + self.pltdata+'_MIX_stem_chinh_newnewnew.png')
            else:
                plt.savefig('SpectraPix/' + self.pltdata + '_' + self.pltdata +'_MIX_.png')
        else:
            if self.stix:
                plt.stem(freq,inten, 'g', markerfmt=" ", basefmt=" ",label='_nolegend_')
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
                plt.savefig('SpectraPix/' + self.cfg+ '_' + self.pltdata+'_NoMIX_stem_chinh.png')
            else:
                plt.savefig('SpectraPix/' + self.cfg + '_' + self.pltdata+'.png')
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
                plt.savefig('SpectraPix/SmoothedNew/HighFreq/hf__' + self.pltdata + '_' + self.pltdata+'_MIX_stem_chinh_newnewnew.png')
            else:
                if self.stix:
                    plt.stem(freq,inten, 'g', markerfmt=" ", basefmt=" ",label='_nolegend_')
                    plt.savefig('SpectraPix/SmoothedNew/HighFreq/hf__' + self.pltdata + '_' + self.pltdata +'_MIX_.png')
        else:
            plt.savefig('SpectraPix/SmoothedNew/HighFreq/hf__' + self.pltdata + '_' + self.pltdata+'.png')
        plt.close()

    def writeAssignSorts(self,asn,fre,inte):
        fl = open("assignSorted"+self.cfg+self.pltdata,"w+")
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
        fundMu = np.loadtxt('redH/1mu0' + self.pltdata)
        overMu = np.loadtxt('redH/2mu0' + self.pltdata)
        comboMu = np.loadtxt('redH/11mu0p' + self.pltdata)
        interestIntsT = np.square(la.norm(fundMu,axis=1))
        interestIntsC = np.square(la.norm(comboMu,axis=1))
        interestIntsO = np.square(la.norm(overMu, axis=1))
        # print np.column_stack((np.array(np.concatenate((interestInts,interestInts2))),np.array(np.concatenate((interestIntsT,interestInts2T,interestInts2pT)))))
        # print np.around(np.array(np.concatenate((interestInts,interestInts2))),1)==np.around(np.array(np.concatenate((interestIntsT,interestInts2T,interestInts2pT))),1)
        # print np.array(np.concatenate((interestInts,interestInts2)))-np.array(np.concatenate((interestIntsT,interestInts2T,interestInts2pT)))
        if 'allD' in self.pltdata:
            print 'exception activated'
            normParam=600
            E = np.linspace(600, 3000,5000)
        elif '1Hh' in self.pltdata:
            print 'exception activated'
            normParam=600
            E = np.linspace(600, 3000,6400)
        else:
            normParam=800
            E = np.linspace(800, 4200, 5000)
        # normParam=0.0
        totalInts = np.concatenate((interestInts, interestInts2))
        totalFreqs = np.concatenate((interestfreqs, interestfreqs2))
        totalAssign = np.concatenate((interestAssign,interestAssign2))
        sortInd = totalInts.argsort()
        sortedTotalFreqs = totalFreqs[sortInd]
        sortedTotalInts = totalInts[sortInd]
        sortedTotalAssign = totalAssign[sortInd]
        if self.mix:
            newFreqs, newInts = self.includeCouplings(interestfreqs, interestfreqs2, interestInts, interestInts2,totalAssign)
            totalInts = np.concatenate((interestInts, interestInts2))
            totalFreqs = np.concatenate((interestfreqs, interestfreqs2))
            sortInd = totalFreqs.argsort()
            sortedTotalFreqs = totalFreqs[sortInd]
            testSortI = np.copy(sortedTotalInts)
            testSortF = np.copy(sortedTotalFreqs)
            sortedTotalInts = totalInts[sortInd]
            print np.sum(totalInts)
            print 'wut'
            print np.around(sortedTotalFreqs[self.center-self.shift:self.center+self.shift],8) == np.around(newFreqs,8)
            # print sortedTotalFreqs[self.center-self.shift:self.center+self.shift]==newFreqs
            print np.around(sortedTotalInts[self.center - self.shift:self.center + self.shift], 8) == np.around(
                newInts, 8)

            print np.around(sortedTotalInts[self.center-self.shift:self.center+self.shift],5) == np.around(newInts,5)

            sortedTotalFreqs[self.center - self.shift:self.center + self.shift] = newFreqs
            sortedTotalInts[self.center - self.shift:self.center + self.shift] = newInts
            print np.around(sortedTotalInts, 10) == np.around(testSortI, 10)
            print np.c_[sortedTotalInts[:],testSortI[:]]
            #print np.around(sortedTotalFreqs,5)==np.around(testSortF,5)
            print np.sum(sortedTotalInts)
            print np.sum(testSortI)
            interestfreqsp, interestIntsp, lose = self.normAll(sortedTotalFreqs, sortedTotalInts, normParam)
            print 'afternorm'
            print np.around(interestIntsp, 5) == np.around(testSortI, 5)
            print np.around(interestfreqsp, 5) == np.around(testSortF, 5)

            sortInd = interestfreqsp.argsort()
            sortedinterestfreqsp = interestfreqsp[sortInd[::-1]]
            sortedinterestIntsp = interestIntsp[sortInd[::-1]]
            sortInd2 = interestIntsp.argsort()
            sortedinterestfreqsp2 = interestfreqsp[sortInd2[::-1]]
            sortedinterestIntsp2 = interestIntsp[sortInd2[::-1]]
            print 'sorted freqs and ints - freq sorted'
            print np.c_[sortedinterestfreqsp[:], sortedinterestIntsp[:]]
            print 'sorted freqs and ints - int sorted'
            print np.c_[sortedinterestfreqsp2[:], sortedinterestIntsp2[:]]
        interestfreqsp, interestIntsp, lose,sortedTotalAssignp = self.normAll(sortedTotalFreqs, sortedTotalInts, normParam,sortedTotalAssign)
        if not self.mix:
            self.writeAssignSorts(sortedTotalAssignp,interestfreqsp,interestIntsp)
        self.gauss(E, interestfreqsp, interestIntsp,lose)
        if self.OC:
            self.gauss_OC(E, interestfreqsp, interestIntsp,lose)


"""def __init__(self, cfg, pltdata, pltVCI=True,stix=False):"""

# mp = myPlot('fSymTet_Units','fSymtet_allHregEckTwoAxes',mix=False,stix=True)
# mp.plotSpec()
#
# mp = myPlot('fSymTet_Units','fSymtet_allHregEckTwoAxes',mix=True,stix=True)
# mp.plotSpec()
#

# mp = myPlot('fSymTetAllH_2Ecks','fSymtet_allHregEckEckEck_newOs',mix=False,stix=True)
# mp.plotSpec()
# #
# mp = myPlot('fSymTetAllH_2Ecks','fSymtet_allHregEckEckEck_newOs',mix=True,stix=True)
# mp.plotSpec()


mp = myPlot('TrimerNewDefsEck','final_allHrnspc_justOThenYzEck',mix=False,stix=False,pltVCI=False)
mp.plotSpec()

mp = myPlot('TrimerNewDefsEck','final_allHrnspc_justOThenYzEck',mix=True,stix=False,pltVCI=False)
mp.plotSpec()

# mp = myPlot('TrimerNewDefsEck','final_allDrnspc_justOThenCartOEck',mix=False,stix=True,pltVCI=False)
# mp.plotSpec()
#
# mp = myPlot('TrimerNewDefsEck','final_allDrnspc_justOThenCartOEck',mix=True,stix=True,pltVCI=False)
# mp.plotSpec()

mp = myPlot('TetramerNewDefsEck','fSymtet_allHrnspc_justOThenCartOEck',mix=False,stix=True,pltVCI=False)
mp.plotSpec()

mp = myPlot('TetramerNewDefsEck','fSymtet_allHrnspc_justOThenCartOEck',mix=True,stix=True,pltVCI=False)
mp.plotSpec()

mp = myPlot('TetramerNewDefsEck','fSymtet_allDrnspc_justOThenCartOEck',mix=False,stix=True,pltVCI=False)
mp.plotSpec()

mp = myPlot('TetramerNewDefsEck','fSymtet_allDrnspc_justOThenCartOEck',mix=True,stix=True,pltVCI=False)
mp.plotSpec()

mp = myPlot('TetramerNewDefsEck','fSymtet_1Hernspc_justOThenCartOEck',mix=False,stix=True,pltVCI=False)
mp.plotSpec()

mp = myPlot('TetramerNewDefsEck','fSymtet_1Hernspc_justOThenCartOEck',mix=True,stix=True,pltVCI=False)
mp.plotSpec()