#!/usr/bin/env python
from ROOT import TFile, TTree, TH1F, TH1D, TH1, TCanvas, TChain,TGraphAsymmErrors, TMath, TH2D, TLorentzVector, AddressOf, gROOT, TNamed
import ROOT as ROOT
import os,traceback
import sys, optparse,argparse
from array import array
import math
import numpy as numpy
import pandas
from root_pandas import read_root
from pandas import  DataFrame, concat
from pandas import Series
import time

# snippet from Deepak
from multiprocessing import Process
import multiprocessing as mp

sys.path.append('utils')
sys.path.append('../utils')
from MathUtils import *

print "starting clock"
start = time.clock()

outfilename= 'SkimmedTree.root'
PUPPI = True
CA15  = False

usage = "analyzer for bb+DM (debugging) "
parser = argparse.ArgumentParser(description=usage)
parser.add_argument("-i", "--inputfile",  dest="inputfile")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="out.root")
parser.add_argument("-D", "--outputdir", dest="outputdir")
parser.add_argument("-F", "--farmout", action="store_true",  dest="farmout")
args = parser.parse_args()

if args.farmout==None:
    isfarmout = False
else:
    isfarmout = args.farmout

infilename = args.inputfile
print 'outfile= ', args.outputfile

def arctan(x,y):
    corr=0
    if (x>0 and y>=0) or (x>0 and y<0):
        corr=0
    elif x<0 and y>=0:
        corr=math.pi
    elif x<0 and y<0:
        corr=-math.pi
    if x!=0.:
        return math.atan(y/x)+corr
    else:
        return math.pi/2+corr

def getPT_skim(P4):
    return P4.Pt()

def runbbdm(infile_):
    debug_ = False
    outputfilename = args.outputfile
#    NEntries = 1000
    h_total = TH1F('h_total','h_total',2,0,2)
    h_total_mcweight = TH1F('h_total_mcweight','h_total_mcweight',2,0,2)

    triglist=["HLT_PFMET170_BeamHaloCleaned_v","HLT_PFMET170_HBHE_BeamHaloCleaned_v","HLT_PFMET170_NotCleaned_v","HLT_PFMET170_NoiseCleaned_v","HLT_PFMET170_JetIdCleaned_v","HLT_PFMET170_HBHECleaned_v","HLT_PFMETNoMu90_PFMHTNoMu90_IDTight_v","HLT_PFMETNoMu100_PFMHTNoMu100_IDTight_v","HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_v","HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v","HLT_PFMET110_PFMHT110_IDTight_v","HLT_IsoMu24_v","HLT_IsoTkMu24_v","HLT_IsoMu27_v","HLT_IsoTkMu27_v","HLT_Ele27_WPTight_Gsf_v","HLT_Ele105_CaloIdVT_GsfTrkIdT_v","HLT_Ele115_CaloIdVT_GsfTrkIdT_v","HLT_Ele32_WPTight_Gsf_v","HLT_IsoMu20_v","HLT_Ele27_eta2p1_WPTight_Gsf_v","HLT_Ele27_WPLoose_Gsf_v","HLT_Ele32_eta2p1_WPTight_Gsf_v","HLT_Photon165_HE10_v","HLT_Photon175_v"]


    outfile = TFile(outfilename,'RECREATE')

    outTree = TTree( 'outTree', 'tree branches' )
    # if isfarmout:
    #     samplepath = TNamed('samplepath', str(f_tmp).split('"')[1])
    # else:

    #if len(sys.argv)>2:
    #    NEntries=int(sys.argv[2])
    #    print "WARNING: Running in TEST MODE"
    jetvariables = ['runId','lumiSection','eventId','isData','mcWeight','pu_nTrueInt','pu_nPUVert','hlt_trigName','hlt_trigResult','hlt_filterName','hlt_filterResult','pfMetCorrPt','pfMetCorrPhi','pfMetCorrUnc','nEle','elePx','elePy','elePz','eleEnergy','eleIsPassLoose','eleIsPassTight','eleCharge','nPho','phoPx','phoPy','phoPz','phoEnergy','phoIsPassLoose','phoIsPassTight','nMu','muPx','muPy','muPz','muEnergy','isLooseMuon','isTightMuon','muChHadIso','muNeHadIso','muGamIso','muPUPt','muCharge','HPSTau_n','HPSTau_Px','HPSTau_Py','HPSTau_Pz','HPSTau_Energy','disc_decayModeFinding','disc_byLooseIsolationMVArun2v1DBoldDMwLT2016','nGenPar','genParId','genMomParId','genParSt','genPx','genPy','genPz','genEnergy','THINnJet', 'THINjetPx', 'THINjetPy', 'THINjetPz', 'THINjetEnergy','THINjetPassIDTight','THINjetDeepCSV_b', 'THINjetHadronFlavor', 'THINjetNHadEF', 'THINjetCHadEF', 'THINjetCEmEF', 'THINjetPhoEF', 'THINjetEleEF', 'THINjetMuoEF', 'THINjetCorrUncUp','THINjetNPV']
    filename = infile_
    ieve = 0;icount = 0
    for df in read_root(filename, columns=jetvariables, chunksize=125000):
        samplepath = TNamed('samplepath', str(infile))

        st_runId                  = numpy.zeros(1, dtype=int)
        st_lumiSection            = array( 'L', [ 0 ] )
        st_eventId                = array( 'L', [ 0 ] )
        st_pfMetCorrPt            = array( 'f', [ 0. ] )
        st_pfMetCorrPhi           = array( 'f', [ 0. ] )
        st_pfMetUncJetResUp       = ROOT.std.vector('float')()
        st_pfMetUncJetResDown     = ROOT.std.vector('float')()
        st_pfMetUncJetEnUp        = ROOT.std.vector('float')()
        st_pfMetUncJetEnDown      = ROOT.std.vector('float')()
        st_isData           = array( 'b', [ 0 ] )

        for trigs in triglist:
            exec("st_"+trigs+"  = array( 'b', [ 0 ] )")

        maxn = 10

        st_THINnJet                     = array( 'L', [ 0 ] ) #ROOT.std.vector('int')()
        st_THINjetPx                    = ROOT.std.vector('float')()
        st_THINjetPy                    = ROOT.std.vector('float')()
        st_THINjetPz                    = ROOT.std.vector('float')()
        st_THINjetEnergy                = ROOT.std.vector('float')()
        st_THINjetDeepCSV               = ROOT.std.vector('float')()
        st_THINjetHadronFlavor          = ROOT.std.vector('int')()
        st_THINjetNHadEF                = ROOT.std.vector('float')()
        st_THINjetCHadEF                = ROOT.std.vector('float')()

        st_THINjetCEmEF                 = ROOT.std.vector('float')()
        st_THINjetPhoEF                 = ROOT.std.vector('float')()
        st_THINjetEleEF                 = ROOT.std.vector('float')()
        st_THINjetMuoEF                 = ROOT.std.vector('float')()
        st_THINjetCorrUnc               = ROOT.std.vector('float')()


        st_nEle                = array( 'L', [ 0 ] ) #ROOT.std.vector('int')()
        st_elePx               = ROOT.std.vector('float')()
        st_elePy               = ROOT.std.vector('float')()
        st_elePz               = ROOT.std.vector('float')()
        st_eleEnergy           = ROOT.std.vector('float')()
        st_eleIsPassLoose      = ROOT.std.vector('bool')()
        st_eleIsPassTight      = ROOT.std.vector('bool')()

        st_nPho                = array( 'L', [ 0 ] ) #ROOT.std.vector('int')()
        st_phoPx               = ROOT.std.vector('float')()
        st_phoPy               = ROOT.std.vector('float')()
        st_phoPz               = ROOT.std.vector('float')()
        st_phoEnergy           = ROOT.std.vector('float')()
        st_phoIsPassTight      = ROOT.std.vector('bool')()

        st_nMu= array( 'L', [ 0 ] ) #ROOT.std.vector('int')()
        st_muPx                = ROOT.std.vector('float')()
        st_muPy                = ROOT.std.vector('float')()
        st_muPz                = ROOT.std.vector('float')()
        st_muEnergy            = ROOT.std.vector('float')()
        st_isTightMuon         = ROOT.std.vector('bool')()
        st_muIso               = ROOT.std.vector('float')()

        st_HPSTau_n= array( 'L', [ 0 ] ) #ROOT.std.vector('int')()
        st_nTauTightElectron= array( 'L', [ 0 ] )
        st_nTauTightMuon= array( 'L', [ 0 ] )
        st_nTauTightEleMu= array( 'L', [ 0 ] )
        st_nTauLooseEleMu= array( 'L', [ 0 ] )

        mcweight = array( 'f', [ 0 ] )
        st_pu_nTrueInt= array( 'f', [ 0 ] ) #ROOT.std.vector('std::vector<float>')()
        st_pu_nPUVert= array( 'f', [ 0 ] )
        st_THINjetNPV= array( 'f', [ 0 ] ) #ROOT.std.vector('std::vector<float>')()

        st_nGenPar = array( 'L', [ 0 ] )
        st_genParId = ROOT.std.vector('int')()
        st_genMomParId = ROOT.std.vector('int')()
        st_genParSt = ROOT.std.vector('int')()
        st_genParPx = ROOT.std.vector('float')()
        st_genParPy = ROOT.std.vector('float')()
        st_genParPz = ROOT.std.vector('float')()
        st_genParEnergy = ROOT.std.vector('float')()

        WenuRecoil = array( 'f', [ 0. ] )
        Wenumass = array( 'f', [ 0. ] )
        WenuPhi = array( 'f', [ 0. ] )

        WmunuRecoil = array( 'f', [ 0. ] )
        Wmunumass = array( 'f', [ 0. ] )
        WmunuPhi = array( 'f', [ 0. ] )

        ZeeRecoil = array( 'f', [ 0. ] )
        ZeeMass = array( 'f', [ 0. ] )
        ZeePhi = array( 'f', [ 0. ] )

        ZmumuRecoil = array( 'f', [ 0. ] )
        ZmumuMass = array( 'f', [ 0. ] )
        ZmumuPhi = array( 'f', [ 0. ] )

        GammaRecoil = array('f',[0.])
        GammaPhi = array( 'f', [ 0. ] )

        outTree.Branch( 'st_runId', st_runId , 'st_runId/L')
        outTree.Branch( 'st_lumiSection', st_lumiSection , 'st_lumiSection/L')
        outTree.Branch( 'st_eventId',  st_eventId, 'st_eventId/L')
        outTree.Branch( 'st_pfMetCorrPt', st_pfMetCorrPt , 'st_pfMetCorrPt/F')
        outTree.Branch( 'st_pfMetCorrPhi', st_pfMetCorrPhi , 'st_pfMetCorrPhi/F')
        outTree.Branch( 'st_pfMetUncJetResUp', st_pfMetUncJetResUp)
        outTree.Branch( 'st_pfMetUncJetResDown', st_pfMetUncJetResDown)
        outTree.Branch( 'st_pfMetUncJetEnUp', st_pfMetUncJetEnUp )
        outTree.Branch( 'st_pfMetUncJetEnDown', st_pfMetUncJetEnDown)
        outTree.Branch( 'st_isData', st_isData , 'st_isData/O')

        for trigs in triglist:
            exec("outTree.Branch( 'st_"+trigs+"', st_"+trigs+" , 'st_"+trigs+"/O')")

        outTree.Branch( 'st_THINnJet',st_THINnJet, 'st_THINnJet/L' )
        outTree.Branch( 'st_THINjetPx', st_THINjetPx  )
        outTree.Branch( 'st_THINjetPy' , st_THINjetPy )
        outTree.Branch( 'st_THINjetPz', st_THINjetPz )
        outTree.Branch( 'st_THINjetEnergy', st_THINjetEnergy )
        outTree.Branch( 'st_THINjetDeepCSV',st_THINjetDeepCSV )
        outTree.Branch( 'st_THINjetHadronFlavor',st_THINjetHadronFlavor )
        outTree.Branch( 'st_THINjetNHadEF',st_THINjetNHadEF )
        outTree.Branch( 'st_THINjetCHadEF',st_THINjetCHadEF )

        outTree.Branch( 'st_THINjetCEmEF',st_THINjetCEmEF )
        outTree.Branch( 'st_THINjetPhoEF',st_THINjetPhoEF )
        outTree.Branch( 'st_THINjetEleEF',st_THINjetEleEF )
        outTree.Branch( 'st_THINjetMuoEF',st_THINjetMuoEF )
        outTree.Branch('st_THINjetCorrUnc', st_THINjetCorrUnc)

        outTree.Branch( 'st_nEle',st_nEle , 'st_nEle/L')
        outTree.Branch( 'st_elePx', st_elePx  )
        outTree.Branch( 'st_elePy' , st_elePy )
        outTree.Branch( 'st_elePz', st_elePz )
        outTree.Branch( 'st_eleEnergy', st_eleEnergy )
        outTree.Branch( 'st_eleIsPassTight', st_eleIsPassTight)#, 'st_eleIsPassTight/O' )
        outTree.Branch( 'st_eleIsPassLoose', st_eleIsPassLoose)#, 'st_eleIsPassLoose/O' )

        outTree.Branch( 'st_nPho',st_nPho , 'st_nPho/L')
        outTree.Branch( 'st_phoIsPassTight', st_phoIsPassTight)#, 'st_phoIsPassTight/O' )
        outTree.Branch( 'st_phoPx', st_phoPx  )
        outTree.Branch( 'st_phoPy' , st_phoPy )
        outTree.Branch( 'st_phoPz', st_phoPz )
        outTree.Branch( 'st_phoEnergy', st_phoEnergy )


        outTree.Branch( 'st_nMu',st_nMu , 'st_nMu/L')
        outTree.Branch( 'st_muPx', st_muPx)
        outTree.Branch( 'st_muPy' , st_muPy)
        outTree.Branch( 'st_muPz', st_muPz)
        outTree.Branch( 'st_muEnergy', st_muEnergy)
        outTree.Branch( 'st_isTightMuon', st_isTightMuon)#, 'st_isTightMuon/O' )
        outTree.Branch( 'st_muIso', st_muIso)#, 'st_muIso/F')

        outTree.Branch( 'st_HPSTau_n', st_HPSTau_n, 'st_HPSTau_n/L')
        outTree.Branch( 'st_nTauTightElectron', st_nTauTightElectron, 'st_nTauTightElectron/L')
        outTree.Branch( 'st_nTauTightMuon', st_nTauTightMuon, 'st_nTauTightMuon/L')
        outTree.Branch( 'st_nTauTightEleMu', st_nTauTightEleMu, 'st_nTauTightEleMu/L')
        outTree.Branch( 'st_nTauLooseEleMu', st_nTauLooseEleMu, 'st_nTauLooseEleMu/L')

        outTree.Branch( 'st_pu_nTrueInt', st_pu_nTrueInt, 'st_pu_nTrueInt/F')
        outTree.Branch( 'st_pu_nPUVert', st_pu_nPUVert, 'st_pu_nPUVert/F')
        outTree.Branch( 'st_THINjetNPV', st_THINjetNPV, 'st_THINjetNPV/F')
        outTree.Branch( 'mcweight', mcweight, 'mcweight/F')
        outTree.Branch( 'st_nGenPar',st_nGenPar,'st_nGenPar/L' )  #nGenPar/I
        outTree.Branch( 'st_genParId',st_genParId )  #vector<int>
        outTree.Branch( 'st_genMomParId',st_genMomParId )
        outTree.Branch( 'st_genParSt',st_genParSt )
        outTree.Branch( 'st_genParPx', st_genParPx  )
        outTree.Branch( 'st_genParPy' , st_genParPy )
        outTree.Branch( 'st_genParPz', st_genParPz )
        outTree.Branch( 'st_genParEnergy', st_genParEnergy )

        outTree.Branch( 'WenuRecoil', WenuRecoil, 'WenuRecoil/F')
        outTree.Branch( 'Wenumass', Wenumass, 'Wenumass/F')
        outTree.Branch( 'WenuPhi', WenuPhi, 'WenuPhi/F')

        outTree.Branch( 'WmunuRecoil', WmunuRecoil, 'WmunuRecoil/F')
        outTree.Branch( 'Wmunumass', Wmunumass, 'Wmunumass/F')
        outTree.Branch( 'WmunuPhi', WmunuPhi, 'WmunuPhi/F')

        outTree.Branch( 'ZeeRecoil', ZeeRecoil, 'ZeeRecoil/F')
        outTree.Branch( 'ZeeMass', ZeeMass, 'ZeeMass/F')
        outTree.Branch( 'ZeePhi', ZeePhi, 'ZeePhi/F')

        outTree.Branch( 'ZmumuRecoil', ZmumuRecoil, 'ZmumuRecoil/F')
        outTree.Branch( 'ZmumuMass', ZmumuMass, 'ZmumuMass/F')
        outTree.Branch( 'ZmumuPhi', ZmumuPhi, 'ZmumuPhi/F')

        outTree.Branch( 'GammaRecoil', GammaRecoil, 'GammaRecoil/F')
        outTree.Branch( 'GammaPhi', GammaPhi, 'GammaPhi/F')
        for run,lumi,event,isData,mcWeight_,pu_nTrueInt_,pu_nPUVert_,trigName_,trigResult_,filterName,filterResult,met_,metphi_,metUnc_,nele_,elepx_,elepy_,elepz_,elee_,elelooseid_,eleTightid_,eleCharge_,npho_,phopx_,phopy_,phopz_,phoe_,pholooseid_,photightID_,nmu_,mupx_,mupy_,mupz_,mue_,mulooseid_,mutightid_,muChHadIso_,muNeHadIso_,muGamIso_,muPUPt_,muCharge_,nTau_,tau_px_,tau_py_,tau_pz_,tau_e_,tau_dm_,tau_isLoose_,nGenPar_,genParId_,genMomParId_,genParSt_,genpx_,genpy_,genpz_,gene_,nak4jet_,ak4px_,ak4py_,ak4pz_,ak4e_,ak4TightID_,ak4deepcsv_,ak4flavor_,ak4NHEF_,ak4CHEF_,ak4CEmEF_,ak4PhEF_,ak4EleEF_,ak4MuEF_, ak4JEC_, ak4NPV_ in zip(df.runId,df.lumiSection,df.eventId,df.isData,df.mcWeight,df.pu_nTrueInt,df.pu_nPUVert,df.hlt_trigName,df.hlt_trigResult,df.hlt_filterName,df.hlt_filterResult,df.pfMetCorrPt,df.pfMetCorrPhi,df.pfMetCorrUnc,df.nEle,df.elePx,df.elePy,df.elePz,df.eleEnergy,df.eleIsPassLoose,df.eleIsPassTight,df.eleCharge,df.nPho,df.phoPx,df.phoPy,df.phoPz,df.phoEnergy,df.phoIsPassLoose,df.phoIsPassTight,df.nMu,df.muPx,df.muPy,df.muPz,df.muEnergy,df.isLooseMuon,df.isTightMuon,df.muChHadIso,df.muNeHadIso,df.muGamIso,df.muPUPt,df.muCharge,df.HPSTau_n,df.HPSTau_Px,df.HPSTau_Py,df.HPSTau_Pz,df.HPSTau_Energy,df.disc_decayModeFinding,df.disc_byLooseIsolationMVArun2v1DBoldDMwLT2016,df.nGenPar,df.genParId,df.genMomParId,df.genParSt,df.genPx,df.genPy,df.genPz,df.genEnergy,df.THINnJet,df.THINjetPx,df.THINjetPy,df.THINjetPz,df.THINjetEnergy,df.THINjetPassIDTight,df.THINjetDeepCSV_b,df.THINjetHadronFlavor,df.THINjetNHadEF,df.THINjetCHadEF,df.THINjetCEmEF,df.THINjetPhoEF,df.THINjetEleEF,df.THINjetMuoEF,df.THINjetCorrUncUp,df.THINjetNPV):
            if ieve%500==0: print "Processed",ieve,"Events"
            ieve = ieve + 1
            # -------------------------------------------------
            # MC Weights
            # -------------------------------------------------
            mcweight[0] = 0.0
            if isData==1:   mcweight[0] =  1.0
            if not isData :
                if mcWeight_<0:  mcweight[0] = -1.0
                if mcWeight_>0:  mcweight[0] =  1.0
            h_total.Fill(1.);
            h_total_mcweight.Fill(1.,mcweight[0]);

            # -------------------------------------------------
            ## Trigger selection
            # -------------------------------------------------

            trigstatus=False
            for itrig in range(len(triglist)):
                exec(triglist[itrig]+" = CheckFilter(trigName_, trigResult_, " + "'" + triglist[itrig] + "')")
                exec("if "+triglist[itrig]+": trigstatus=True")
                exec("st_"+triglist[itrig]+"[0]="+triglist[itrig])
            if not isData: trigstatus=True
            if not trigstatus: continue

            # ------------------------------------------------------
            ## Filter selection
            # ------------------------------------------------------
            filterstatus = False
            filter1 = False; filter2 = False;filter3 = False;filter4 = False; filter5 = False; filter6 = False; filter7=False
            ifilter_=0
            filter1 = CheckFilter(filterName, filterResult, 'Flag_HBHENoiseFilter')
            filter2 = CheckFilter(filterName, filterResult, 'Flag_globalTightHalo2016Filter')
            filter3 = CheckFilter(filterName, filterResult, 'Flag_eeBadScFilter')
            filter4 = CheckFilter(filterName, filterResult, 'Flag_goodVertices')
            filter5 = CheckFilter(filterName, filterResult, 'Flag_EcalDeadCellTriggerPrimitiveFilter')
            fileer6 = CheckFilter(filterName, filterResult, 'Flag_BadPFMuonFilter')
            filter7 = CheckFilter(filterName, filterResult, 'Flag_HBHENoiseIsoFilter')
            if not isData:
                filterstatus = True
            if isData:
                filterstatus =  filter1 & filter2 & filter3 & filter4 & filter5 & filter6 & filter7
            if filterstatus == False: continue

            # ------------------------------------------------------
            ## PFMET Selection
            # --------------------------------------------------------
            pfmetstatus = ( met_ > 200.0 )

            '''
            *******   *      *   ******
            *     *   *      *  *      *
            *******   ********  *      *
            *         *      *  *      *
            *         *      *   ******
            '''
            phopt = [getPt(phopx_[ip], phopy_[ip]) for ip in range(npho_)]
            phoeta = [getEta(phopx_[ip], phopy_[ip], phopz_[ip]) for ip in range(npho_)]

            pho_pt15 = [(phopt[ip] > 15.0) for ip in range(npho_)]
            pho_eta2p5 = [(abs(phoeta[ip]) < 2.5) for ip in range(npho_)]
            pho_IDLoose = [(pholooseid_[ip]) for ip in range(npho_)]

            pho_pt15_eta2p5_looseID = []
            if len(pho_pt15) > 0:
                pho_pt15_eta2p5_looseID = logical_AND_List3(pho_pt15, pho_eta2p5, pho_IDLoose)

            pass_pho_index = WhereIsTrue(pho_pt15_eta2p5_looseID, 1)
            if debug_: print "pass_ele_index",pass_ele_index
            '''
            ****   *      ****
            *      *      *
            ***    *      ***
            *      *      *
            ****   ****   ****
            '''
            elept = [getPt(elepx_[ie], elepy_[ie]) for ie in range(nele_)]
            eleeta = [getEta(elepx_[ie], elepy_[ie], elepz_[ie]) for ie in range(nele_)]
            elephi = [getPhi(elepx_[ie], elepy_[ie]) for ie in range(nele_)]

            ele_pt10 = [(elept[ie] > 10.0) for ie in range(nele_)]
            ele_eta2p5 = [((abs(eleeta[ie]) > 1.566 or abs(eleeta[ie]) < 1.4442) and (abs(eleeta[ie]) < 2.5)) for ie in range(nele_)]
            ele_IDLoose = [(elelooseid_[ie]) for ie in range(nele_)]

            ele_pt10_eta2p5_looseID = []
            if len(ele_pt10) > 0:
                ele_pt10_eta2p5_looseID = logical_AND_List3(ele_pt10, ele_eta2p5, ele_IDLoose)

            pass_ele_index = WhereIsTrue(ele_pt10_eta2p5_looseID, 1)
            if debug_: print "pass_ele_index",pass_ele_index

            '''
            **     *  *     *
            * *  * *  *     *
            *  *   *  *     *
            *      *  *     *
            *      *   *****
            '''
            mupt = [getPt(mupx_[imu], mupy_[imu]) for imu in range(nmu_)]
            mueta = [getEta(mupx_[imu], mupy_[imu], mupz_[imu]) for imu in range(nmu_)]
            muphi = [getPhi(mupx_[imu], mupy_[imu]) for imu in range(nmu_)]
            muIso_ = [((muChHadIso_[imu]+ max(0., muNeHadIso_[imu] + muGamIso_[imu] - 0.5*muPUPt_[imu]))/mupt[imu]) for imu in range(nmu_)]

            mu_pt10 = [(mupt[imu] > 10.0) for imu in range(nmu_)]
            mu_eta2p4 = [(abs(mueta[imu]) < 2.4) for imu in range(nmu_)]
            mu_IDLoose = [mulooseid_[imu] for imu in range(nmu_)]
            mu_IsoLoose = [(muIso_[imu] < 0.25) for imu in range(nmu_)]

            mu_pt10_eta2p4_looseID_looseISO = []
            if len(mu_pt10) > 0:
                mu_pt10_eta2p4_looseID_looseISO = logical_AND_List2(mu_IDLoose, mu_IsoLoose)

            pass_mu_index = WhereIsTrue(mu_pt10_eta2p4_looseID_looseISO, 1)
            if debug_: print "pass_mu_index",pass_mu_index

            '''
            *******   *****   *******
               *      *          *
               *      ****       *
               *      *          *
            ***       *****      *
            '''
            ak4pt = [getPt(ak4px_[ij], ak4py_[ij]) for ij in range(nak4jet_)]
            ak4eta = [getEta(ak4px_[ij], ak4py_[ij], ak4pz_[ij]) for ij in range(nak4jet_)]
            ak4phi = [getPhi(ak4px_[ij], ak4py_[ij]) for ij in range(nak4jet_)]
            ak4_IDTightVeto = [ak4TightID_[ij] for ij in range(nak4jet_)]

            ak4_pt30 = [(ak4pt[ij] > 30.0) for ij in range(nak4jet_)]
            ak4_eta4p5 = [(abs(ak4eta[ij]) < 4.5) for ij in range(nak4jet_)]

            ak4_pt30_eta4p5_IDT = []
            if len(ak4_pt30) > 0:
                ak4_pt30_eta4p5_IDT = logical_AND_List3(ak4_pt30, ak4_eta4p5, ak4_IDTightVeto)

            jetCleanAgainstEle = []
            jetCleanAgainstMu = []
            pass_jet_index_cleaned = []
            if len(ak4_pt30_eta4p5_IDT) > 0:
                for ijet in range(len(ak4_pt30_eta4p5_IDT)):
                    pass_ijet_iele_ = []
                    for iele in range(len(ele_pt10_eta2p5_looseID)):
                        pass_ijet_iele_.append(ak4_pt30_eta4p5_IDT[ijet] and ele_pt10_eta2p5_looseID[iele] and (
                            Delta_R(ak4eta[ijet], eleeta[iele], ak4phi[ijet], elephi[iele]) > 0.4))
                    # if the number of true is equal to length of vector then it is ok to keep this jet, otherwise this is not cleaned
                    jetCleanAgainstEle.append(len(WhereIsTrue(pass_ijet_iele_)) == len(pass_ijet_iele_))
                    if debug_:
                        print "pass_ijet_iele_ = ", pass_ijet_iele_
                        print "jetCleanAgainstEle = ", jetCleanAgainstEle
                for ijet in range(len(ak4_pt30_eta4p5_IDT)):
                    pass_ijet_imu_ = []
                    for imu in range(len(mu_pt10_eta2p4_looseID_looseISO)):
                        pass_ijet_imu_.append(ak4_pt30_eta4p5_IDT[ijet] and mu_pt10_eta2p4_looseID_looseISO[imu] and (Delta_R(ak4eta[ijet], mueta[imu], ak4phi[ijet], muphi[imu]) > 0.4))
                    # if the number of true is equal to length of vector then it is ok to keep this jet, otherwise this is not cleaned
                    if debug_:print "pass_ijet_imu_ = ", pass_ijet_imu_
                    jetCleanAgainstMu.append(len(WhereIsTrue(pass_ijet_imu_)) == len(pass_ijet_imu_))
                    if debug_:print "jetCleanAgainstMu = ", jetCleanAgainstMu

                jetCleaned = logical_AND_List2(jetCleanAgainstEle, jetCleanAgainstMu)
                pass_jet_index_cleaned = WhereIsTrue(jetCleaned, 3)
                if debug_:print "pass_jet_index_cleaned = ", pass_jet_index_cleaned,"nJets= ",len(ak4px_)

            '''
            ********    *        *       *
               *      *    *     *       *
               *     *      *    *       *
               *     ********    *       *
               *     *      *    *       *
               *     *      *     *******
            '''
            taupt = [getPt(tau_px_[itau], tau_py_[itau]) for itau in range(nTau_)]
            taueta = [getEta(tau_px_[itau], tau_py_[itau], tau_pz_[itau]) for itau in range(nTau_)]
            tauphi = [getPhi(tau_px_[itau], tau_py_[itau]) for itau in range(nTau_)]

            tau_pt18 = [(taupt[itau] > 18.0) for itau in range(nTau_)]
            tau_eta2p3 = [(abs(taueta[itau]) < 2.3) for itau in range(nTau_)]
            tau_IDLoose = [(tau_isLoose_[itau]) for itau in range(nTau_)]
            tau_DM = [(tau_dm_[itau]) for itau in range(nTau_)]

            tau_pt18_eta2p3 = []
            if (len(tau_pt18) > 0 and len(tau_eta2p3) > 0):
                tau_pt18_eta2p3 = logical_AND_List2(tau_pt18, tau_eta2p3)
            if debug_:print "tau_pt18_eta2p3 = ", tau_pt18_eta2p3

            ''' take AND of all the tau cuts (just take the lists) '''
            tau_eta2p3_iDLdm_pt18 = []
            if (len(tau_eta2p3) > 0):
                tau_eta2p3_iDLdm_pt18 = logical_AND_List4(
                    tau_eta2p3, tau_DM, tau_pt18,tau_IDLoose )
            if debug_:print "tau_eta2p3_iDLdm_pt18 = ", tau_eta2p3_iDLdm_pt18

            tauCleanAgainstEle = []
            tauCleanAgainstMu = []
            pass_tau_index_cleaned = []
            if len(tau_pt18_eta2p3)>0:
                for itau in range(len(tau_pt18_eta2p3)):
                    pass_itau_iele_ = []
                    for iele in range(len(ele_pt10_eta2p5_looseID)):
                        pass_itau_iele_.append(tau_pt18_eta2p3[itau] and ele_pt10_eta2p5_looseID[iele] and (Delta_R(taueta[itau], eleeta[iele], tauphi[itau], elephi[iele]) > 0.4))
                    # if the number of true is equal to length of vector then it is ok to keep this jet, otherwise this is not cleaned
                    tauCleanAgainstEle.append(len(WhereIsTrue(pass_itau_iele_)) == len(pass_itau_iele_))
                    if debug_:
                        print "pass_itau_iele_ = ", pass_itau_iele_
                        print "tauCleanAgainstEle = ", tauCleanAgainstEle
                for itau in range(len(tau_pt18_eta2p3)):
                    pass_itau_imu_ = []
                    for imu in range(len(mu_pt10_eta2p4_looseID_looseISO)):
                        pass_itau_imu_.append(tau_pt18_eta2p3[itau] and mu_pt10_eta2p4_looseID_looseISO[imu] and (Delta_R(taueta[itau], mueta[imu], tauphi[itau], muphi[imu]) > 0.4))
                    # if the number of true is equal to length of vector then it is ok to keep this jet, otherwise this is not cleaned
                    if debug_:print "pass_itau_imu_ = ", pass_itau_imu_
                    tauCleanAgainstMu.append(len(WhereIsTrue(pass_itau_imu_)) == len(pass_itau_imu_))
                    if debug_:print "tauCleanAgainstMu = ", tauCleanAgainstMu

                tauCleaned = logical_AND_List2(tauCleanAgainstEle, tauCleanAgainstMu)
                pass_tau_index_cleaned = WhereIsTrue(tauCleaned,3)
                if debug_:print "pass_tau_index_cleaned",pass_tau_index_cleaned

            # -------------------------------------------------------------
            st_runId[0]             = long(run)
            st_lumiSection[0]       = lumi
            st_eventId[0]           = event
            st_isData[0]            = isData
            st_pfMetCorrPt[0]       = met_
            st_pfMetCorrPhi[0]      = metphi_

            st_pfMetUncJetResUp.clear()
            st_pfMetUncJetResDown.clear()

            st_pfMetUncJetEnUp.clear()
            st_pfMetUncJetEnDown.clear()

            st_THINjetPx.clear()
            st_THINjetPy.clear()
            st_THINjetPz.clear()
            st_THINjetEnergy.clear()
            st_THINjetDeepCSV.clear()
            st_THINjetHadronFlavor.clear()
            st_THINjetNHadEF.clear()
            st_THINjetCHadEF.clear()

            st_THINjetCEmEF.clear()
            st_THINjetPhoEF.clear()
            st_THINjetEleEF.clear()
            st_THINjetMuoEF.clear()
            st_THINjetCorrUnc.clear()

            st_elePx.clear()
            st_elePy.clear()
            st_elePz.clear()
            st_eleEnergy.clear()
            st_eleIsPassTight.clear()
            st_eleIsPassLoose.clear()

            st_muPx.clear()
            st_muPy.clear()
            st_muPz.clear()
            st_muEnergy.clear()
            st_isTightMuon.clear()
            st_muIso.clear()

            st_phoPx.clear()
            st_phoPy.clear()
            st_phoPz.clear()
            st_phoEnergy.clear()
            st_phoIsPassTight.clear()

            st_genParId.clear()
            st_genMomParId.clear()
            st_genParSt.clear()
            st_genParPx.clear()
            st_genParPy.clear()
            st_genParPz.clear()
            st_genParEnergy.clear()

            st_THINnJet[0] = len(pass_jet_index_cleaned)
            for ithinjet in pass_jet_index_cleaned:
                st_THINjetPx.push_back(ak4px_[ithinjet])
                st_THINjetPy.push_back(ak4py_[ithinjet])
                st_THINjetPz.push_back(ak4pz_[ithinjet])
                st_THINjetEnergy.push_back(ak4e_[ithinjet])
                st_THINjetDeepCSV.push_back(ak4deepcsv_[ithinjet])
                st_THINjetHadronFlavor.push_back(int(ak4flavor_[ithinjet]))
                st_THINjetNHadEF.push_back(ak4NHEF_[ithinjet])
                st_THINjetCHadEF.push_back(ak4CHEF_[ithinjet])

                st_THINjetCEmEF.push_back(ak4CEmEF_[ithinjet])
                st_THINjetPhoEF.push_back(ak4PhEF_[ithinjet])
                st_THINjetEleEF.push_back(ak4EleEF_[ithinjet])
                st_THINjetMuoEF.push_back(ak4MuEF_[ithinjet])
                st_THINjetCorrUnc.push_back(ak4JEC_[ithinjet])
            if debug_:print 'njets: ',len(pass_jet_index_cleaned)

            st_nEle[0] = len(pass_ele_index)
            for iele in pass_ele_index:
                st_elePx.push_back(elepx_[iele])
                st_elePy.push_back(elepy_[iele])
                st_elePz.push_back(elepz_[iele])
                st_eleEnergy.push_back(elee_[iele])
                st_eleIsPassTight.push_back(bool(eleTightid_[iele]))
            if debug_:print 'nEle: ',len(pass_ele_index)

            st_nMu[0] = len(pass_mu_index)
            for imu in pass_mu_index:
                st_muPx.push_back(mupx_[imu])
                st_muPy.push_back(mupy_[imu])
                st_muPz.push_back(mupz_[imu])
                st_muEnergy.push_back(mue_[imu])
                st_isTightMuon.push_back(bool(mutightid_[imu]))
                st_muIso.push_back(muIso_[imu])
            if debug_:print 'nMu: ',len(pass_mu_index)

            st_HPSTau_n[0] = len(pass_tau_index_cleaned)
            # st_nTauTightElectron[0] = len(myTausTightElectron)
            # st_nTauTightMuon[0] = len(myTausTightMuon)
            # st_nTauTightEleMu[0] = len(myTausTightEleMu)
            # st_nTauLooseEleMu[0] = len(myTausLooseEleMu)
            if debug_:print 'nTau: ',len(pass_tau_index_cleaned)

            st_nPho[0]=len(pass_pho_index)
            for ipho in pass_pho_index:
                st_phoPx.push_back(phopx_[ipho])
                st_phoPy.push_back(phopy_[ipho])
                st_phoPz.push_back(phopz_[ipho])
                st_phoEnergy.push_back(phoe_[ipho])
                st_phoIsPassTight.push_back(bool(photightID_[ipho]))
            if debug_:print 'nPho: ',len(pass_pho_index)

            st_pu_nTrueInt[0] = pu_nTrueInt_
            st_pu_nPUVert[0] = pu_nPUVert_
            st_THINjetNPV[0] = ak4NPV_

            st_nGenPar[0] =  nGenPar_
            for igp in range(nGenPar_):
                st_genParId.push_back(int(genParId_[igp]))
                st_genMomParId.push_back(int(genMomParId_[igp]))
                st_genParSt.push_back(int(genParSt_[igp]))
                st_genParPx.push_back(genpx_[igp])
                st_genParPy.push_back(genpy_[igp])
                st_genParPz.push_back(genpz_[igp])
                st_genParEnergy.push_back(gene_[igp])
            if debug_: print 'nGen: ',nGenPar_

            st_pfMetUncJetResUp.push_back(metUnc_[0])
            st_pfMetUncJetResDown.push_back(metUnc_[1])
            st_pfMetUncJetEnUp.push_back(metUnc_[2])
            st_pfMetUncJetEnDown.push_back(metUnc_[3])

            ## Fill variables for the CRs.
            WenuRecoil[0] = -1.0
            Wenumass[0] = -1.0
            WenuPhi[0] = -10.

            WmunuRecoil[0] = -1.0
            Wmunumass[0] = -1.0
            WmunuPhi[0] = -10.

            ZeeMass[0] = -1.0
            ZeeRecoil[0] = -1.0
            ZeePhi[0] = -10.

            ZmumuMass[0] = -1.0
            ZmumuRecoil[0] = -1.0
            ZmumuPhi[0] = -10.

            GammaRecoil[0] = -1.0
            GammaPhi[0]  = -10.
            if debug_: print 'Reached Fill variables'

            # ------------------
            # Z CR
            # ------------------
            ## for dielectron
            if len(pass_ele_index) == 2:
                iele1=pass_ele_index[0]
                iele2=pass_ele_index[1]
                if eleCharge_[iele1]*eleCharge_[iele2]<0:
                    ee_mass = InvMass(elepx_[iele1],elepy_[iele1],elepz_[iele1],elee_[iele1],elepx_[iele2],elepy_[iele2],elepz_[iele2],elee_[iele2])
                    zeeRecoilPx = -( met_*math.cos(metphi_) + elepx_[iele1] + elepx_[iele2])
                    zeeRecoilPy = -( met_*math.sin(metphi_) + elepy_[iele1] + elepy_[iele2])
                    ZeeRecoilPt =  math.sqrt(zeeRecoilPx**2  +  zeeRecoilPy**2)
                    if ee_mass > 70.0 and ee_mass < 110.0 and ZeeRecoilPt > 200.:
                        ZeeRecoil[0] = ZeeRecoilPt
                        ZeeMass[0] = ee_mass
                        ZeePhi[0] = arctan(zeeRecoilPx,zeeRecoilPy)
            ## for dimu
            if len(pass_mu_index) ==2:
                imu1=pass_mu_index[0]
                imu2=pass_mu_index[1]
                if muCharge_[imu1]*muCharge_[imu2]<0:
                    mumu_mass = InvMass(mupx_[imu1],mupy_[imu1],mupz_[imu1],mue_[imu1],mupx_[imu2],mupy_[imu2],mupz_[imu2],mue_[imu2] )
                    zmumuRecoilPx = -( met_*math.cos(metphi_) + mupx_[imu1] + mupx_[imu2])
                    zmumuRecoilPy = -( met_*math.sin(metphi_) + mupy_[imu1] + mupy_[imu2])
                    ZmumuRecoilPt =  math.sqrt(zmumuRecoilPx**2  +  zmumuRecoilPy**2)
                    if mumu_mass > 70.0 and mumu_mass < 110.0 and ZmumuRecoilPt > 200.:
                        ZmumuRecoil[0] = ZmumuRecoilPt
                        ZmumuMass[0] = mumu_mass
                        ZmumuPhi[0] = arctan(zmumuRecoilPx,zmumuRecoilPy)
            if len(pass_ele_index) == 2:
                ZRecoilstatus =(ZeeRecoil[0] > 200)
            elif len(pass_mu_index) == 2:
                ZRecoilstatus =(ZmumuRecoil[0] > 200)
            else:
                ZRecoilstatus=False
            if debug_: print 'Reached Z CR'

            # ------------------
            # W CR
            # ------------------
            ## for Single electron
            if len(pass_ele_index) == 1:
               ele1 = pass_ele_index[0]
               e_mass = MT(elept[ele1],met_, DeltaPhi(elephi[ele1],metphi_)) #transverse mass defined as sqrt{2pT*MET*(1-cos(dphi)}
               WenuRecoilPx = -( met_*math.cos(metphi_) + elepx_[ele1])
               WenuRecoilPy = -( met_*math.sin(metphi_) + elepy_[ele1])
               WenuRecoilPt = math.sqrt(WenuRecoilPx**2  +  WenuRecoilPy**2)
               if WenuRecoilPt > 200.:
                   WenuRecoil[0] = WenuRecoilPt
                   Wenumass[0] = e_mass
                   WenuPhi[0] = arctan(WenuRecoilPx,WenuRecoilPy)
            ## for Single muon
            if len(pass_mu_index) == 1:
               mu1 = pass_mu_index[0]
               mu_mass = MT(mupt[mu1],met_, DeltaPhi(muphi[mu1],metphi_)) #transverse mass defined as sqrt{2pT*MET*(1-cos(dphi)}
               WmunuRecoilPx = -( met_*math.cos(metphi_) + mupx_[mu1])
               WmunuRecoilPy = -( met_*math.sin(metphi_) + mupy_[mu1])
               WmunuRecoilPt = math.sqrt(WmunuRecoilPx**2  +  WmunuRecoilPy**2)
               if WmunuRecoilPt > 200.:
                   WmunuRecoil[0] = WmunuRecoilPt
                   Wmunumass[0] = mu_mass
                   WmunuPhi[0] = arctan(WmunuRecoilPx,WmunuRecoilPy)
            if len(pass_ele_index) == 1:
                WRecoilstatus =(WenuRecoil[0] > 200)
            elif len(pass_mu_index) == 1:
                WRecoilstatus =(WmunuRecoil[0] > 200)
            else:
                WRecoilstatus=False
            if debug_: print 'Reached W CR'

            # ------------------
            # Gamma CR
            # ------------------
            ## for Single photon
            if len(pass_pho_index) >= 1:
               pho1 = pass_pho_index[0]
               GammaRecoilPx = -( met_*math.cos(metphi_) + phopx_[pho1])
               GammaRecoilPy = -( met_*math.sin(metphi_) + phopy_[pho1])
               GammaRecoilPt = math.sqrt(GammaRecoilPx**2  +  GammaRecoilPy**2)
               if GammaRecoilPt > 200.:
                   GammaRecoil[0] = GammaRecoilPt
                   GammaPhi[0] = arctan(GammaRecoilPx,GammaRecoilPy)
            GammaRecoilStatus = (GammaRecoil[0] > 200)
            if debug_: print 'Reached Gamma CR'

            if pfmetstatus==False and ZRecoilstatus==False and WRecoilstatus==False and GammaRecoilStatus==False: continue
            outTree.Fill()
        outfile.cd()
        h_total_mcweight.Write()
        h_total.Write()
        samplepath.Write()
        outfile.Write()

    end = time.clock()
    print "%.4gs" % (end-start)

def CheckFilter(filterName, filterResult,filtercompare):
    ifilter_=0
    filter1 = False
    for ifilter in filterName:
        filter1 = (ifilter.find(filtercompare) != -1)  & (bool(filterResult[ifilter_]) == True)
        if filter1: break
        ifilter_ = ifilter_ + 1
    return filter1

def DeltaR(p4_1, p4_2):
    eta1 = p4_1.Eta()
    eta2 = p4_2.Eta()
    eta = eta1 - eta2
    eta_2 = eta * eta

    phi1 = p4_1.Phi()
    phi2 = p4_2.Phi()
    phi = Phi_mpi_pi(phi1-phi2)
    phi_2 = phi * phi

    return math.sqrt(eta_2 + phi_2)

def Phi_mpi_pi(x):
    kPI = 3.14159265358979323846
    kTWOPI = 2 * kPI

    while (x >= kPI): x = x - kTWOPI;
    while (x < -kPI): x = x + kTWOPI;
    return x;

def DeltaPhi(phi1,phi2):
   phi = Phi_mpi_pi(phi1-phi2)

   return abs(phi)

def CheckFilter(filterName, filterResult,filtercompare):
    ifilter_=0
    filter1 = False
    for ifilter in filterName:
        filter1 = (ifilter.find(filtercompare) != -1)  & (bool(filterResult[ifilter_]) == True)
        if filter1: break
        ifilter_ = ifilter_ + 1
    return filter1

def MT(Pt, met, dphi):
    return ROOT.TMath.Sqrt( 2 * Pt * met * (1.0 - ROOT.TMath.Cos(dphi)) )

def InvMass(px1,py1,pz1,pe1,px2,py2,pz2,pe2):
    p1 = TLorentzVector(px1, py1, pz1, pe1)
    p2 = TLorentzVector(px2, py2, pz2, pe2)
    inv_mass = (p1+p2).M()
    return inv_mass

files = []
if not isfarmout:
    files.append(infilename)
elif isfarmout:
    infile = open(infilename)
    failcount=0
    for ifile in infile:
        files.append(ifile.rstrip())

if __name__ == '__main__':
    try:
        pool = mp.Pool()
        pool.map(runbbdm, files)
        pool.close()
    except Exception as e:
        print traceback.format_exc()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        pass
