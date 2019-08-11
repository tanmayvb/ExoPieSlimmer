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

################################################################################################################
##---changes to be made: 
##--------- 1. one trigger flag for one physics object or one CR 
##--------- 2. write trigger list in another python file 
##--------- 3. filter list in another python file 
##--------- 4.  import the class TLorentzVector.h from ROOT, so that p4 can be used without using ROOT. 
##--------- 4.1 tried it, it is very difficult, its better to write own class p4. and use it for all kind of operation. in this one can implement, the functions 
##---------      pt, eta, phi, mass, delta R, delta phi, + , -  (anything else we use?) 
##--------- 5. rename jetvariables and move to a new file 
##--- 6. move https://github.com/ExoPie/ExoPieSlimmer/blob/master/SkimTree.py#L183-L273 into a function
##--- 7. add AK8 jet information
##-------- 8. update the variable names
##-------- 9. use the out of new ExoPieElement as input for the skimmer
##--- 10. convert this to function: https://github.com/ExoPie/ExoPieSlimmer/blob/master/SkimTree.py#L455-L462
################################################################################################################
from multiprocessing import Process
import multiprocessing as mp




## user packages 
## in local dir 
import  triggers as trig
import variables as branches 
import filters as filters

## from commonutils
sys.path.append('../ExoPieUtils/commonutils/')
import MathUtils as mathutil
from MathUtils import *
import BooleanUtils as boolutil


## from analysisutils
sys.path.append('../ExoPieUtils/analysisutils/')
import analysis_utils as anautil 

######################################################################################################
## All import are done before this 
######################################################################################################


## ----- start if clock 

start = time.clock()





## ----- command line argument 
usage = "analyzer for bb+DM (debugging) "
parser = argparse.ArgumentParser(description=usage)
parser.add_argument("-i", "--inputfile",  dest="inputfile")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="out.root")
parser.add_argument("-D", "--outputdir", dest="outputdir")
parser.add_argument("-F", "--farmout", action="store_true",  dest="farmout")
## add argument for debug, default to be false

args = parser.parse_args()

if args.farmout==None:
    isfarmout = False
else:
    isfarmout = args.farmout

infilename = "NCUGlobalTuples.root"

debug_ = False


'''
def jetcleaning(ak4_pt30_eta4p5_IDT, lep_looseID, ak4eta, lepeta, ak4phi, lepphi, DRCut):
    ## usage: (obj_to_clean, obj_cleaned_against, so on
    if debug_: print "njet, nlep", len(ak4_pt30_eta4p5_IDT), len(lep_looseID)
    jetCleanAgainstLep = []
    pass_jet_index_cleaned = []
    if len(ak4_pt30_eta4p5_IDT) > 0:
        for ijet in range(len(ak4_pt30_eta4p5_IDT)):
            pass_ijet_ilep_ = []
            for ilep in range(len(lep_looseID)):
                pass_ijet_ilep_.append(ak4_pt30_eta4p5_IDT[ijet] and lep_looseID[ilep] and (Delta_R(ak4eta[ijet], lepeta[ilep], ak4phi[ijet], lepphi[ilep]) > 0.4))
            if debug_: print "-------- pass_ijet_ilep_ = ",pass_ijet_ilep_
                # if the number of true is equal to length of vector then it is ok to keep this jet, otherwise this is not cleaned
            jetCleanAgainstLep.append(len(boolutil.WhereIsTrue(pass_ijet_ilep_)) == len(pass_ijet_ilep_))
            if debug_: print "inside function pass_ijet_ilep_ = ", pass_ijet_ilep_
            if debug_: print "inside function jetCleanAgainstLep = ", jetCleanAgainstLep
    return jetCleanAgainstLep
'''
def runbbdm(infile_):
    prefix="Skimmed_"
    outfilename= prefix+infile_.split("/")[-1]
    
    
    outputfilename = args.outputfile
    h_total = TH1F('h_total','h_total',2,0,2)
    h_total_mcweight = TH1F('h_total_mcweight','h_total_mcweight',2,0,2)
    
    triglist = trig.trigger2016
    
    outfile = TFile(outfilename,'RECREATE')

    outTree = TTree( 'outTree', 'tree branches' )
    
    jetvariables = branches.allvars2017
    
    filename = infile_
    ieve = 0;icount = 0
    for df in read_root(filename, columns=jetvariables, chunksize=125000):

        st_runId                  = numpy.zeros(1, dtype=int)
        st_lumiSection            = array( 'L', [ 0 ] )
        st_eventId                = array( 'L', [ 0 ] )
        st_pfMetCorrPt            = array( 'f', [ 0. ] )
        st_pfMetCorrPhi           = array( 'f', [ 0. ] )
        st_pfMetUncJetResUp       = ROOT.std.vector('float')()
        st_pfMetUncJetResDown     = ROOT.std.vector('float')()
        st_pfMetUncJetEnUp        = ROOT.std.vector('float')()
        st_pfMetUncJetEnDown      = ROOT.std.vector('float')()
        st_isData                 = array( 'b', [ 0 ] )
        
        st_eleTrig                = array( 'b', [0] )
        st_muTrig                = array( 'b', [0] )
        st_metTrig                = array( 'b', [0] )
        st_phoTrig                = array( 'b', [0] )
        
        
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
        #st_muIso               = ROOT.std.vector('float')()

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

        #for trigs in triglist:
        #    exec("outTree.Branch( 'st_"+trigs+"', st_"+trigs+" , 'st_"+trigs+"/O')")

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
        #outTree.Branch( 'st_muIso', st_muIso)#, 'st_muIso/F')

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
        
        
        for run,lumi,event,isData,mcWeight_,\
        pu_nTrueInt_,pu_nPUVert_,\
        trigName_,trigResult_,filterName,filterResult,\
        met_,metphi_,metUnc_,\
        nele_,elepx_,elepy_,elepz_,elee_,elevetoid_, elelooseid_,eletightid_,\
        eleCharge_, npho_,phopx_,phopy_,phopz_,phoe_,pholooseid_,photightID_,\
        nmu_,mupx_,mupy_,mupz_,mue_,mulooseid_,mutightid_,muisoloose, muisomedium, muisotight, muisovtight, muCharge_,\
        nTau_,tau_px_,tau_py_,tau_pz_,tau_e_,tau_dm_,tau_isLoose_,\
        nGenPar_,genParId_,genMomParId_,genParSt_,genpx_,genpy_,genpz_,gene_,\
        nak4jet_,ak4px_,ak4py_,ak4pz_,ak4e_,\
        ak4TightID_,ak4deepcsv_,ak4flavor_,ak4NHEF_,ak4CHEF_,\
        ak4CEmEF_,ak4PhEF_,ak4EleEF_,ak4MuEF_, ak4JEC_, ak4NPV_,\
        fatnJet, fatjetPx, fatjetPy, fatjetPz, fatjetEnergy,fatjetTightID,\
        fatjet_DoubleSV, fatjet_probQCDb, fatjet_probHbb, fatjet_probQCDc, fatjet_probHcc, fatjet_probHbbc,\
        fatjet_prob_bbvsLight, fatjet_prob_ccvsLight, fatjet_prob_TvsQCD, fatjet_prob_WvsQCD, fatjet_prob_ZHbbvsQCD,\
        fatjetSDmass, fatN2_Beta1_, fatN2_Beta2_, fatjetCHSPRmassL2L3Corr, fatjetCHSSDmassL2L3Corr\
        in zip(df.runId,df.lumiSection,df.eventId,df.isData,df.mcWeight,\
                   df.pu_nTrueInt,df.pu_nPUVert,\
                   df.hlt_trigName,df.hlt_trigResult,df.hlt_filterName,df.hlt_filterResult,\
                   df.pfMetCorrPt,df.pfMetCorrPhi,df.pfMetCorrUnc,\
                   df.nEle,df.elePx,df.elePy,df.elePz,df.eleEnergy,df.eleIsPassVeto, df.eleIsPassLoose,df.eleIsPassTight,\
                   df.eleCharge,df.nPho,df.phoPx,df.phoPy,df.phoPz,df.phoEnergy,df.phoIsPassLoose,df.phoIsPassTight,\
                   df.nMu,df.muPx,df.muPy,df.muPz,df.muEnergy,df.isLooseMuon,df.isTightMuon,df.PFIsoLoose, df.PFIsoMedium, df.PFIsoTight, df.PFIsoVeryTight, df.muCharge,\
                   df.HPSTau_n,df.HPSTau_Px,df.HPSTau_Py,df.HPSTau_Pz,df.HPSTau_Energy,df.disc_decayModeFinding,df.disc_byLooseIsolationMVArun2017v2DBoldDMwLT2017,\
                   df.nGenPar,df.genParId,df.genMomParId,df.genParSt,df.genParPx,df.genParPy,df.genParPz,df.genParE,\
                   df.THINnJet,df.THINjetPx,df.THINjetPy,df.THINjetPz,df.THINjetEnergy,\
                   df.THINjetPassIDTight,df.THINjetDeepCSV_b,df.THINjetHadronFlavor,df.THINjetNHadEF,df.THINjetCHadEF,\
                   df.THINjetCEmEF,df.THINjetPhoEF,df.THINjetEleEF,df.THINjetMuoEF,df.THINjetCorrUncUp,df.THINjetNPV, \
                   df.FATnJet, df.FATjetPx, df.FATjetPy, df.FATjetPz, df.FATjetEnergy, df.FATjetPassIDTight,\
                   df.FATjet_DoubleSV, df.FATjet_probQCDb, df.FATjet_probHbb, df.FATjet_probQCDc, df.FATjet_probHcc, df.FATjet_probHbbc,\
                   df.FATjet_prob_bbvsLight, df.FATjet_prob_ccvsLight, df.FATjet_prob_TvsQCD, df.FATjet_prob_WvsQCD, df.FATjet_prob_ZHbbvsQCD,\
                   df.FATjetSDmass, df.FATN2_Beta1_, df.FATN2_Beta2_, df.FATjetCHSPRmassL2L3Corr, df.FATjetCHSSDmassL2L3Corr               ):
            

            if ieve%1000==0: print "Processed",ieve,"Events"
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
            
            eletrigdecision=False 
            mudecision=False 
            metdecision=False 
            phodecision=False 
            
            eletrigstatus = [( anautil.CheckFilter(trigName_, trigResult_, trig.Electrontrigger2017[itrig] ) ) for itrig in range(len(trig.Electrontrigger2017))]
            mutrigstatus  = [( anautil.CheckFilter(trigName_, trigResult_, trig.Muontrigger2017[itrig]     ) ) for itrig in range(len(trig.Muontrigger2017))    ]
            mettrigstatus = [( anautil.CheckFilter(trigName_, trigResult_, trig.METtrigger2017[itrig]       ) ) for itrig in range(len(trig.METtrigger2017))     ]
            photrigstatus = [( anautil.CheckFilter(trigName_, trigResult_, trig.Photontrigger2017[itrig]   ) ) for itrig in range(len(trig.Photontrigger2017))  ]
            
            eletrigdecision = boolutil.logical_OR(eletrigstatus)
            mutrigdecision  = boolutil.logical_OR(mutrigstatus)
            mettrigdecision = boolutil.logical_OR(mettrigstatus)
            photrigdecision = boolutil.logical_OR(photrigstatus)
            
            if not isData:
                eletrigdecision = True
                mutrigdecision = True
                mettrigdecision = True
                photrigdecision = True
                
        
            # ------------------------------------------------------
            ## Filter selection
            # ------------------------------------------------------
            filterdecision=False
            filterstatus = [False for ifilter in range(len(filters.filters2017)) ]
            filterstatus = [anautil.CheckFilter(filterName, filterResult, filters.filters2017[ifilter]) for ifilter in range(len(filters.filters2017)) ]
            
            
            if not isData:     filterdecision = True
            if isData:         filterdecision  = boolutil.logical_OR(filterstatus)
            
            if filterdecision == False: continue

            

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
            
            pho_pt15_eta2p5_looseID = [ (phopt[ip] > 15.0) and (abs(phoeta[ip]) < 2.5) and (pholooseid_[ip])               for ip in range(npho_)]
            pass_pho_index = boolutil.WhereIsTrue(pho_pt15_eta2p5_looseID, 1)
            
                        
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
            
            ele_pt10_eta2p5_vetoID   = [(elept[ie] > 10.0) and (elevetoid_[ie])  and (((abs(eleeta[ie]) > 1.566 or abs(eleeta[ie]) < 1.4442) and (abs(eleeta[ie]) < 2.5))) for ie in range(nele_)]
            ele_pt10_eta2p5_looseID  = [(elept[ie] > 10.0) and (elelooseid_[ie]) and (((abs(eleeta[ie]) > 1.566 or abs(eleeta[ie]) < 1.4442) and (abs(eleeta[ie]) < 2.5))) for ie in range(nele_)]
            ele_pt10_eta2p5_tightID  = [(elept[ie] > 30.0) and (eletightid_[ie]) and (((abs(eleeta[ie]) > 1.566 or abs(eleeta[ie]) < 1.4442) and (abs(eleeta[ie]) < 2.5))) for ie in range(nele_)]
            
            pass_ele_veto_index      = boolutil.WhereIsTrue(ele_pt10_eta2p5_vetoID, 1)
            
            
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
            
            mu_pt10_eta2p4_looseID_looseISO  = [ ( (mupt[imu] > 10.0) and (abs(mueta[imu])) and (mulooseid_[imu])  and (muisoloose[imu]) )  for imu in range(nmu_) ]
            mu_pt30_eta2p4_tightID_tightISO  = [ ( (mupt[imu] > 10.0) and (abs(mueta[imu])) and (mutightid_[imu])  and (muisotight[imu]) )  for imu in range(nmu_) ]
            
            pass_mu_index = boolutil.WhereIsTrue(mu_pt10_eta2p4_looseID_looseISO, 1)


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

            ak4_pt30_eta4p5_IDT  = [ ( (ak4pt[ij] > 30.0) and (abs(ak4eta[ij]) < 4.5) and (ak4TightID_[ij] ) ) for ij in range(nak4jet_)]
            
            ##--- jet cleaning 
            jetCleanAgainstEle = []
            jetCleanAgainstMu = []
            pass_jet_index_cleaned = []
            
            
            if len(ak4_pt30_eta4p5_IDT) > 0:
                DRCut = 0.4
                jetCleanAgainstEle = anautil.jetcleaning(ak4_pt30_eta4p5_IDT, ele_pt10_eta2p5_vetoID, ak4eta, eleeta, ak4phi, elephi, DRCut)
                jetCleanAgainstMu  = anautil.jetcleaning(ak4_pt30_eta4p5_IDT, mu_pt10_eta2p4_looseID_looseISO, ak4eta, mueta, ak4phi, muphi, DRCut)
                
                jetCleaned = boolutil.logical_AND_List2(jetCleanAgainstEle, jetCleanAgainstMu)
                pass_jet_index_cleaned = boolutil.WhereIsTrue(jetCleaned, 3)
                if debug_:print "pass_jet_index_cleaned = ", pass_jet_index_cleaned,"nJets= ",len(ak4px_)



            '''
            ******      *******   *****   *******
            *              *      *          *
            *****  ----    *      ****       *
            *              *      *          *
            *           ***       *****      *

            
            '''
            fatjetpt = [getPt(fatjetPx[ij], fatjetPy[ij]) for ij in range(fatnJet)]
            fatjeteta = [getEta(fatjetPx[ij], fatjetPy[ij], fatjetPz[ij]) for ij in range(fatnJet)]
            fatjetphi = [getPhi(fatjetPx[ij], fatjetPy[ij]) for ij in range(fatnJet)]
            
            fatjet_pt200_eta2p5_IDT  = [ ( (fatjetpt[ij] > 200.0) and (abs(fatjeteta[ij]) < 2.5) and (fatjetTightID[ij] ) ) for ij in range(fatnJet)]
            
            ##--- fat jet cleaning 
            fatjetCleanAgainstEle = []
            fatjetCleanAgainstMu = []
            pass_fatjet_index_cleaned = []
            
            
            if len(fatjet_pt200_eta2p5_IDT) > 0:
                fatjetCleanAgainstEle = anautil.jetcleaning(fatjet_pt200_eta2p5_IDT, ele_pt10_eta2p5_vetoID, fatjeteta, eleeta, fatjetphi, elephi, DRCut)
                fatjetCleanAgainstMu  = anautil.jetcleaning(fatjet_pt200_eta2p5_IDT, mu_pt10_eta2p4_looseID_looseISO, fatjeteta, mueta, fatjetphi, muphi, DRCut)
                
                fatjetCleaned = boolutil.logical_AND_List2(fatjetCleanAgainstEle, fatjetCleanAgainstMu)
                pass_fatjet_index_cleaned = boolutil.WhereIsTrue(fatjetCleaned, 3)
                if debug_:print "pass_fatjet_index_cleaned = ", pass_fatjet_index_cleaned," nJets =   ",len(fatjetpx)
                
            
            
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

            tau_eta2p3_iDLdm_pt18 = [ ( (taupt[itau] > 18.0) and (abs(taueta[itau]) < 2.3) and (tau_isLoose_[itau]) and (tau_dm_[itau]) ) for itau in range(nTau_)]
            if debug_:print "tau_eta2p3_iDLdm_pt18 = ", tau_eta2p3_iDLdm_pt18

            tauCleanAgainstEle = []
            tauCleanAgainstMu = []
            pass_tau_index_cleaned = []
            if len(tau_pt18_eta2p3)>0:
                tauCleanAgainstEle = jetcleaning(tau_eta2p3_iDLdm_pt18, ele_pt10_eta2p5_looseID,         taueta, eleeta, tauphi, elephi, DRCut)
                tauCleanAgainstMu  = jetcleaning(tau_eta2p3_iDLdm_pt18, mu_pt10_eta2p4_looseID_looseISO, taueta, mueta,  tauphi, muphi,  DRCut)
                tauCleaned = boolutil.logical_AND_List2(tauCleanAgainstEle, tauCleanAgainstMu)
                pass_tau_index_cleaned = boolutil.WhereIsTrue(tauCleaned,3)
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
            #st_muIso.clear()

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

            st_nEle[0] = len(pass_ele_veto_index)
            for iele in pass_ele_veto_index:
                st_elePx.push_back(elepx_[iele])
                st_elePy.push_back(elepy_[iele])
                st_elePz.push_back(elepz_[iele])
                st_eleEnergy.push_back(elee_[iele])
                st_eleIsPassTight.push_back(bool(eletightid_[iele]))
            if debug_:print 'nEle: ',len(pass_ele_veto_index)

            st_nMu[0] = len(pass_mu_index)
            for imu in pass_mu_index:
                st_muPx.push_back(mupx_[imu])
                st_muPy.push_back(mupy_[imu])
                st_muPz.push_back(mupz_[imu])
                st_muEnergy.push_back(mue_[imu])
                st_isTightMuon.push_back(bool(mutightid_[imu]))
                #st_muIso.push_back(muIso_[imu])
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
            if len(pass_ele_veto_index) == 2:
                iele1=pass_ele_veto_index[0]
                iele2=pass_ele_veto_index[1]
                if eleCharge_[iele1]*eleCharge_[iele2]<0:
                    ee_mass = InvMass(elepx_[iele1],elepy_[iele1],elepz_[iele1],elee_[iele1],elepx_[iele2],elepy_[iele2],elepz_[iele2],elee_[iele2])
                    zeeRecoilPx = -( met_*math.cos(metphi_) + elepx_[iele1] + elepx_[iele2])
                    zeeRecoilPy = -( met_*math.sin(metphi_) + elepy_[iele1] + elepy_[iele2])
                    ZeeRecoilPt =  math.sqrt(zeeRecoilPx**2  +  zeeRecoilPy**2)
                    if ee_mass > 70.0 and ee_mass < 110.0 and ZeeRecoilPt > 200.:
                        ZeeRecoil[0] = ZeeRecoilPt
                        ZeeMass[0] = ee_mass
                        ZeePhi[0] = mathutil.ep_arctan(zeeRecoilPx,zeeRecoilPy)
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
                        ZmumuPhi[0] = mathutil.ep_arctan(zmumuRecoilPx,zmumuRecoilPy)
            if len(pass_ele_veto_index) == 2:
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
            if len(pass_ele_veto_index) == 1:
               ele1 = pass_ele_veto_index[0]
               e_mass = MT(elept[ele1],met_, DeltaPhi(elephi[ele1],metphi_)) #transverse mass defined as sqrt{2pT*MET*(1-cos(dphi)}
               WenuRecoilPx = -( met_*math.cos(metphi_) + elepx_[ele1])
               WenuRecoilPy = -( met_*math.sin(metphi_) + elepy_[ele1])
               WenuRecoilPt = math.sqrt(WenuRecoilPx**2  +  WenuRecoilPy**2)
               if WenuRecoilPt > 200.:
                   WenuRecoil[0] = WenuRecoilPt
                   Wenumass[0] = e_mass
                   WenuPhi[0] = mathutil.ep_arctan(WenuRecoilPx,WenuRecoilPy)
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
                   WmunuPhi[0] = mathutil.ep_arctan(WmunuRecoilPx,WmunuRecoilPy)
            if len(pass_ele_veto_index) == 1:
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
                   GammaPhi[0] = mathutil.ep_arctan(GammaRecoilPx,GammaRecoilPy)
            GammaRecoilStatus = (GammaRecoil[0] > 200)
            if debug_: print 'Reached Gamma CR'

            if pfmetstatus==False and ZRecoilstatus==False and WRecoilstatus==False and GammaRecoilStatus==False: continue
            outTree.Fill()
        outfile.cd()
        h_total_mcweight.Write()
        h_total.Write()
        outfile.Write()

    end = time.clock()
    print "%.4gs" % (end-start)

#files=["/eos/cms//store/group/phys_exotica/bbMET/ExoPieElementTuples/MC_2017miniaodV2_06082019/DYJetsToLL_M-50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL_M_50_HT_400to600_TuneCP5_13TeV_30K/190808_201541/0000/ExoPieElementTuples_232.root"]
files=["/tmp/khurana/Merged_DYJets.root"]

if __name__ == '__main__':
    runbbdm(files[0])
    '''
    try:
        pool = mp.Pool(2)
        pool.map(runbbdm, files)
        pool.close()
    except Exception as e:
        print traceback.format_exc()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        pass
    '''
