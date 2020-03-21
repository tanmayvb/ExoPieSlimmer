from ROOT import TFile, TTree, TH1F, TH1D, TH1, TCanvas, TChain,TGraphAsymmErrors, TMath, TH2D, TLorentzVector, AddressOf, gROOT, TNamed, gStyle, TF1
import ROOT as ROOT
import os
import sys, optparse
from array import array
import math
import numpy as numpy_

#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooGaussian.h"
#include "TCanvas.h"
#include "RooPlot.h"
#include "TTree.h"
#include "TH1D.h"
#include "TRandom.h"
#ROOT.gROOT.LoadMacro("Loader.h+")

skimmedTree = TChain("outTree")
skimmedTree.Add(sys.argv[1])

outfilename= 'Outputplot.root'
outfile = TFile(outfilename,'RECREATE')

def SetCanvas():

    # CMS inputs
    # -------------
    H_ref = 1000;
    W_ref = 1000;
    W = W_ref
    H  = H_ref

    T = 0.08*H_ref
    B = 0.21*H_ref
    L = 0.12*W_ref
    R = 0.08*W_ref
    # --------------

    c1 = TCanvas("c2","c2",0,0,2000,1500)
    c1.SetFillColor(0)
    c1.SetBorderMode(0)
    c1.SetFrameFillStyle(0)
    c1.SetFrameBorderMode(0)
    c1.SetLeftMargin( L/W )
    c1.SetRightMargin( R/W )
    c1.SetTopMargin( T/H )
    c1.SetBottomMargin( B/H )
    c1.SetTickx(0)
    c1.SetTicky(0)
    c1.SetTickx(1)
    c1.SetTicky(1)
    c1.SetGridy()
    c1.SetGridx()
    #c1.SetLogy(1)
    return c1
def CreateLegend(x1, y1, x2, y2, header):

    leg = ROOT.TLegend(x1, y1, x2, y2)
    leg.SetFillColor(0)
    leg.SetFillStyle(3002)
    leg.SetBorderSize(0)
    leg.SetHeader(header)
    return leg

gStyle.SetOptTitle(0)
gStyle.SetOptStat(0)
gStyle.SetErrorX(0.)
ROOT.gROOT.SetBatch(True)

def Analyze():



    DCSVMWP=0.6324
    NEntries = skimmedTree.GetEntries()

    h_Mbb                     =TH1F('h_Mbb_',  'h_Mbb_',  25,100.,150.)
    h_regMbb                  =TH1F('h_regMbb_',  'h_regMbb_',  25,100.,150.)

    h_jet1_pT                 =TH1F('h_jet1_pT_',  'h_jet1_pT_',  50,0.0,1500.)
    h_jet2_pT                 =TH1F('h_jet2_pT',  'h_jet2_pT',  50,0.0,1500.)

    h_regjet1_pT              =TH1F('h_regjet1_pT_',  'h_regjet1_pT_',  50,0.0,1500.)
    h_regjet2_pT              =TH1F('h_regjet2_pT',  'h_regjet2_pT',  50,0.0,1500.)


    if len(sys.argv)>2:
        NEntries=int(sys.argv[2])
        print "WARNING: Running in TEST MODE"

    for ievent in range(NEntries):
        if ievent%100==0: print "Processed "+str(ievent)+" of "+str(NEntries)+" events."
        skimmedTree.GetEntry(ievent)


        nTHINJets              = skimmedTree.__getattr__('st_THINnJet')
        THINJetsPx             = skimmedTree.__getattr__('st_THINjetPx')
        THINJetsPy             = skimmedTree.__getattr__('st_THINjetPy')
        THINJetsPz             = skimmedTree.__getattr__('st_THINjetPz')
        THINJetsEnergy         = skimmedTree.__getattr__('st_THINjetEnergy')
        thinbRegNNCorr         = skimmedTree.__getattr__('st_THINbRegNNCorr')
        thinbRegNNResolution   = skimmedTree.__getattr__('st_THINbRegNNResolution')
        thinjetDeepCSV         =skimmedTree.__getattr__('st_THINjetDeepCSV')

        nfJets              = skimmedTree.__getattr__('st_nfjet')
        fJetsPx             = skimmedTree.__getattr__('st_fjetPx')
        fJetsPy             = skimmedTree.__getattr__('st_fjetPy')
        fJetsPz             = skimmedTree.__getattr__('st_fjetPz')
        fJetsEnergy         = skimmedTree.__getattr__('st_fjetEnergy')


        nEle                   = skimmedTree.__getattr__('st_nEle') 
        ElePx                  = skimmedTree.__getattr__('st_elePx') 
        ElePy                  = skimmedTree.__getattr__('st_elePy') 
        ElePz                  = skimmedTree.__getattr__('st_elePz') 
        EleEnergy              = skimmedTree.__getattr__('st_eleEnergy') 
        eleIsPassLoose         =skimmedTree.__getattr__('st_eleIsPassLoose')
        eleIsPassTight         =skimmedTree.__getattr__('st_eleIsPassTight')

        
        nMu                    = skimmedTree.__getattr__('st_nMu') 
        muPx                   = skimmedTree.__getattr__('st_muPx') 
        muPy                   = skimmedTree.__getattr__('st_muPy') 
        muPz                   = skimmedTree.__getattr__('st_muPz') 
        muEnergy               = skimmedTree.__getattr__('st_muEnergy') 
        isTightMuon         =skimmedTree.__getattr__('st_isTightMuon')



        nPho                    = skimmedTree.__getattr__('st_nPho') 
        phoPx                   = skimmedTree.__getattr__('st_phoPx') 
        phoPy                   = skimmedTree.__getattr__('st_phoPy') 
        phoPz                   = skimmedTree.__getattr__('st_phoPz') 
        phoEnergy               = skimmedTree.__getattr__('st_phoEnergy') 
        phoIsPassTight       =skimmedTree.__getattr__('st_phoIsPassTight')

        pfMet                      = skimmedTree.__getattr__('st_pfMetCorrPt')
        pfMetPhi                      = skimmedTree.__getattr__('st_pfMetCorrPhi')
#        passThinJetLooseID         = skimmedTree.__getattr__('THINjetPassIDLoose')
#        thinJetdeepCSV             = skimmedTree.__getattr__('AK4deepCSVjetDeepCSV_b')


   #--------------------------------------------------
   # Met Cut
        if(pfMet<200.):continue

   #--------------------------------------------------
   #Lepton Collection
        myEles=[]
        myElesP4=[]
        for iele in range(nEle):
            tempele = ROOT.TLorentzVector()
            tempele.SetPxPyPzE(ElePx[iele],ElePy[iele],ElePz[iele],EleEnergy[iele])
            if (bool(eleIsPassLoose[iele]) == False):continue
            if (tempele.Pt() < 10. ) or (abs(tempele.Eta()) > 2.5) or (abs(tempele.Eta())>1.44 and abs(tempele.Eta())<1.57):continue
            myEles.append(iele)
            myElesP4.append(tempele)

        myMuos = []
        myMuosP4 = []
        for imu in range(nMu):
            tempmu = ROOT.TLorentzVector()
            tempmu.SetPxPyPzE(muPx[imu],muPy[imu],muPz[imu],muEnergy[imu])
            if (tempmu.Pt()>10.) & (abs(tempmu.Eta()) < 2.4) & (bool(isTightMuon[imu]) == True):
                myMuos.append(imu)
                myMuosP4.append(tempmu)
    
        myPhos=[]
        myPhosP4=[]
        for ipho in range(nPho):
            temppho = ROOT.TLorentzVector()
            temppho.SetPxPyPzE(phoPx[ipho],phoPy[ipho],phoPz[ipho],phoEnergy[ipho])
            if (bool(phoIsPassTight[ipho]) == False): continue
            if (temppho.Pt() < 15.) or (abs(temppho.Eta()) > 2.5) : continue
            if isMatch(myMuosP4,temppho,0.4) or isMatch(myElesP4,temppho,0.4):continue
            myPhos.append(ipho)
            myPhosP4.append(temppho) 


   #--------------------------------------------------
        #thinAK4Jets
        thinjetpassindex=[]
        for ithinjet in range(nTHINJets):
            tempthinjet = ROOT.TLorentzVector()
            tempthinjet.SetPxPyPzE(THINJetsPx[ithinjet],THINJetsPy[ithinjet],THINJetsPz[ithinjet],THINJetsEnergy[ithinjet])
        #if (bool(passThinJetLooseID[ithinjet])==False):continue
            if (tempthinjet.Pt() < 30.0) or (abs(tempthinjet.Eta())>4.5):continue
            if isMatch(myMuosP4,tempthinjet,0.4) or isMatch(myElesP4,tempthinjet,0.4) or isMatch(myPhosP4,tempthinjet,0.4):continue
            thinjetpassindex.append(ithinjet)

    #---------------------------------------------------- 

        
    #--------------------------------------------------
        #thinAK8Jets
        fjetpassindex=[]
        for ifjet in range(nfJets):
            tempfjet = ROOT.TLorentzVector()
            tempfjet.SetPxPyPzE(fJetsPx[ifjet],fJetsPy[ifjet],fJetsPz[ifjet],fJetsEnergy[ifjet])
        #if (bool(passThinJetLooseID[ithinjet])==False):continue
            if (tempfjet.Pt() < 200.0) or (abs(tempfjet.Eta())>2.4):continue
            if isMatch(myMuosP4,tempfjet,0.8) or isMatch(myElesP4,tempfjet,0.8) or isMatch(myPhosP4,tempfjet,0.8):continue
            fjetpassindex.append(ifjet)

    #----------------------------------------------------
        jetP4Corr=[]
        myBjetsP4=[]
        for jthinjet in thinjetpassindex:
	    j1 = thinjetDeepCSV[jthinjet]
            #print 'CVS PT =', len(j1)
            tempjet = ROOT.TLorentzVector()
            px=THINJetsPx[jthinjet]
            py=THINJetsPy[jthinjet]
            pz=THINJetsPz[jthinjet]
            en=THINJetsEnergy[jthinjet]
            tempjet.SetPxPyPzE(px,py,pz,en)
            if (tempjet.Pt() > 30.0) and (abs(tempjet.Eta())<2.4) and (thinjetDeepCSV[jthinjet] >0.4941) and (abs(DeltaPhi(pfMetPhi, tempjet.Phi()))>0.4):
                NNCorr=thinbRegNNCorr[jthinjet]
                pt=math.sqrt(px*px+py*py)
                if NNCorr==0.0:
                    NNCorr=1.0	
                tempjetP4Corr =  ROOT.TLorentzVector()
                corpt=tempjet.Pt()
                coreta=tempjet.Eta()
                corphi=tempjet.Phi()
                tempjetP4Corr.SetPtEtaPhiE(pt*NNCorr,coreta,corphi,en*NNCorr)
                #tempjetP4Corr.SetPtEtaPhiE(pt*NNCorr,eta,phi,ene*NNCorr)

                #print "pT before correction:  ", pt, NNCorr
                #print "pT after correction:   ", corpt, coreta

                jetP4Corr.append(tempjetP4Corr)
                myBjetsP4.append(tempjet)
   
        #print len(myBjetsP4)
        if (len(myBjetsP4)==2):
            j1p4   = myBjetsP4[0]
            h_jet1_pT.Fill(j1p4.Pt())
            j2p4   = myBjetsP4[1]
            h_jet2_pT.Fill(j2p4.Pt())
            addJet=(j1p4+j2p4)
            if(addJet.M() > 100 and addJet.M()<150):
                h_Mbb.Fill(addJet.M())
            j1p4Corr=jetP4Corr[0]
            h_regjet1_pT.Fill(j1p4Corr.Pt())
            j2p4Corr=jetP4Corr[1]
            h_regjet2_pT.Fill(j2p4Corr.Pt())
            addJetCorr=(j1p4Corr+j2p4Corr)
            if(addJetCorr.M() > 100 and addJetCorr.M()<150):
                h_regMbb.Fill(addJetCorr.M())
      
    f1 = TF1("f1", "gaus",  100, 150);
    f2 = TF1("f2", "gaus",  100, 150);
    c      = SetCanvas()
    legend = CreateLegend(0.13, 0.64, 0.45, 0.92, "")
    legend.SetTextSize(0.03)

    h_Mbb.SetLineColor(2)
    h_Mbb.SetMarkerColor(2)
    h_Mbb.SetMarkerSize(1.5)
    h_Mbb.SetLineWidth(3)
    h_Mbb.SetMarkerStyle(23)
    h_Mbb.SetXTitle("M_{bb}")
    h_Mbb.SetTitle("")
    norm_h_Mbb = h_Mbb.Integral();
    print 'Entries after selection =', norm_h_Mbb

    h_Mbb.Scale(1/norm_h_Mbb)
    h_Mbb.Draw('P')
    h_Mbb.Fit("f1","","",100,150)
    f1.SetLineColor(2)
    f1.Draw('same')

    '''
    x=ROOT.RooRealVar ("x", "x", 30, 200)
    dh=ROOT.RooDataHist("dh", "dh", ROOT.RooArgList(x), ROOT.RooFit.Import(h_Mbb))
    frame=x.frame(ROOT.RooFit.Title("Imported TH1 with Poisson error bars"))
    dh.plotOn(frame)

    mean=ROOT.RooRealVar("mean", "mean", 100, 30, 200)
    sigma=ROOT.RooRealVar("sigma", "sigma", 80, 30, 200)
    gauss=ROOT.RooGaussian("gauss", "gauss", x, mean, sigma)
    gauss.fitTo(dh,ROOT.RooFit.Range(80,140))
    gauss.plotOn(frame)

    c = ROOT.TCanvas("rf102_dataimport", "rf102_dataimport", 800, 800)
    c.cd()
    ROOT.gPad.SetLeftMargin(0.15)
    frame.GetYaxis().SetTitleOffset(1.4)
    frame.Draw()
    '''

    h_regMbb.SetLineColor(3)
    h_regMbb.SetMarkerColor(3)
    h_regMbb.SetMarkerSize(1.5)
    h_regMbb.SetLineWidth(3)
    h_regMbb.SetMarkerStyle(22)
    h_regMbb.SetXTitle("Mbb")
    h_regMbb.SetTitle("")
    h_regMbb.GetEntries()
    norm_h_regMbb = h_regMbb.Integral();
    h_regMbb.Scale(1/norm_h_regMbb)
    h_regMbb.Draw('Same P')
    h_regMbb.Fit("f2","","",100,150)
    f2.SetLineColor(3)
    f2.Draw('same')

    f1_mu=int(f1.GetParameter(1))
    f1_sigma=int(f1.GetParameter(2))
    f2_mu=int(f2.GetParameter(1))
    f2_sigma=int(f2.GetParameter(2))

    f1_fitvalues = "#mu="+str(f1_mu)+ ', ' "#sigma="+str(f1_sigma)
    f2_fitvalues = "#mu="+str(f2_mu)+ ', ' "#sigma="+str(f2_sigma)

    mbb_entry="Baseline+selection"+"; Entries="+str(int(norm_h_Mbb))
    regmbb_entry="Baseline+selection+DNN"+"; Entries="+str(int(norm_h_regMbb))

    legend.AddEntry(h_Mbb,mbb_entry)
    legend.AddEntry(f1,f1_fitvalues)
    legend.AddEntry(h_regMbb,regmbb_entry)
    legend.AddEntry(f2,f2_fitvalues)

    legend.Draw()
 #   outfile.write(h_regMbb)
    c.SaveAs('Mbb_Mbbreg.pdf')
    c2 = TCanvas("c2","c2",0,0,2000,1500)
#    c2.Setlogy(1)
#    gPad.SetLogy()
    c2.Divide(2,2)
    c2.cd(1)
    ROOT.gPad.SetLogy()
    h_jet1_pT.Draw('P')
    c2.cd(2)
    h_jet2_pT.Draw('P')
    c2.cd(3)
    h_regjet1_pT.Draw('P')
    c2.cd(4)
    h_regjet2_pT.Draw('P')
    c2.Update()
    c2.SaveAs("pts.pdf")
def DeltaPhi(ph1, ph2):
    dphi=Phi_mpi_pi(ph1-ph2)
    return dphi

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

def isMatch(P4Coll, objP4, cut):
    match=False
    for ojb in range(len(P4Coll)):
	if DeltaR(P4Coll[ojb],objP4) < cut: match=True
        break
    return match
def Phi_mpi_pi(x):
    kPI = 3.14159265358979323846
    kTWOPI = 2 * kPI

    while (x >= kPI): x = x - kTWOPI;
    while (x < -kPI): x = x + kTWOPI;
    return x;
if __name__ == "__main__":
    Analyze()
