from ROOT import TFile, TTree, TH1F, TH1D, TH1, TCanvas, TChain, TGraph, TMultiGraph, TGraphAsymmErrors, TMath, TH2D, TLorentzVector, AddressOf, gROOT, TNamed, gStyle, TF1
import ROOT as ROOT
import os
import sys, optparse
from array import array
import math
import numpy as numpy_

def Analyze():

	path = "/Users/user/Work/MonoHiggs/BJetRegression/"

	filename=["EXO-ggToXdXdHToBB_sinp_0p35_tanb_1p0_mXd_10_MH3_300_MH4_150_MH2_300_MHC_300_CP3Tune_13TeV_0000.root","EXO-ggToXdXdHToBB_sinp_0p35_tanb_1p0_mXd_10_MH3_400_MH4_150_MH2_400_MHC_400_CP3Tune_13TeV_0000.root","EXO-ggToXdXdHToBB_sinp_0p35_tanb_1p0_mXd_10_MH3_500_MH4_150_MH2_500_MHC_500_CP3Tune_13TeV_0000.root","EXO-ggToXdXdHToBB_sinp_0p35_tanb_1p0_mXd_10_MH3_600_MH4_150_MH2_600_MHC_600_CP3Tune_13TeV_0000.root", "EXO-ggToXdXdHToBB_sinp_0p35_tanb_1p0_mXd_10_MH3_1000_MH4_150_MH2_1000_MHC_1000_CP3Tune_13TeV_0000.root","EXO-ggToXdXdHToBB_sinp_0p35_tanb_1p0_mXd_10_MH3_1200_MH4_150_MH2_1200_MHC_1200_CP3Tune_13TeV_0000.root","EXO-ggToXdXdHToBB_sinp_0p35_tanb_1p0_mXd_10_MH3_1400_MH4_150_MH2_1400_MHC_1400_CP3Tune_13TeV_0000.root","EXO-ggToXdXdHToBB_sinp_0p35_tanb_1p0_mXd_10_MH3_1600_MH4_150_MH2_1600_MHC_1600_CP3Tune_13TeV_0000.root"]

	sigmass=array('d',[300.,400.,500.,600.,1000.,1200.,1400.,1600.])
	h_Mbb=[]
	h_regMbb=[]
	h_higgs_pT=[]
	h_reghiggs_pT=[]
	for i in range (len(sigmass)):
		namex= 'Mbb_'+str(i)
		titlex= 'hist_Mbb_with_range_'+str(sigmass[i])
		mbbx= TH1F(namex,  titlex,  25,100.,150.)
 		h_Mbb.append(mbbx)

		namey= 'regMbb_'+str(i)
		titley= 'hist_regMbb_with_range_'+str(sigmass[i])
		mbby= TH1F(namey,  titley,  25,100.,150.)
		h_regMbb.append(mbby)

		namexx= 'higgspt_'+str(i)
		titlexx= 'hist_pt_with_range_'+str(sigmass[i])
		h_higgs_pT_x= TH1F(namexx,  titlexx, 50,0.0,1500.)
		h_higgs_pT.append(h_higgs_pT_x)

		nameyy= 'reg_higgspt_'+str(i)
		titleyy= 'reg_hist_pt_with_range_'+str(sigmass[i])
		h_reghiggs_pT_x= TH1F(nameyy,  titleyy, 50,0.0,1500.)
		h_reghiggs_pT.append(h_reghiggs_pT_x)

	for i in range(len(filename)):
		openf = TFile(path+filename[i], "read")
		skimmedTree = openf.Get("outTree")
		NEntries = skimmedTree.GetEntries()
		print filename[i], NEntries

		for ievent in range(NEntries):
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
				j2p4   = myBjetsP4[1]
				addJet=(j1p4+j2p4)
				higgs_pt=addJet.Pt()
	    	#print "pt higgs", addJet.Pt() 
				if(addJet.M() > 100 and addJet.M()<150):
					h_higgs_pT[i].Fill(higgs_pt)
					h_Mbb[i].Fill(addJet.M())
				j1p4Corr=jetP4Corr[0]
				j2p4Corr=jetP4Corr[1]
				addJetCorr=(j1p4Corr+j2p4Corr)
				Corrhiggs_pt=addJetCorr.Pt()
				if(addJetCorr.M() > 100 and addJetCorr.M()<150):
					h_reghiggs_pT[i].Fill(Corrhiggs_pt)
					h_regMbb[i].Fill(addJetCorr.M())

	effi_Mbb, effi_regMbb = array( 'd' ), array( 'd' )
	for i in range (len(sigmass)):
		norm_h_Mbb =0.
		norm_h_Mbb = h_Mbb[i].Integral();
		#effi_Mbb[i]=norm_h_Mbb/NEntries	
		effi_Mbb.append(norm_h_Mbb/NEntries)
		print "effi " 
		print(' i %i %f %f ' % (i,sigmass[i],effi_Mbb[i]))
		norm_h_regMbb =0.
		norm_h_regMbb = h_regMbb[i].Integral()
		effi_regMbb.append(norm_h_regMbb/NEntries)
		print "effi after ", effi_regMbb[i]

	c1 = TCanvas( 'c1', 'Significance', 200, 10, 700, 500 )
	leg = ROOT.TLegend(.73,.32,.97,.53)
	leg.SetBorderSize(0)
	leg.SetFillColor(0)
	leg.SetFillStyle(0)
	leg.SetTextFont(42)
	leg.SetTextSize(0.035)

	n=len(sigmass)
	gr1 = TGraph( n, sigmass, effi_Mbb )
	gr1.SetLineColor( 2 )
	gr1.SetLineWidth( 4 )
	gr1.SetMarkerColor( 4 )
	gr1.SetMarkerStyle( 21 )
	gr1.SetTitle("Before regression")
	#gr1.SetDrawOption("AP");
	#gr1.SetTitle( 'Punzi Significance' )
	#gr1.GetXaxis().SetTitle( 'Signal Mass Point' )
	#gr1.GetYaxis().SetTitle( 'Punzi' )

	gr2 = TGraph( n, sigmass, effi_regMbb )
	gr2.SetLineColor( 3 )
	gr2.SetLineWidth( 4 )
	gr2.SetMarkerColor( 3 )
	gr2.SetMarkerStyle( 24 )
	gr2.SetTitle("After regression")
	#gr2.SetDrawOption("AP")
	#gr1.Draw("ACP")
	#gr2.Draw("ACP")

	mg = TMultiGraph('mg', 'mg')
	mg.Add(gr1, "LP")
	mg.Add( gr2,"LP")
	mg.SetTitle("Punzi Significance with Signal mass Point; Signal Mass Point; Punzi Significance");
	mg.Draw("ACP")
	c1.BuildLegend()
	c1.SaveAs("sig.pdf")


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
