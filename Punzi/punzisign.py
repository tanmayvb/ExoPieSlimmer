from ROOT import TFile, TTree, TH1F, TH1D, TH1, TCanvas, TChain, TGraph, TMultiGraph, TGraphAsymmErrors, TMath, TH2D, TLorentzVector, AddressOf, gROOT, TNamed, gStyle, TF1
import ROOT as ROOT
import os
import sys, optparse
from array import array
import math
import numpy as numpy_
import sample_xsec_2017 as crsec 

def Analyze():

	path = "/eos/cms/store/group/phys_exotica/monoHiggs/monoHbb/skimmedFiles/V0_fixedjetID_bjetReg/"
	outfilename= 'Outputplot.root'
	outfile = TFile(outfilename,'RECREATE')
	txtfile = open("copy.txt", "w")
	filename = list()        
	with open ("sigfile.txt", "r") as myfile:
		for line in myfile:
			filename.append(line.strip()) 

	sigmass=array('d',[300.,400.,500.,600.,1000.,1200.,1400.,1600.])
	signal_entry=TH1F('signal_entry', 'signal_entry_eachbin', len(filename),0,len(filename))
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

	Entry=[]
	for i in range(len(filename)):
		openf = TFile(path+filename[i], "read")
		skimmedTree = openf.Get("outTree")
		NEntries = skimmedTree.GetEntries()
		print filename[i], NEntries
		Entry.append(NEntries)

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
	outfile.cd()
	for i in range(len(filename)):
		h_Mbb[i].Write()
		h_higgs_pT[i].Write()
		h_regMbb[i].Write()
		h_reghiggs_pT[i].Write()
		signal_entry.SetBinContent(i,Entry[i])
	signal_entry.Write()
# BackGround Samples and selections
	bkgfilename = list()
	with open ("bkgfile.txt", "r") as mybkgfile:
		for line in mybkgfile:
			bkgfilename.append(line.strip())

	L=41000.0

	h_bkgMbb= []
	h_bkgregMbb= []

	h_bkgMbbx=[]
        h_bkgregMbbx=[]
        for i in range (len(bkgfilename)):
		name1='Mbb_bkg_with_mass_window_'+str(i)
                title1= 'hist_bkgMbb_with_mass_window_'+str(bkgfilename[i].replace('.root','_file'))
		mbb1= TH1F(name1, title1, 25,100.,150.)
                h_bkgMbb.append(mbb1)

                namex= 'Mbb_bkg_without_mass_window_'+str(i)
                titlex= 'hist_bkgMbb_without_mass_window'+str(bkgfilename[i].replace('.root','_file'))
                mbbxx= TH1F(namex,  titlex,  120,20.,1000.)
                h_bkgMbbx.append(mbbxx)

		name2='regMbb_bkg_with_mass_window_'+str(i)
                title2= 'hist_bkgMbb_with_mass_window_'+str(bkgfilename[i].replace('.root','_file'))
                mbb2= TH1F(name2, title2, 25,100.,150.)
                h_bkgregMbb.append(mbb2)

                namey= 'regMbb_bkg_without_mass_window_'+str(i)
                titley= 'hist_bkgregMbb_without_mass_window_'+str(bkgfilename[i].replace('.root','_file'))
                mbbyy= TH1F(namey,  titley,  120,20.,1000.)
                h_bkgregMbbx.append(mbbyy)

	#h_bkgMbbx= TH1F('h_bkgmbbx',  'Mbb beforex',  50,0.,1500.)
	#h_bkgregMbbx= TH1F('h_bkgregMbbx', 'Mbb after x',  50,0.,1500.)

	Count_bkgregMbb=Count_bkgMbb=0
	for i in range(len(bkgfilename)):
		bkgopenf = TFile(path+bkgfilename[i], "read")

		h_total_mcweight_bkg = bkgopenf.Get("h_total_mcweight")
		h_total_mcweight_bkgx=h_total_mcweight_bkg.Clone()
		weight_name=h_total_mcweight_bkg.GetName()
		weight_name1=weight_name+'_'+str(i)
		h_total_mcweight_bkgx.SetName(weight_name1)
		weight_title=bkgfilename[i].replace('.root','_file')
		h_total_mcweight_bkgx.SetTitle(weight_title)
		outfile.cd()
		h_total_mcweight_bkgx.Write()
		eventsbkg = h_total_mcweight_bkg.Integral()

		bkgskimmedTree = bkgopenf.Get("outTree")
		bkgNEntries = bkgskimmedTree.GetEntries()
		print bkgfilename[i], eventsbkg, ' ', bkgNEntries
		print "Next Check \n"
		for ievent in range(bkgNEntries):
			bkgskimmedTree.GetEntry(ievent)
		
			nTHINJets              = bkgskimmedTree.__getattr__('st_THINnJet')
			THINJetsPx             = bkgskimmedTree.__getattr__('st_THINjetPx')
			THINJetsPy             = bkgskimmedTree.__getattr__('st_THINjetPy')
			THINJetsPz             = bkgskimmedTree.__getattr__('st_THINjetPz')
			THINJetsEnergy         = bkgskimmedTree.__getattr__('st_THINjetEnergy')
			thinbRegNNCorr         = bkgskimmedTree.__getattr__('st_THINbRegNNCorr')
			thinbRegNNResolution   = bkgskimmedTree.__getattr__('st_THINbRegNNResolution')
			thinjetDeepCSV         = bkgskimmedTree.__getattr__('st_THINjetDeepCSV')

			nfJets              = bkgskimmedTree.__getattr__('st_nfjet')
			fJetsPx             = bkgskimmedTree.__getattr__('st_fjetPx')
			fJetsPy             = bkgskimmedTree.__getattr__('st_fjetPy')
			fJetsPz             = bkgskimmedTree.__getattr__('st_fjetPz')
			fJetsEnergy         = bkgskimmedTree.__getattr__('st_fjetEnergy')

			nEle                   = bkgskimmedTree.__getattr__('st_nEle') 
			ElePx                  = bkgskimmedTree.__getattr__('st_elePx') 
			ElePy                  = bkgskimmedTree.__getattr__('st_elePy') 
			ElePz                  = bkgskimmedTree.__getattr__('st_elePz') 
			EleEnergy              = bkgskimmedTree.__getattr__('st_eleEnergy') 
			eleIsPassLoose         = bkgskimmedTree.__getattr__('st_eleIsPassLoose')
			eleIsPassTight         = bkgskimmedTree.__getattr__('st_eleIsPassTight')

        
			nMu                    = bkgskimmedTree.__getattr__('st_nMu') 
			muPx                   = bkgskimmedTree.__getattr__('st_muPx') 
			muPy                   = bkgskimmedTree.__getattr__('st_muPy') 
			muPz                   = bkgskimmedTree.__getattr__('st_muPz') 
			muEnergy               = bkgskimmedTree.__getattr__('st_muEnergy') 
			isTightMuon            = bkgskimmedTree.__getattr__('st_isTightMuon')

			nPho                    = bkgskimmedTree.__getattr__('st_nPho') 
			phoPx                   = bkgskimmedTree.__getattr__('st_phoPx') 
			phoPy                   = bkgskimmedTree.__getattr__('st_phoPy') 
			phoPz                   = bkgskimmedTree.__getattr__('st_phoPz') 
			phoEnergy               = bkgskimmedTree.__getattr__('st_phoEnergy') 
			phoIsPassTight          = bkgskimmedTree.__getattr__('st_phoIsPassTight')

			pfMet                   = bkgskimmedTree.__getattr__('st_pfMetCorrPt')
			pfMetPhi                = bkgskimmedTree.__getattr__('st_pfMetCorrPhi')

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
				#print "pt higgs", addJet.Pt(), 'Mass =', addJet.M()
				if(addJet.M() > 100 and addJet.M()<150):
					h_bkgMbb[i].Fill(addJet.M())
				h_bkgMbbx[i].Fill(addJet.M())
				j1p4Corr=jetP4Corr[0]
				j2p4Corr=jetP4Corr[1]
				addJetCorr=(j1p4Corr+j2p4Corr)
				Corrhiggs_pt=addJetCorr.Pt()
				if(addJetCorr.M() > 100 and addJetCorr.M()<150):
					h_bkgregMbb[i].Fill(addJetCorr.M())
				h_bkgregMbbx[i].Fill(addJetCorr.M())

		#xsBkg = crsec.getXsec(bkgfilename[i])
		txtfile.write(bkgfilename[i]+" "+str(h_bkgMbb[i].Integral()))
		txtfile.write(" After ")
		txtfile.write(str(h_bkgregMbb[i].Integral())+"\n")
		#print 'Before scale \n'
		#print  h_bkgMbb.Integral(), ' ', h_bkgregMbb.Integral() 
		#scalefact=(41000*xsBkg)/eventsbkg
		#print 'xsec= ', xsBkg, ' Nevt= ', eventsbkg, ' scalefact = ', scalefact
		#h_bkgMbb.Scale(scalefact)
		#h_bkgMbbx[i].Scale(scalefact)
		#h_bkgMbb = h_bkgMbb*(L*xsBkg/eventsbkg)
		#h_bkgMbbx[i]=h_bkgMbbx[i]*(L*xsBkg/eventsbkg)

		#h_bkgregMbb = h_bkgregMbb*(L*xsBkg/eventsbkg)
		#h_bkgregMbbx[i] = h_bkgregMbbx[i]*(L*xsBkg/eventsbkg)	

		#Count_bkgMbb+=h_bkgMbb.Integral()
		#Count_bkgregMbb+=h_bkgregMbb.Integral()
		#h_bkgMbb.Reset()
		#h_bkgregMbb.Reset()
	for i in range(len(bkgfilename)):
		outfile.cd()
		h_bkgMbb[i].Write()	
        	h_bkgMbbx[i].Write()
		h_bkgregMbb[i].Write()
        	h_bkgregMbbx[i].Write()
	outfile.Close()
        txtfile.close()
	'''	
	txtfile.write("\n  PunZi Significance \n")
# Calculate and plot Punzi significance
	punzi_Mbb, punzi_regMbb = array( 'd' ), array( 'd' )
	for i in range (len(sigmass)):
		norm_h_Mbb =0.
		norm_h_Mbb = h_Mbb[i].Integral();
		#punzi_Mbb[i]=norm_h_Mbb/NEntries	
		effi_Mbb=norm_h_Mbb/Entry[i]
		punzi_Mbb.append(effi_Mbb/(1+TMath.Sqrt(Count_bkgMbb)))
		txtfile.write(str(effi_Mbb)+" Total Count Before "+str(Count_bkgMbb)+"\n")
		#print "effi " 
		#print(' i %i %f %f ' % (i,sigmass[i],punzi_Mbb[i]))
		norm_h_regMbb =0.
		norm_h_regMbb = h_regMbb[i].Integral()
		effi_regMbb = norm_h_regMbb/Entry[i]
		punzi_regMbb.append(effi_regMbb/(1+TMath.Sqrt(Count_bkgregMbb)))
		txtfile.write(str(effi_regMbb)+" Total Count after "+str(Count_bkgregMbb)+"\n")
		h_Mbb[i].Write()
                h_regMbb[i].Write()
		#print "effi after ", punzi_regMbb[i]

	c1 = TCanvas( 'c1', 'Significance', 200, 10, 700, 500 )
	leg = ROOT.TLegend(.73,.32,.97,.53)
	leg.SetBorderSize(0)
	leg.SetFillColor(0)
	leg.SetFillStyle(0)
	leg.SetTextFont(42)
	leg.SetTextSize(0.035)
	n=len(sigmass)
	gr1 = TGraph( n, sigmass, punzi_Mbb )
	gr1.SetLineColor( 2 )
	gr1.SetLineWidth( 4 )
	gr1.SetMarkerColor( 4 )
	gr1.SetMarkerStyle( 21 )
	gr1.SetTitle("Before regression")
	gr1.SetName("graph1")
	#gr1.SetDrawOption("AP");
	#gr1.SetTitle( 'Punzi Significance' )
	#gr1.GetXaxis().SetTitle( 'Signal Mass Point' )
	#gr1.GetYaxis().SetTitle( 'Punzi' )
	gr1.Write()

	gr2 = TGraph( n, sigmass, punzi_regMbb )
	gr2.SetLineColor( 3 )
	gr2.SetLineWidth( 4 )
	gr2.SetMarkerColor( 3 )
	gr2.SetMarkerStyle( 24 )
	gr2.SetTitle("After regression")
	gr2.SetName("graph2")
	#gr2.SetDrawOption("AP")
	#gr1.Draw("ACP")
	#gr2.Draw("ACP")
	gr2.Write()

	mg = TMultiGraph('mg', 'mg')
	mg.Add(gr1, "LP")
	mg.Add( gr2,"LP")
	mg.SetTitle("  ; Signal Mass Point; Punzi Significance");
	mg.Draw("ACP")
	c1.BuildLegend()
	c1.SaveAs("sig.pdf")
	
	outfile.Close()
	txtfile.close()
	'''
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
