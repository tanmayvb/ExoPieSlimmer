from ROOT import TFile, TTree, TH1F, TH1D, TH1, TCanvas, TChain, TGraph, TMultiGraph, TGraphAsymmErrors, TMath, TH2D, TLorentzVector, AddressOf, gROOT, TNamed, gStyle, TF1, TLegend
import ROOT as ROOT
import os
import sys, optparse
from array import array
import math
import numpy as numpy_
import sample_xsec_2017 as crsec

gROOT.Reset()
gStyle.SetOptTitle(0)
gStyle.SetOptStat(0)
gStyle.SetErrorX(0.)
sigmass=array('d',[300.,400.,500.,600.,1000.,1200.,1400.,1600.])
def bkghist (hist_name):
	mbbxx= TH1F('mbbxx',  'add all mbbxx',  120,20.,1000.)
	regmbbxx= TH1F('regmbbxx',  'add all regmbbxx',  120,20.,1000.)
	openf = TFile("/Users/user/Work/MonoHiggs/BJetRegression/Punzi/ALL-BKG-ADD/allbkg_with_mass_window.root", "read")
	'''
	for i in range(0,548):
		name = 'Mbb_bkg_'+str(i)
		hist1d=openf.Get(name)
		mbbxx.Add(hist1d)
		hist1d.Reset()

		name1 = 'regMbb_bkg_'+str(i)
        	hist1xd=openf.Get(name1)
        	regmbbxx.Add(hist1xd)
        	hist1xd.Reset()
	print(regmbbxx.Integral())
	c1 = TCanvas( 'c1', 'Example with Formula', 200, 10, 700, 500 )
	legend2 = TLegend(0.64, 0.62, 0.76, 0.92, "")
	legend2.SetTextSize(0.03)
	#legend2.SetFillColor(0)
	legend2.SetFillStyle(3002)
	legend2.SetBorderSize(0)
	mbbxx.GetXaxis().SetRangeUser(25.0,400.0)
	mbbxx.SetMarkerStyle( 21 )
	mbbxx.SetMarkerColor(2)
	#mbbxx.SetMarkerStyle( 21 )
	mbbxx.SetXTitle("M_{bb} Bkg")
	mbbxx.SetYTitle("Entries")
	mbbxx.Draw("p")
	regmbbxx.SetLineColor(3)
	regmbbxx.SetMarkerStyle( 21 )
	regmbbxx.SetMarkerColor(3)
	regmbbxx.Draw("p same")

	legend2.AddEntry("mbbxx", "M_bb Bkg Before Correction","p")
	legend2.AddEntry("regmbbxx", "M_bb Bkg After Correction","p")
	legend2.Draw()
	c1.SaveAs("addallbkg.pdf")
	'''
	hist_DYJets = []; hist_ZJets = []; hist_GJets= []; hist_DIBOSON = []; 
	hist_WJets =[]
	hist_STop = []; hist_Top= []; hist_QCD= []; hist_SMH = []
	DIBOSON = ROOT.TH1F()
	Top = ROOT.TH1F()
	WJets = ROOT.TH1F()
	DYJets = ROOT.TH1F()
	ZJets = ROOT.TH1F()
	STop = ROOT.TH1F()
	GJets = ROOT.TH1F()
	QCD = ROOT.TH1F()
	SMH = ROOT.TH1F()
	hists_total_bkg= ROOT.TH1F()

	for i in range(0,546):
		name = hist_name+'_'+str(i)
		name1='h_total_mcweight_'+str(i)
        	hist1d=openf.Get(name)
		histname=hist1d.GetTitle()
		#hist1dx=hist1d.Clone()
		xsBkg = crsec.getXsec(histname)
		#print histname

		hist1dy=openf.Get(name1)
		print 'Integral before= ', hist1d.Integral(), '\n'
		#scalefact=1
		lumi=41.5*1000
		if hist1dy.Integral()>0: scalefact=(lumi*xsBkg)/hist1dy.Integral()
		#if scalefact<0: continue
		print 'scale= ', xsBkg, hist1dy.Integral(), scalefact
		hist1d.Scale(scalefact)
		#print 'Integral after= ', hist1d.Integral()
		#if i==0:
		#	hists_total_bkg=hist1d.Clone()
			#hists_total_bkg.Scale(scalefact)
		#else:
			#hists_total_bkg.Scale(scalefact)
		#	hists_total_bkg.Add(hist1d)
		if 'WJetsToLNu_HT' in histname:
			hist_WJets.append(hist1d)
		elif 'DYJetsToLL_M-50' in histname:
			hist_DYJets.append(hist1d)
		elif 'ZJetsToNuNu' in histname:
			hist_ZJets.append(hist1d)
		elif 'GJets_HT' in histname:
			hist_GJets.append(hist1d)
		elif ('WW' in histname) or ('WZ' in histname) or ('ZZ' in histname):
			hist_DIBOSON.append(hist1d)
		elif ('ST_t' in histname) or ('ST_s' in histname):
			hist_STop.append(hist1d)
		elif 'TTTo' in histname:
			hist_Top.append(hist1d)
		elif 'QCD' in histname:
			hist_QCD.append(hist1d)
			print hist1d.Integral()
		elif 'HToBB' in histname:
			hist_SMH.append(hist1d)


	for i in range(len(hist_WJets)):
		if i==0:
			WJets=hist_WJets[i]
		else:WJets.Add(hist_WJets[i])
		#print hist_WJets[i].Integral()
		#WJets.Sumw2()

	for i in range(len(hist_DYJets)):
        	if i==0:
			DYJets =hist_DYJets[i]		
		else:DYJets.Add(hist_DYJets[i])
	for i in range(len(hist_ZJets)):
		if i==0:
			ZJets =hist_ZJets[i]
		else:ZJets.Add(hist_ZJets[i])
	for i in range(len(hist_GJets)):
		if i==0:
			GJets =hist_GJets[i]
		else:GJets.Add(hist_GJets[i])
	for i in range(len(hist_DIBOSON)):
		if i==0:
			DIBOSON =hist_DIBOSON[i]
		else:DIBOSON.Add(hist_DIBOSON[i])
	for i in range(len(hist_STop)):
		if i==0:
			STop =hist_STop[i]
		else:STop.Add(hist_STop[i])
	for i in range(len(hist_Top)):
		if i==0:
			Top =hist_Top[i]
		else:Top.Add(hist_Top[i])

	for i in range(len(hist_QCD)):
		if i==0:
			QCD =hist_QCD[i]
		else:QCD.Add(hist_QCD[i])
	for i in range(len(hist_SMH)):
		if i==0:
			SMH =hist_SMH[i]
		else:SMH.Add(hist_SMH[i])

	count_ZJets  = ZJets.Integral()
	count_DYJets = DYJets.Integral()
	count_WJets  = WJets.Integral()
	count_STop   = STop.Integral()
	count_GJets  = GJets.Integral()
	count_TT     = Top.Integral()
	count_DB     = DIBOSON.Integral()
	count_QCD    = QCD.Integral()
	count_SMH    = SMH.Integral()


	count_bkgsum = count_ZJets+ count_DYJets+ count_WJets + count_STop + count_GJets + count_TT + count_DB + count_QCD + count_SMH

	print 'tot bkg: ', count_bkgsum, ' tot bkg1: '#, hists_total_bkg.Integral()


	errhist_ZJets  = ZJets.Clone()
	errhist_DYJets = DYJets.Clone()
	errhist_WJets  = WJets.Clone()
	errhist_STop   = STop.Clone()
	errhist_GJets  = GJets.Clone()
	errhist_Top    = Top.Clone()
	errhist_DIBOSON=DIBOSON.Clone()
	errhist_QCD    = QCD.Clone()
	errhist_SMH    = SMH.Clone()

	errhist_ZJets.Rebin(errhist_ZJets.GetNbinsX())
	errhist_DYJets.Rebin(errhist_DYJets.GetNbinsX())
	errhist_WJets.Rebin(errhist_WJets.GetNbinsX())
	errhist_STop.Rebin(errhist_STop.GetNbinsX())
	errhist_GJets.Rebin(errhist_STop.GetNbinsX())
	errhist_Top.Rebin(errhist_Top.GetNbinsX())
	errhist_DIBOSON.Rebin(errhist_DIBOSON.GetNbinsX())
	errhist_QCD.Rebin(errhist_QCD.GetNbinsX())
	errhist_SMH.Rebin(errhist_SMH.GetNbinsX())

	ststerr_ZJets = errhist_ZJets.GetBinError(1) 
	ststerr_DYJets= errhist_DYJets.GetBinError(1)
	ststerr_WJetss  = errhist_WJets.GetBinError(1)
	ststerr_STop   = errhist_STop.GetBinError(1)
	ststerr_GJets  = errhist_GJets.GetBinError(1)
	ststerr_Top    = errhist_Top.GetBinError(1)
	ststerr_DIBOSON= errhist_DIBOSON.GetBinError(1)
	ststerr_QCD    = errhist_QCD.GetBinError(1)
	ststerr_SMH    = errhist_SMH.GetBinError(1)

	print 'count_ZJets: ', count_ZJets
	print 'count_DYJets: ', count_DYJets 
	print 'count_WJets: ', count_WJets
	print 'count_STop: ', count_STop
	print 'count_GJets: ', count_GJets
	print 'cout_Top: ', count_TT
	print 'count_DB: ', count_DB 
	print 'count_QCD: ', count_QCD
	print 'count_SMH: ', count_SMH 


	filename_tex=hist_name+'.tex'
	count_file = open(filename_tex,'w')
	count_file.write('\documentclass{article}\n')
	count_file.write('\usepackage[utf8]{inputenc}\n')
	count_file.write('\usepackage{booktabs}\n')
	count_file.write('\usepackage{multirow}\n')
	count_file.write('\usepackage{longtable}\n')
	count_file.write('\usepackage{graphicx}\n')


	count_file.write('\usepackage{natbib}\n')
	count_file.write('\usepackage{graphicx}\n')

	count_file.write('\\begin{document}\n')

	count_file.write('\\begin{table}[tbp]\n')
	count_file.write('\centering\n')
	count_file.write('\\begin{tabular}{|cc|}\n')
	count_file.write('\hline\n')
	count_file.write('Sample Name & Yield')
	count_file.write('\\\\ \n \hline\n')
	count_file.write('ZJets & ') 
	count_file.write(str('%.2f' %count_ZJets)) 
	count_file.write('\\\\ \n \hline\n')
	count_file.write('DYJets & ')
	count_file.write(str('%.2f' %count_DYJets))
	count_file.write('\\\\ \n \hline\n')
	count_file.write('WJets & ')
	count_file.write(str('%.2f' %count_WJets))
	count_file.write('\\\\ \n \hline\n')
	count_file.write('STop & ')
	count_file.write(str('%.2f' %count_STop))
	count_file.write('\\\\ \n \hline\n')
	count_file.write('GJets & ')
	count_file.write(str('%.2f' %count_GJets))
	count_file.write('\\\\ \n\hline\n')
	count_file.write('Top & ')
	count_file.write(str('%.2f' %count_TT))
	count_file.write('\\\\ \n\hline\n')
	count_file.write('DIBoson & ')
	count_file.write(str('%.2f' %count_DB))
	count_file.write('\\\\ \n\hline\n')
	count_file.write('QCD & ')
	count_file.write(str('%.2f' %count_QCD))
	count_file.write('\\\\ \n\hline\n')
	count_file.write('SMH & ')
	count_file.write(str('%.2f' %count_SMH))
	count_file.write('\\\\ \n\hline\n')
	count_file.write('\end{tabular}\n')
	count_file.write('\caption{BackGround yield table for $M_{bb}$: 2017.}\n')
	count_file.write('\end{table}\n')



	count_file.write('\end{document}\n')
	return count_bkgsum 

punzi_Mbb, punzi_regMbb = array( 'd' ), array( 'd' )

openfsig = TFile("/Users/user/Work/MonoHiggs/BJetRegression/Punzi/ALL-BKG-ADD/Outputplot1.root", "read")
name='signal_entry'
hist_sig_entry=openfsig.Get(name)
for i in range (len(sigmass)):
	sig_entry =0.0
	sig_entry = hist_sig_entry.GetBinContent(i);
	print "Sig entry = ", sig_entry
	name1='Mbb_'+str(i)
	name2='regMbb_'+str(i)
	hist_Mbb=openfsig.Get(name1)
	hist_regMbb=openfsig.Get(name2)
	norm_h_Mbb=hist_Mbb.Integral()
	effi_Mbb=norm_h_Mbb/sig_entry
	mbb_count=bkghist('Mbb_bkg_with_mass_window')
	regmbb_count=bkghist('regMbb_bkg_with_mass_window')
	punzi_Mbb.append(effi_Mbb/(1+TMath.Sqrt(mbb_count)))
	norm_h_regMbb =0.
	norm_h_regMbb = hist_regMbb.Integral()
	effi_regMbb = norm_h_regMbb/sig_entry
	punzi_regMbb.append(effi_regMbb/(1+TMath.Sqrt(regmbb_count)))

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
#gr1.Write()

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
#gr2.Write()

mg = TMultiGraph('mg', 'mg')
mg.Add(gr1, "LP")
mg.Add( gr2,"LP")
mg.SetTitle("  ; Signal Mass Point; Punzi Significance");
mg.Draw("ACP")
c1.BuildLegend()
c1.SaveAs("sig.pdf")






