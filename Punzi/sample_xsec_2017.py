import os, sys
#//--------------------------------------------------------------------------------------
def getXsec(samplename):
    Xsec = 1.0

    if 'DYJetsToLL_M-50_HT-100to200'   in samplename: Xsec  = 161.1
    if 'DYJetsToLL_M-50_HT-200to400'   in samplename: Xsec  = 48.66
    if 'DYJetsToLL_M-50_HT-400to600'   in samplename: Xsec  = 6.968
    if 'DYJetsToLL_M-50_HT-600to800'   in samplename: Xsec  = 1.743
    if 'DYJetsToLL_M-50_HT-800to1200'  in samplename: Xsec  = 0.8052
    if 'DYJetsToLL_M-50_HT-1200to2500' in samplename: Xsec  = 0.1933
    if 'DYJetsToLL_M-50_HT-2500toInf'  in samplename: Xsec  = 0.003468


    if 'ZJetsToNuNu_HT-100to200'   in samplename: Xsec  = 280.35 
    if 'ZJetsToNuNu_HT-200to400'   in samplename: Xsec  = 77.67 
    if 'ZJetsToNuNu_HT-400to600'   in samplename: Xsec  = 10.73 
    if 'ZJetsToNuNu_HT-600to800'   in samplename: Xsec  = 2.559 
    if 'ZJetsToNuNu_HT-800to1200'  in samplename: Xsec  = 1.1796 
    if 'ZJetsToNuNu_HT-1200to2500' in samplename: Xsec  = 0.28833 
    if 'ZJetsToNuNu_HT-2500toInf'  in samplename: Xsec  = 0.006945 


    if 'WJetsToLNu_HT-100to200'   in samplename: Xsec  = 1395.0
    if 'WJetsToLNu_HT-200to400'   in samplename: Xsec  = 407.9
    if 'WJetsToLNu_HT-400to600'   in samplename: Xsec  = 57.48
    if 'WJetsToLNu_HT-600to800'   in samplename: Xsec  = 12.87
    if 'WJetsToLNu_HT-800to1200'  in samplename: Xsec  = 5.366
    if 'WJetsToLNu_HT-1200to2500' in samplename: Xsec  = 1.074
    if 'WJetsToLNu_HT-2500toInf'  in samplename: Xsec  = 0.008001


    if 'GJets_HT-40to100'    in samplename: Xsec  = 18700.0
    if 'GJets_HT-100to200'   in samplename: Xsec  = 8640.0 
    if 'GJets_HT-200to400'   in samplename: Xsec  = 2185.0
    if 'GJets_HT-400to600'   in samplename: Xsec  = 259.9
    if 'GJets_HT-600toInf'   in samplename: Xsec  = 85.31 

    if 'QCD_HT100to200'    in samplename: Xsec  = 23700000.0
    if 'QCD_HT200to300'    in samplename: Xsec  = 1547000.0
    if 'QCD_HT300to500'    in samplename: Xsec  = 322600.0   
    if 'QCD_HT500to700'    in samplename: Xsec  = 29980.0
    if 'QCD_HT700to1000'   in samplename: Xsec  = 6334.0 
    if 'QCD_HT1000to1500'  in samplename: Xsec  = 1088.0
    if 'QCD_HT1500to2000'  in samplename: Xsec  = 99.11
    if 'QCD_HT2000toInf'   in samplename: Xsec  = 20.23 

    if 'TTToSemiLeptonic'  in samplename: Xsec = 373.91
    if 'TTToHadronic'      in samplename: Xsec = 367.83
    if 'TTTo2L2Nu'         in samplename: Xsec = 95.02

    if 'ST_s-channel_4f_leptonDecays' in samplename: Xsec = 3.74
    if 'ST_t-channel_antitop_4f'      in samplename: Xsec = 67.91
    if 'ST_t-channel_top_4f'          in samplename: Xsec = 113.3
    if 'ST_tW_antitop_5f'             in samplename: Xsec = 34.97
    if 'ST_tW_top_5f'                 in samplename: Xsec = 34.91

    if 'ZZ_TuneCP5_13TeV' in samplename: Xsec = 12.14
    if 'WW_TuneCP5_13TeV' in samplename: Xsec = 75.8
    if 'WZ_TuneCP5_13TeV' in samplename: Xsec = 27.6

    if 'ggZH_HToBB_ZToNuNu_M125'  in samplename: Xsec = 0.01222
    if 'ggZH_HToBB_ZToLL_M125'    in samplename: Xsec = 0.006185
    if 'WminusH_HToBB_WToQQ_M125' in samplename: Xsec = 0.3654
    if 'WplusH_HToBB_WToLNu_M125' in samplename: Xsec = 0.2819
    if 'ZH_HToBB_ZToLL_M125'      in samplename and 'ggZH_HToBB_ZToLL_M125' not in samplename:   Xsec = 0.07924
    if 'ZH_HToBB_ZToNuNu_M125'    in samplename and 'ggZH_HToBB_ZToNuNu_M125' not in samplename: Xsec = 0.1565


    return Xsec
