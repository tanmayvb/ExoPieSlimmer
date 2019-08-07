import os 
import sys 
from SFFactory import * 
etabins=eta_bins_eleTrig_hEffEtaPt
ptbins=pt_bins_eleTrig_hEffEtaPt

ele_trig_eff_sf=dict_eleTrig_hEffEtaPt

print etabins, ptbins, ele_trig_eff_sf

#may be add everything in one object so that just one object is enough to play with. like this 

ele_trig_sf_info=[etabins, ptbins, ele_trig_eff_sf]

def getptetabin(pt,eta):
    
    ## your code here 
    '''
    iptbin=
    ietabin=
    '''
    #return [iptbin,ietabin]
    return [15,4]


def getKey(iptbin,ietabin):
    ptetakey='pt_'+str(iptbin)+'_eta_'+str(ietabin)
    return ptetakey
def readEleTrigSF(pt, eta):
    
    iptetabin=getptetabin(pt,eta)
    iptbin = iptetabin[0]
    ietabin = iptetabin[1]
    print iptetabin
    key__ = getKey(iptbin,ietabin)
    
    print "key = ", key__
    sf = ele_trig_eff_sf[key__]
    return sf

print "sf=", readEleTrigSF(12,1.4)
