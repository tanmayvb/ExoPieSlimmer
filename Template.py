import os 
import sys 

etabins=[]
ptbins=[]

ele_trig_eff_sf=get_ele_trig_eff_from_dict_files

of just 

{
'''
add dict here 
'''
}


#may be add everything in one object so that just one object is enough to play with. like this 

ele_trig_sf_info=[etabins, ptbins, ele_trig_eff_sf]

def getptetabin(pt,eta):
    
    ## your code here 
    '''
    iptbin=
    ietabin=
    '''
    return [iptbin,ietabin]


def getKey(iptbin,ietabin):
    ptetakey='pt_'+str(iptbin)+'eta_'+str(ietabin)

def readEleTrigSF(pt, eta):
    
    iptetabin=getptbin(pt,eta)
    iptbin = iptetabin[0]
    ietabin = iptetabin[1]
    
    key__ = getKey(iptbin,ietabin)
    
    
    sf = ele_trig_eff_sf[key__]
