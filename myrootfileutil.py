import os 

inputpath= "/eos/cms/store/group/phys_exotica/bbMET/ExoPieElementTuples/MC_2017miniaodV2_V1/"

os.system('rm dirlist.txt')
os.system("ls -1 "+inputpath+" > dirlist.txt")

allkeys=[idir.rstrip() for idir in open('dirlist.txt')]
alldirs=[inputpath+"/"+idir.rstrip() for idir in open('dirlist.txt')]


def TextToList(textfile):
    return([iline.rstrip()    for iline in open(textfile)])

        
    
dic={}
for ikey in allkeys:
    print ikey
    dirpath=inputpath+"/"+ikey
    txtfile=ikey+".txt"
    os.system ("find "+dirpath+"  -name \"*.root\" | grep -v \"failed\"  > "+txtfile)
    fileList=TextToList(txtfile)
    os.system('rm '+txtfile)
    dic[ikey]=fileList

print dic
