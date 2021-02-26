import math
import numpy as np
import json
from functools import reduce #compute the product of a list used in MatchProbability
import glob
from scipy import stats


lociMarkers = ["D3S1358", "D1S1656", "D2S441", "D10S1248", "D13S317", "Penta E", "D16S539",
    "D18S51", "D2S1338", "CSF1PO", "Penta D", "TH01", "vWA", "D21S11", "D7S820", "D5S818",
    "TPOX", "D8S1179", "D12S391", "D19S433", "SE33", "D22S1045", "FGA"
]

data = {}
for files in glob.glob(r"D:\Documenten\TUdelft\thesis\mep_veldhuis\data\raw_sampled_5\*\Trace*"):
    file = open(files, "r")
    line1 = file.readline()
    delim = line1.split("Name")[1].split("Marker")[0]
    headers = line1.rstrip().split(delim)
    colA = [i for i,x in enumerate(headers) if x.split(" ")[0] == "Allele"]
    colH = [i for i,x in enumerate(headers) if x.split(" ")[0] == "Height"]
    for lines in file:
        line = lines.rstrip().split(delim)
        sample = line[0]
        locus = line[1]
        if locus in ["DYS391","DYS576","DYS570", "AMEL"]: # Can be removed when gender-specific loci are used and can be adapted when more loci in the kit are not used for machine learning.
            continue
        if sample not in data:
            data[sample] = {}
            data[sample][locus] = {}
        else:
            if locus not in data[sample]:
                data[sample][locus] = {}

        for alleleNumber, colNumber in enumerate(colA):
            height = line[colH[alleleNumber]]
            if height == "":
                continue
            data[sample][locus][line[colNumber]] = {}
            data[sample][locus][line[colNumber]]["height"] = float(height)
    file.close()


#AlleleFrequency
freqs={}
AFfile = open(r"D:\Documenten\TUdelft\thesis\mep_veldhuis\data\AlleleFrequencies_2085.txt", "r")
AFfile.readline()
totalAinF = 0
for lines in AFfile:
    line = lines.rstrip().split("\t")
    locus = line[0]
    allele = line[1]
    frequency = line[2]
    totalAinF += 1
    if locus not in freqs:
        freqs[locus] = {}
        freqs[locus][allele] = float(frequency)
    else:
        freqs[locus][allele] = float(frequency)

AFfile.close()





Features = {}
allelecount = {}
locusAlleles = {}  
noAF = []
AlFreqs = {}
for sample in data:
    if sample not in Features:
        Features[sample] = {}
        Features[sample]["Locus"] = {}
        Features[sample]["MAC0"] = 0
        Features[sample]["MAC1-2"] = 0
        Features[sample]["MAC3-4"] = 0
        Features[sample]["MAC5-6"] = 0
        Features[sample]["MAC7-8"] = 0
        Features[sample]["MAC9"] = 0
        Features[sample]["NOC"] = 6 #default
        allelecount[sample] = []
    for locus, alleles in data[sample].items():
        AC = len(alleles.keys())
        Features[sample]["Locus"][locus] = {}
        Features[sample]["Locus"][locus]["AlleleCount"+"_"+locus] = AC
        Features[sample]["Locus"][locus]["MinNOC"+"_"+locus] = math.ceil(float(AC)/float(2))
        allelecount[sample].append(AC)
        if AC == 0:
            Features[sample]["MAC0"]+=1
        if AC == 1 or AC == 2:
            Features[sample]["MAC1-2"]+=1
        if AC == 3 or AC == 4:
            Features[sample]["MAC3-4"]+=1
        if AC == 5 or AC == 6:
            Features[sample]["MAC5-6"]+=1
        if AC == 7 or AC == 8:
            Features[sample]["MAC7-8"]+= 1
        if AC > 8:
            Features[sample]["MAC9"]+=1
        #AlleleFrequency
        for allele in alleles.keys():
            if sample not in locusAlleles:
                locusAlleles[sample] = {}
            if allele in freqs[locus]:
                AF = freqs[locus][allele]
                if locus not in locusAlleles[sample]:
                    locusAlleles[sample][locus] = []
                    locusAlleles[sample][locus].append(allele)
                else:
                    locusAlleles[sample][locus].append(allele)
                if sample in AlFreqs:
                    if locus in AlFreqs[sample]:
                        AlFreqs[sample][locus].append(AF)
                    else:
                        AlFreqs[sample][locus] = []
                        AlFreqs[sample][locus].append(AF)
                else:
                    AlFreqs[sample] = {}
                    AlFreqs[sample][locus] = []
                    AlFreqs[sample][locus].append(AF)
            else:
                noAF.append(locus)
    for loci in lociMarkers:
        if loci in Features[sample]["Locus"]:
            continue
        else:
            Features[sample]["Locus"][loci] = {}
            Features[sample]["Locus"][loci]["AlleleCount"+"_"+loci] = 0
            Features[sample]["Locus"][loci]["MinNOC"+"_"+loci] = 0
            Features[sample]["MAC0"]+=1
            allelecount[sample].append(0)

#Allelecounts
for sample in allelecount:
    if len(allelecount[sample]) < len(lociMarkers):
        allelecount[sample].extend([0]*(len(lociMarkers)-len(allelecount[sample])))
    Features[sample]["TAC"] = sum(allelecount[sample])
    Features[sample]["MAC"] = max(allelecount[sample])
    Features[sample]["MinAC"] = min(allelecount[sample])
    Features[sample]["MinNOC"] = math.ceil(float(min(allelecount[sample]))/float(2))
    Features[sample]["meanAllele"] = np.mean(allelecount[sample])
    Features[sample]["medianAllele"] = np.median(allelecount[sample])
    Features[sample]["stdAllele"] = np.std(allelecount[sample])

#AlleleFrequencies
for sample in Features:
    for loci in lociMarkers:
        if loci in AlFreqs[sample]:
            Features[sample]["Locus"][loci]["LowAF"+"_"+loci] = min(AlFreqs[sample][loci])
            Features[sample]["Locus"][loci]["HighAF"+"_"+loci] = max(AlFreqs[sample][loci])
            Features[sample]["Locus"][loci]["SumAF"+"_"+loci] = sum(AlFreqs[sample][loci])
            Features[sample]["Locus"][loci]["PercAF"+"_"+loci] = (len(AlFreqs[sample][loci])/len(freqs[locus]))*100
        else:
            Features[sample]["Locus"][loci]["LowAF"+"_"+loci] = 0
            Features[sample]["Locus"][loci]["HighAF"+"_"+loci] = 0
            Features[sample]["Locus"][loci]["SumAF"+"_"+loci] = 0
            Features[sample]["Locus"][loci]["PercAF"+"_"+loci] = 0




'''
Calculate MatchProbability

The formula when the alleles are the same:
(2*q+(1-q)*(AF1*pop*2+2*sb)/(2*pop+2*sb))*(3*q+(1-q)*(AF1*pop*2+2*sb)/(2*pop+2*sb))/((1+q)*(1+2*q))

The formula, when allele 1 & 2 are different, is:
(2*((q+(1-q)*(AF1*pop*2+sb)/(2*pop+2*sb))*(q+(1-q)*(AF2*pop*2+sb)/(2*pop+2*sb))/((1+q)*(1+2*q))))

Parameters:
AF1 = allelefrequency of allele 1, AF2 = allelefrequency of allele 2, pop =
population size of the allele frequency file, q = theta correction, sb = size bias.
'''
chanceLoci = {}
MatchProbability = {}
q = 0.0 # Change 
pop = 2085 # Change
sb = 2 # Change
for sample in locusAlleles:
    chanceLoci[sample] = []
    for loci in locusAlleles[sample]:
        alleles = locusAlleles[sample][loci]
        pairs = [(alleles[i],alleles[j]) for i in range(len(alleles)) for j in range(i, len(alleles))]
        sumLoci = []
        for pair in pairs:
            allele1 = pair[0]
            allele2 = pair[1]
            AF1 = freqs[loci][allele1]
            AF2 = freqs[loci][allele2]
            if allele1 == allele2:
                kans = (2*q+(1-q)*(AF1*pop*2+2*sb)/(2*pop+2*sb))*(3*q+(1-q)*(AF1*pop*2+2*sb)/(2*pop+2*sb))/((1+q)*(1+2*q))
                sumLoci.append(kans)
            else:
                kans = (2*((q+(1-q)*(AF1*pop*2+sb)/(2*pop+2*sb))*(q+(1-q)*(AF2*pop*2+sb)/(2*pop+2*sb))/((1+q)*(1+2*q))))
                sumLoci.append(kans)
        chanceLoci[sample].append(sum(sumLoci))
    MatchProbability[sample] = reduce((lambda x,y: x*y), chanceLoci[sample])

for sample in MatchProbability:
    Features[sample]["MatchProbability"] = MatchProbability[sample]



for sample in Features:
    Features[sample]["NOC"] = 5



#Heights
HSample = {}
Heights = {}
Threshold = {}
Sizes = {}
stochasticThreshold = 800 # Change
for sample in data:
    for locus, alleles in data[sample].items():
        for i in alleles:
            height = float(alleles[i]["height"])
            # size = float(alleles[i]["size"]) # No size in this dataset.
            if sample in Heights.keys():
                HSample[sample].append(height)
                if height < stochasticThreshold :
                    Threshold[sample]["below"]+= 1            
                else:
                    Threshold[sample]["above"]+=1
                if locus in Heights[sample].keys():
                    Heights[sample][locus].append(height)
                else:
                    Heights[sample][locus] = []
                    Heights[sample][locus].append(height)
            else:
                Heights[sample]= {}
                Heights[sample][locus] = []
                Heights[sample][locus].append(height)
                HSample[sample] = []
                HSample[sample].append(height)
                Threshold[sample] = {}
                Threshold[sample]["above"] = 0
                Threshold[sample]["below"] = 0
                if height < stochasticThreshold :
                    Threshold[sample]["below"]+= 1            
                else:
                    Threshold[sample]["above"]+=1          
            # if sample in Sizes:
            #     if locus in Sizes[sample]:
            #         Sizes[sample][locus].append(size)
            #     else:
            #         Sizes[sample][locus] = []
            #         Sizes[sample][locus].append(size)
            # else:
            #     Sizes[sample] = {}
            #     Sizes[sample][locus] = []
            #     Sizes[sample][locus].append(size)


for sampleNr in HSample.keys():
    Features[sampleNr]["minHeight"] = min(HSample[sampleNr])
    Features[sampleNr]["maxHeight"] = max(HSample[sampleNr])
    Features[sampleNr]["meanHeight"] = np.mean(HSample[sampleNr])
    Features[sampleNr]["stdHeight"] = np.std(HSample[sampleNr])
    Features[sampleNr]["medianHeight"] = np.median(HSample[sampleNr])
    Features[sampleNr]["peaksAboveRFU"] = Threshold[sampleNr]["above"]
    Features[sampleNr]["peaksBelowRFU"] = Threshold[sampleNr]["below"]
    if Threshold[sampleNr]["above"] == 0:
        Features[sampleNr]["Below/AboveRFU"] = Threshold[sampleNr]["below"] / 0.5
    else: 
        Features[sampleNr]["Below/AboveRFU"] = Threshold[sampleNr]["below"] / Threshold[sampleNr]["above"]
    if Threshold[sampleNr]["below"] == 0:
        Features[sampleNr]["Above/BelowRFU"] = Threshold[sampleNr]["above"] / 0.5
    else: 
        Features[sampleNr]["Above/BelowRFU"] = Threshold[sampleNr]["above"] / Threshold[sampleNr]["below"]

for key in Heights.keys():
    for loci in lociMarkers:
        if loci in Heights[key].keys():
            Features[key]["Locus"][loci]["minHeight"+"_"+loci] = min(Heights[key][loci])
            Features[key]["Locus"][loci]["maxHeight"+"_"+loci] = max(Heights[key][loci])
            Features[key]["Locus"][loci]["meanHeight"+"_"+loci] = np.mean(Heights[key][loci])
            Features[key]["Locus"][loci]["stdHeight"+"_"+loci] = np.std(Heights[key][loci])
            Features[key]["Locus"][loci]["medianHeight"+"_"+loci] = np.median(Heights[key][loci])
        if loci not in Heights[key].keys():
            Features[key]["Locus"][loci]["minHeight"+"_"+loci] = 0.0
            Features[key]["Locus"][loci]["maxHeight"+"_"+loci] = 0.0
            Features[key]["Locus"][loci]["meanHeight"+"_"+loci] = 0.0
            Features[key]["Locus"][loci]["stdHeight"+"_"+loci] = 0.0
            Features[key]["Locus"][loci]["medianHeight"+"_"+loci] = 0.0


#Degredation Slopes
# slopeData = {}
# for sample in Sizes:
#     for loci in lociMarkers:
#         if loci in Heights[sample]:
#             sumH = sum(Heights[sample][loci])
#             avH = np.mean(Heights[sample][loci])
#             avSize = np.mean(Sizes[sample][loci])
#         else:
#             sumH = 0
#             avH = 0
#             avSize = 0
#         if sample not in slopeData.keys():
#             slopeData[sample] = {}
#             slopeData[sample]["sumY"] = []
#             slopeData[sample]["avY"] = []
#             slopeData[sample]["sizeX"] = []
#             slopeData[sample]["sumY"].append(sumH)
#             slopeData[sample]["avY"].append(avH)
#             slopeData[sample]["sizeX"].append(avSize)
#         else:
#             slopeData[sample]["sumY"].append(sumH)
#             slopeData[sample]["avY"].append(avH)
#             slopeData[sample]["sizeX"].append(avSize)        


# for i in slopeData.keys():
#     X = slopeData[i]["sizeX"]
#     Ysum = slopeData[i]["sumY"]
#     Yav = slopeData[i]["avY"]
#     slopeSum = stats.linregress(X,Ysum)[0]
#     slopeAv = stats.linregress(X,Yav)[0]
#     Features[i]["slopeSum"] = slopeSum
#     Features[i]["slopeAv"] = slopeAv  


#Save features dictoniary to file.
with open(r"D:\Documenten\TUdelft\thesis\mep_veldhuis\data\features5000\Features5_5000.txt", "w") as outfile:
    json.dump(Features, outfile)
