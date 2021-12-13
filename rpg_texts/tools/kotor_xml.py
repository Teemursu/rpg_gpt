import xml.etree.ElementTree as ET
import os
import xmltodict, json, xmljson
from pprint import pprint
import pandas as pd

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

files = getListOfFiles('infinity_engine/kotor1/xml/')
#files = [list(files)[0]]
for file in files:
    with open(file, 'rb') as f:
        xml_dict = xmltodict.parse(f)
        xml_dict = json.dumps(xml_dict)
        xml_dict = json.loads(xml_dict)
        #pprint(xml_dict['gff3']['struct'].values())
        #for elem in xml_dict['gff3']['struct'].values():
            #print(elem)
            
        
        for elem in xml_dict['gff3']['struct']['list']:
            try:
            
                if elem['@label'] == 'EntryList':
                    entry_dict = elem['struct']
                    for elem in entry_dict:
                
                        try:
                            string_id = elem['locstring']['@strref']
                        except:
                            print("Strref not found for", file)
                            string_id = "not found"
                        try:
                            speaker = elem['exostring'][0]['#text']
                        except:
                           # print("Speaker not found in", file)
                            speaker = 'empty'
                        print(string_id, speaker)
            except TypeError:
                print("EntryList not found for",elem,"in", file)
                
            try:   #print()
                if elem['@label'] == 'ReplyList':
                    reply_dict = elem['struct']
                    for elem in reply_dict:
                        print("replylist goes here")
            except :
                print("ReplyList not found in ", file)