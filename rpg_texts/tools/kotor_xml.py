import xml.etree.ElementTree as ET
import os
import xmltodict, json, xmljson
from pprint import pprint
import pandas as pd
import ast
import re
import numpy as np 
dfs = []
id_sid_mapping = {}

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
def update_idx(xml_dict):
    for entry in entry_dict:
        
        # TODO:
        try:
            entry_id = entry['@id']+"_entry"
        except TypeError:
            return None
        
        entry_keys = entry.keys()
        for key in entry_keys:
            if key == 'locstring':              
                string_id = entry['locstring']['@strref']
                
                if elem['@label'] == 'ReplyList':
                    entry_id = entry['@id']+"_reply"
                                    
                    if string_id == '4294967295':
                            string_id = "EoC"
                df.at[entry_id, 'id'] = entry_id
                df.at[entry_id, 'sid'] = string_id
                id_sid_mapping[entry_id] = string_id
def update_speakers(xml_dict):
    for entry in entry_dict:
        try:
            entry_id = entry['@id']+"_entry"
        except TypeError:
            return None 
        for key in entry.keys():
            if key == 'exostring':
                for elem in entry['exostring']:
                    try:
                        if elem["@label"] == 'Speaker':
                            df.at[entry_id, 'speaker'] = elem['#text']
                    except KeyError:
                        df.at[entry_id, 'speaker'] = 'not_found'
def update_replies(xml_dict):
    for entry in xml_dict:
        try:
            entry_keys = entry.keys()
        except AttributeError:
            return None
        
        for key in entry_keys:
            if key == 'list':
                for elem in entry['list']:                                                              
                    if elem["@label"] == 'RepliesList':
                        try:
                            entry_id = elem['struct']['uint32']['#text']+"_entry"
                        #reply_id = elem['struct']['uint32']['#text']+"_reply"
                        except TypeError:
                            print("Error getting ENTRY ID for",file,elem)
                            print()
                            return None
                        except KeyError:
                            print("struct not found for", file, elem)
                            print()
                            return None

                            df.at[entry_id, 'replies'] = [id_sid_mapping[entry_id]]
                            replies = []
                            for reply in elem['struct']:
                                try:
                                    entry_id = reply['uint32']['#text']+"_entry"
                                except TypeError:
                                    print("entry id not found for", reply)
                                    print()
                                    return None
                                replies.append(id_sid_mapping[entry_id])
                                df.at[entry_id, 'replies'] = replies
                                                                            
                                df.at[entry_id, 'replies'] = [id_sid_mapping[entry_id]]
                                replies = []
                                for reply in elem['struct']:
                                    entryid = reply['uint32']['#text']+"_reply"
                                    replies.append(id_sid_mapping[entry_id])
                                    df.at[entry_id, 'replies'] = replies
                                
files = getListOfFiles('infinity_engine/kotor1/xml/')
files = [list(files)[5]]
for file in files:
    df = pd.DataFrame(columns=['id', 'sid', 'speaker', 'replies'])
    
    with open(file, 'rb') as f:
        xml = xmltodict.parse(f)
        xml = json.dumps(xml)
        xml = json.loads(xml)
        xml_dict = xml['gff3']['struct']['list']
        print(file)
        for elem in xml_dict:
            try:                     
                if elem['@label'] == 'EntryList' or elem['@label'] == 'ReplyList' or elem['@label'] == 'StartingList':
                    try:
                        entry_dict = elem['struct']
                    except KeyError:
                        print("Error finding struct for",elem)
                    update_idx(entry_dict)
                    update_speakers(entry_dict)
                    update_replies(entry_dict)
            except TypeError:
                print(elem)

        
        
                            
    df['replies'] = [ [] if x is np.NaN else x for x in df['replies'] ]
    df['speaker'] = df['speaker'].fillna("PC")
    df = df[df.sid != 'EoC']
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #print(df.to_string(index=False))
    dfs.append(df)

df = [df.set_index('sid') for df in dfs]
print(pd.concat(dfs, axis=1))