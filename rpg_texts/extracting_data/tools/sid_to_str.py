import pandas as pd
import xmltodict
import json
from pprint import pprint
import numpy as np
import re

perses = ["[Persu", ';Persu',
          "berreden]", "berreden&"
          "[Überre", ";Überreden", "[ Überre",
          '[Peru', ';Peru', "[ Persu", '[ Peru',
          '[Überreden/Lügen]', '[Überreden/Bluff]','[Force Persuade]']

kotor_df = pd.read_csv('data/kotor_dataset.csv')
nwn_df = pd.read_csv('data/nwn_dataset.csv')
df = pd.concat([kotor_df, nwn_df])

df['sentence'] = np.nan
df['is_persuasion'] = np.nan
df['sid'] = pd.to_numeric(df['sid'], downcast='integer')
df.set_index('sid', inplace=True)
kotor1_tlk = 'extracting_data/infinity_engine/kotor1/kotor1_en.xml'
kotor2_tlk = 'extracting_data/infinity_engine/kotor2/kotor2_en.xml'
nwn_tlk = 'extracting_data/infinity_engine/nwn/nwn_en.xml'
files = [kotor1_tlk, kotor2_tlk, nwn_tlk]
for file in files:
    if file == files[0]:
        game = 'KotOR 1'
    if file == files[1]:
        game = 'KotOR 2'
    if file == files:
        game = 'NWN 1'
    with open(file, 'rb') as f:
            xml = xmltodict.parse(f)
            xml = json.dumps(xml)
            xml = json.loads(xml)
            for elem in xml['tlk']['string']:
                try:
                    df.loc[df.index == int(elem['@id']), 'sentence'] = elem['#text']
                    if any(map(elem['#text'].__contains__, perses)):
                        split_pers = elem['#text'].split("]")
                        pers_tag = re.sub("\[", "", split_pers[0])
                        df.loc[df.index == int(elem['@id']), 'persuasion'] = pers_tag
                        print(pers_tag)
                        
                except KeyError:
                    pass

filename = 'persuasion_dialogue_dataset.csv'
df.to_csv(filename, encoding='utf-8')
