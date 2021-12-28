import pandas as pd
import xmltodict
import json
from pprint import pprint
import numpy as np

kotor_df = pd.read_csv('data/kotor_dataset.csv')
nwn_df = pd.read_csv('data/nwn_dataset.csv')
kotor_df['sentence'] = np.nan
kotor_df['is_persuasion'] = np.nan
kotor_df['sid'] = pd.to_numeric(kotor_df['sid'], downcast='integer')
kotor_df.set_index('sid', inplace=True)
print(kotor_df['sentence'][35659])
kotor1_tlk = 'extracting_data/infinity_engine/kotor1/kotor1_en.xml'
kotor2_tlk = 'extracting_data/infinity_engine/kotor2/kotor2_en.xml'
nwn_tlk = 'extracting_data/infinity_engine/nwn/nwn_en.xml'
files = [kotor1_tlk, kotor2_tlk, nwn_tlk]
for file in files:
    if file == files[0]:
        game = 'KotOR 1'
    if file == files[1]:
        game = 'KotOR 2'
    if file == files[2]:
        game = 'NWN 1'
    with open(file, 'rb') as f:
            xml = xmltodict.parse(f)
            xml = json.dumps(xml)
            xml = json.loads(xml)
            for elem in xml['tlk']['string']:
                #mask = (kotor_df['game'] == game)
                #mask = (kotor_df['sid'] == np.intc(elem['@id']))
                try:
                    kotor_df.loc[(kotor_df.index == int(elem['@id'])) & (kotor_df['game'] == game), 'sentence'] = elem['#text']
                    #print(kotor_df['sentence'])
                except KeyError:
                    pass

filename = 'dialogue_dataset.csv'
kotor_df.to_csv(filename, encoding='utf-8')
