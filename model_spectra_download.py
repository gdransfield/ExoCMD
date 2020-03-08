#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:06:54 2020

@author: Georgina Dransfield
"""

#Run this script to download all Molliere model spectra and save to folder called 'model_spectra'

import os
import requests as req
import numpy as np

Model = ['{0:05d}'.format(x) for x in np.arange(1,10641)]
urls = ['http://cdsarc.u-strasbg.fr/viz-bin/nph-Plot/w/Vgraph/txt?J%2fApJ%2f813%2f47%2f.%2f' \
            + i + '&--bitmap-size&600x400' for i in Model]

loc = os.getcwd()
filepaths = [os.path.join(loc, 'model_spectra', (i + '.txt')) for i in Model]

def downloads(url_list):
    for filepath, url in zip(filepaths, url_list):
        if not os.path.exists(filepath):
            print(filepath)
            visit_page = req.get(url)
            with open(filepath, 'w') as open_file:
                open_file.write(visit_page.text)

    return

downloads(urls)
