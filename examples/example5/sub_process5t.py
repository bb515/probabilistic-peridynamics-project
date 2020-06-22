# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:21:16 2020

@author: Ben Boys
"""

import subprocess

beams = ['1650beam792t.msh',
         '1650beam2652t.msh',
         '1650beam3570t.msh',
         '1650beam4095t.msh',
         '1650beam6256t.msh',
         '1650beam15840t.msh',
         '1650beam32370t.msh',
         '1650beam74800t.msh',
         '1650beam144900t.msh']
with open("data_force_1650t.txt", "w+") as output:
    for beam in beams:
        subprocess.call(["python", "./example5t.py", beam, "--profile"], stdout=output);
