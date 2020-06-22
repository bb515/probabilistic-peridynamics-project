# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:21:16 2020

@author: Ben Boys
"""

import subprocess

beams = ['1650beam792.msh', '1650beam2652.msh', '1650beam3570.msh', '1650beam4095.msh', '1650beam6256.msh', '1650beam15840.msh', '1650beam32370.msh', '1650beam74800.msh', '1650beam144900.msh', '1650beam247500.msh']
# =============================================================================
# with open("data_force_optimised.txt", "w+") as output:
#     for beam in beams:
#         subprocess.call(["python", "./example5.py", beam, "--optimised", "--profile"], stdout=output);
# =============================================================================
with open("data_displacement_optimised.txt", "w+") as output:
    for beam in beams:
        subprocess.call(["python", "./example5d.py", beam, "--optimised", "--lumped", "--profile"], stdout=output);