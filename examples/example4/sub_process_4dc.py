# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:21:16 2020

@author: Ben Boys
"""

import subprocess

beams = ['3300beam952.msh',
         '3300beam2970.msh',
         '3300beam4392.msh',
         '3300beam6048.msh',
         '3300beam11836.msh',
         '3300beam17600.msh',
         '3300beam31680.msh',
         '3300beam64350.msh',
         '3300beam149600.msh']
#beams = ['3300beam952t.msh',
#         '3300beam2970t.msh',
#         '3300beam4392t.msh',
#         '3300beam6048t.msh',
#         '3300beam11836t.msh',
#         '3300beam17600t.msh',
#         '3300beam31680t.msh',
#         '3300beam64350t.msh',
#         '3300beam149600t.msh',
#          '3300beam495000t.msh']
with open("data_displacement_3300.txt", "w+") as output:
    for beam in beams:
        subprocess.call(["python", "./example4dc.py", "--optimised", "--lumped2", beam], stdout=output);
# =============================================================================
# with open("data_displacement_optimised_3300.txt", "w+") as output:
#     for beam in beams:
#         subprocess.call(["python", "./example4d.py", beam, "--optimised", "--profile"], stdout=output);
# =============================================================================
