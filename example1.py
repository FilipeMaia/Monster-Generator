#!/usr/bin/env python

import makeMonsters

# instantiate monster generator with default parameters
m = makeMonsters.monsterGenerator()

# create the particle and particle support (for phasing)
m.makeParticle()
# write the particle and support to the default filename (density.dat and support.dat)
m.writeDensityToFile()
m.writeSupportToFile()

# define canonical beamstop and detector, then create 3D Fourier intensities
m.diffract()
# write detector pixel positions (detector.dat)
m.writeDetectorToFile() 
# write 3D intensities (intensity.dat)
m.writeIntensitiesToFile()

# Could use m.writeAllOuputToFile() aggregate function to perform all the write functions above

# Uncomment these lines to view created monsters
# m.showDensity()
# m.showLogIntensity()
# m.showLogIntensitySlices()
