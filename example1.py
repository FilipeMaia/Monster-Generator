#!/usr/bin/env python

import makeMonsters

# instantiate monster generator
m = makeMonsters.monsterGenerator()
# create the particle
m.makeParticle()
# write the particle to the default filename (density.dat)
m.writeDensityToFile()
