#!/usr/bin/env python

import makeMonsters
import h5py

# instantiate monster generator with a large particle size
m = makeMonsters.monsterGenerator(31.9)
# create the particle
m.makeParticle()
# write the particle to an HDF5 file
f = h5py.File('density.h5','w')
f.create_dataset('density', data=m.density)
f.close()
