import numpy as N
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.ticker as Tick 

#Add optParse module

class monsterGenerator(object):
	def __init__(self, inParticleRadius=5.9, inDamping=1.5, inFrac=0.5, inPad=1.8):
		"""
		Contains recipe to create, diffract and show a low-pass-filtered, random particle contrast.
		
		If particle and diffraction parameters are not given, then default ones are used:
		particleRadius  = 5.9	(num. of pixels; good results if number is x.9, where x is an integer),
		damping         = 1.5	(larger damping=larger DeBye-Waller factor),
		frac            = 0.5	(frac. of most intense realspace voxels forced to persist in iterative particle generation),
		pad             = 1.8	(extra voxels to pad on 3D particle density to create support for phasing),
		radius          = N.floor(particleRadius) + N.floor(pad) (half length of cubic volume that holds particle),
		size            = 2*radius + 1 (length of cubic volume that holds particle).

		Defined in function helpfile for diffract():
		support,
		density,
		supportPositions.
		
		Defined in function helpfile for diffract():
		z,
		sigma,
		qmax,
		qmin,
		numPixToEdge,
		detectorDist,
		detector,
		beamstop,
		intensities.
		"""
		self.particleRadius = inParticleRadius
		self.damping = inDamping
		self.frac = inFrac
		self.pad = inPad
		
		self.radius = int(N.floor(self.particleRadius) + N.floor(self.pad))
		self.size = 2*self.radius + 1
		
		#Re-defined after makeMonster() is called
		self.support = []
		self.density = []
		self.supportPositions = []

		#Re-defined after diffract() is called
		self.z = 1. 
		self.sigma = 6.0				
		self.qmax = N.ceil(self.sigma * self.particleRadius)
		self.qmin = 1.4302966531242025 * (self.qmax / self.particleRadius)
		zSq = self.z*self.z 
		self.numPixToEdge = N.floor(self.qmax / N.sqrt(zSq/(1.+zSq) + (zSq/N.sqrt(1+zSq) -self.z)))
		self.detectorDist = self.z * self.numPixToEdge
		self.detector = []
		self.beamstop = []
		self.intensities = []

	def makeParticle(self):
		"""
		Recipe for creating random, "low-passed-filtered binary" contrast by 
		alternating binary projection and low-pass-filter on an random, 3D array of numbers.
		
		Variables defined here:
		support             =	sphereical particle support (whose radius is less than particleRadius given),
		density             =	3D particle contrast,
		supportPositions    =	voxel position of support used in phasing.
		"""
		[x,y,z] = N.mgrid[-self.radius:self.radius+1, -self.radius:self.radius+1, -self.radius:self.radius+1]
		self.support = N.sqrt(x*x + y*y + z*z) < self.particleRadius
		filter = N.fft.ifftshift( N.exp(-self.damping * (x*x + y*y + z*z) / (self.radius*self.radius)) )
		suppRad = N.floor(self.radius)
		flatSupport = self.support.flatten()
		
		lenIter = self.size * self.size * self.size
		iter = N.random.rand(lenIter)
		numPixsToKeep = N.ceil(self.frac * self.support.sum())
		
		#Recipe for making binary particle.
		for i in range(4):
			#Sets the largest numPixsToKeep pixels to one
			#	and the rest to zero
			iter *= flatSupport
			ordering = iter.argsort()[-1:-numPixsToKeep-1:-1]
			iter[:] = 0
			iter[ordering] = 1.
			
			#Smoothing with Gaussian filter
			temp = N.fft.fftn(iter.reshape(self.size, self.size, self.size))
			iter = N.real( N.fft.ifftn(filter*temp).flatten() )
			
		#Create padded support
		paddedSupport = N.sqrt(x*x + y*y + z*z) < (self.particleRadius + self.pad)
		self.supportPositions = N.array([[self.radius+i,self.radius+j,self.radius+k] for i,j,k,l in zip(x.flat, y.flat, z.flat, paddedSupport.flat) if l >0]).astype(int)
	
		self.density = iter.reshape(self.size, self.size, self.size)

	def placePixel(self, ii, jj, zL):
		"""
		Gives (qx,qy,qz) position of pixels on Ewald sphere when given as input
		the (x,y,z)=(ii,jj,zL) position of pixel in the diffraction laboratory. 
		The latter is measured in terms of the size of each realspace pixel.
		"""
		v = N.array([ii,jj,zL])
		vDenom = N.sqrt(1 + (ii*ii + jj*jj)/(zL*zL))
		return v/vDenom - N.array([0,0,zL])


	def diffract(self, inMaxScattAngDeg=45., inSigma=6.0, inQminNumShannonPix=1.4302966531242025):
		"""
		Requires makeMonster() to first be called, so that particle density is created.
		
		Function diffract() needs the maximum scattering angle to the edge of the detector, the 
		sampling rate of Shannon pixels (inSigma=6 means each Shannon pixel is sampled 
		by roughly 6 pixels), and the central missing data region has a radius of 
		inQminNumShannonPix (in units of Shannon pixels).
		
		Variables redefined here:
		z               =	cotangent of maximum scattering angle,
		sigma           =	sampling rate on Shannon pixels,
		qmax            =	number of pixels to edge of detector,
		numPixToEdge    =	same as qmax,
		detectorDist    =	detector-particle distance (units of detector pixels),
		beamstop        =	voxel positions of central disk of missing data on detector,
		detector        =	pixel position of 2D area detector (projected on Ewald sphere),
		intensities     =	3D Fourier intensities of particle.
		"""
		self.z = 1/N.tan(N.pi * inMaxScattAngDeg / 180.)
		self.sigma = inSigma
		self.qmax = N.ceil(self.sigma * self.particleRadius) 
		zSq = self.z*self.z
		self.numPixToEdge = N.floor(self.qmax / N.sqrt(zSq/(1.+zSq) + (zSq/N.sqrt(1+zSq) -self.z)))
		self.detectorDist = self.z * self.numPixToEdge 
		self.qmin = inQminNumShannonPix * (self.qmax / self.particleRadius)
		
		#make beamstop
		fQmin = N.floor(self.qmin)
		[x,y,z] = N.mgrid[-fQmin:fQmin+1, -fQmin:fQmin+1, -fQmin:fQmin+1]
		tempBeamstop = [[i,j,k] for i,j,k in zip(x.flat, y.flat, z.flat) if (N.sqrt(i*i + j*j + k*k) < (self.qmin - N.sqrt(3.)))]
		self.beamstop = N.array(tempBeamstop).astype(int)
		
		#make detector
		[x,y] = N.mgrid[-self.numPixToEdge:self.numPixToEdge+1, -self.numPixToEdge:self.numPixToEdge+1]
		tempDetectorPix = [self.placePixel(i,j,self.detectorDist) for i,j in zip(x.flat, y.flat)]
		qualifiedDetectorPix = [i for i in tempDetectorPix if (self.qmin<N.sqrt(i[0]*i[0] +i[1]*i[1] + i[2]*i[2])<self.qmax)]
		self.detector = N.array(qualifiedDetectorPix)
		
		#make fourier intensities
		intensSize = 2 * self.qmax + 1
		self.intensities = N.zeros((intensSize, intensSize, intensSize))
		self.intensities[:self.size, :self.size, :self.size] = self.density
		self.intensities = N.fft.fftshift(N.fft.fftn(self.intensities))
		self.intensities = N.abs(self.intensities * self.intensities.conjugate())
		
		
	def showDetector(self):
		"""
		Shows detector pixels as points on scatter plot; could be slow for large detectors.
		"""
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(self.detector[:,0], self.detector[:,1], self.detector[:,2], c='r', marker='s')
		ax.set_zlim3d(-self.qmax, self.qmax)
		plt.show()
		
		
	def showDensity(self):
		"""
		Shows particle density as an array of sequential, equal-sized 2D sections.
		"""
		subplotlen = int(N.ceil(N.sqrt(len(self.density))))
		fig = plt.figure(figsize=(9.5, 9.5))
		for i in range(len(self.density)):
			ax = fig.add_subplot(subplotlen, subplotlen, i+1)
			ax.imshow(self.density[:,:,i], vmin=0, vmax=1.1, interpolation='nearest', cmap=plt.cm.bone)
			ax.set_title('z=%d'%i, color='white', position=(0.85,0.))
		plt.show()
		
	def showLogIntensity(self, inSection=0):
		"""
		Show a particular intensities section of Fourier intensities.
		Sections range from -qmax to qmax.
		"""
		plotSection = inSection
		if(plotSection<=0):
			plotSection += self.qmax
			
		fig = plt.figure(figsize=(13.9,9.5))
		ax = fig.add_subplot(111)
		ax.set_title("log(intensities) of section q=%d"%plotSection)
		self.currPlot = plt.imshow(N.log(self.intensities[:,:,plotSection]), interpolation='nearest')
		self.colorbar = plt.colorbar(self.currPlot, pad=0.02)
		plt.show()

	def showLogIntensitySlices(self):
		"""
		Shows Fourier intensities as an array of sequential, equal-sized 2D sections.
		Maximum intensities set to logarithm of maximum intensity in 3D Fourier volume.
		"""
		subplotlen = int(N.ceil(N.sqrt(len(self.intensities))))
		maxLogIntens = N.log(self.intensities.max())
		fig = plt.figure(figsize=(13.5, 9.5))
		for i in range(len(self.intensities)):
			ax = fig.add_subplot(subplotlen, subplotlen, i+1)
			ax.imshow(N.log(self.intensities[:,:,i]), vmin=0, vmax=maxLogIntens, interpolation='nearest')
			ax.set_xticks(())
			ax.set_yticks(())
			ax.set_title('%d'%(i-self.qmax), color='white', position=(0.85,0.))
		plt.show()
		
		
	def writeSupportToFile(self, filename="support.dat"):
		header = "%d\t%d\n" % (self.qmax, len(self.supportPositions))
		f = open(filename, 'w')
		f.write(header)
		for i in self.supportPositions:
			text = "%d\t%d\t%d\n" % (i[0], i[1], i[2])
			f.write(text)
		f.close()
		

	def writeDensityToFile(self, filename="density.dat"):		
		f = open(filename, "w")
		self.density.tofile(f, sep="\t")
		f.close()
		

	def writeDetectorToFile(self, filename="detector.dat"):
		header = "%d\t%d\t%d\n" % (self.qmax, len(self.detector), len(self.beamstop))
		f = open(filename, 'w')
		f.write(header)
		for i in self.detector:
			text = "%e\t%e\t%e\n" % (i[0], i[1], i[2])
			f.write(text)
		for i in self.beamstop:
			text = "%d\t%d\t%d\n" % (i[0], i[1], i[2])
			f.write(text)
		f.close()

		
	def writeIntensitiesToFile(self, filename="intensity.dat"):
		f = open(filename, "w")
		self.intensities.tofile(f, sep="\t")
		f.close()
		
		
	def writeAllOuputToFile(self, supportFileName="support.dat", densityFileName="density.dat", detectorFileName="detector.dat", intensitiesFileName="intensity.dat"):
		"""
		Convenience function for writing output
		"""
		self.writeSupportToFile(supportFileName)
		self.writeDensityToFile(densityFileName)
		self.writeDetectorToFile(detectorFileName)
		self.writeIntensitiesToFile(intensitiesFileName)
		
