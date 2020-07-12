from PIL import Image
from decimal import Decimal

import sympy as sym
from sympy import *

import os
import math
import random
import shutil

sdl = float(input("Length of simulation domain in centimeters: "))
cupd = float(input("Separation between the brim of the detector and the flow field inlet in centimeters: "))
cupor = float(input("Outer radius of the detector in centimeters: "))
ref = float(input("Thickness of the cup (Outer radius - Inner radius) in cm to be used as reference for evaluating pixel length: "))

print("\n The release plane for MSPs is a vertical line segment. User entry for the horizontal distance between the plane and the cup and the Y co-ordinates of the starting (lowest) and ending point (highest) of the line segment is required. \n")

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
datafile = open(os.path.join(__location__, 'Data.txt'),'a')

shutil.copy2(os.path.join(__location__, 'Contours\\P.JPG'),os.path.join(__location__, 'Track\\TrackP.JPG'))

def minsum(field):
	legend = Image.open(os.path.join(__location__, 'Legends\\'+'legend'+field+'.JPG'),'r')
	return min(map(math.fsum, list(legend.getdata())))

def minavg(field):
	legend = Image.open(os.path.join(__location__, 'Legends\\'+'legend'+field+'.JPG'),'r')
	return min(map(math.fsum, list(legend.getdata()))/3)

legendP = Image.open(os.path.join(__location__, 'Legends\\legendP.JPG'),'r')
pix = list(legendP.getdata())
flag = 0
Pminij = pix[0][0]+pix[0][1]
Pminjk = pix[0][1]+pix[0][2]
Pminik = pix[0][0]+pix[0][2]
while flag < len(pix) :
	if((pix[flag][0]+pix[flag][1]) < Pminij) :
		Pminij = pix[flag][0]+pix[flag][1]
	if((pix[flag][1]+pix[flag][2]) < Pminjk) :
		Pminjk = pix[flag][1]+pix[flag][2]
	if((pix[flag][0]+pix[flag][2]) < Pminik) :
		Pminik = pix[flag][0]+pix[flag][2]
	flag = flag + 1
legendP.close()
	
def origin(field, sdl, cupd, cupor):
	contour = Image.open(os.path.join(__location__,'Contours\\'+field+'.JPG'),'r')
	pix=contour.load()
	sdb = (sdl/contour.width)*(contour.height)    #Breadth of simulation domain
	f = 0.05
	while f <= 1 :
		mini = int(((cupd*(1-f))/sdl)*contour.width)
		maxi = int(((cupd*(1+f))/sdl)*contour.width)
		minj = int((1-((cupor*(1+f))/sdb))*(contour.height))
		maxj = int((1-((cupor*(1-f))/sdb))*(contour.height))
		if(minj < 0):
			minj = 0
		if(maxi > contour.width-1):
			maxi = contour.width-1
		for j in range(minj,maxj):
			for i in range(mini,maxi):
				if ((math.fsum(pix[i,j])) < 150):
					for k in range(0,i):
						pix[k,j] = (0,0,0)
					contour.save('Check\\'+field+'mod.JPG')
					return [i,j]
		f += 0.05
	contour.close()

originP = origin('P',sdl,cupd,cupor)
originT = origin('T',sdl,cupd,cupor)
originU = origin('U',sdl,cupd,cupor)
originV = origin('V',sdl,cupd,cupor)
originN = origin('N',sdl,cupd,cupor)

print(originP,originT,originU,originV,originN)

rd = float(input("Distance between release plane and the brim of the cup (in cm): "))
rsy = float(input("Release surface starting point Y co-ordinate (in cm): "))
rey = float(input("Release surface ending point Y co-ordinate (in cm): "))

contour = Image.open(os.path.join(__location__,'Contours\\P.JPG'),'r')
pix=contour.load()

trackcontour = Image.open(os.path.join(__location__,'Track\\TrackP.JPG'),'r')
pixels = trackcontour.load()

#i = originP[0]
#j = originP[1]

#print(i,j, pix[i,j])
#print(i-1,j, pix[i-1,j])
#print(i,j-1, pix[i,j-1])
#print(i-1,j-1, pix[i-1,j-1])

#while math.fsum(pix[i,j]) < 100:
#	j = j+1
#	print(pix[i,j], i, j, j-originP[1] )

#i = originP[0]
#j = originP[1]

#while math.fsum(pix[i,j]) < 100:
#	i = i+1
#	print(pix[i,j], i, j, i-originP[0])

#pixsize = ref/((y-originP[1]-1.17))
pixsize = float(ref/15)

#for j in range(int(contour.height-rey*((contour.height-originP[1])/cupor)),int(contour.height-rsy*((contour.height-originP[1])/cupor))):

for j in range(int(contour.height-(rey/pixsize)),int(contour.height-(rsy/pixsize))):
	pixels[originP[0]-int(rd/pixsize),j] = (0,0,0)
	#if(j == int(contour.height-rsy*((contour.height-originP[1])/cupor))-1):
	if(j == int(contour.height-(rsy/pixsize))-1):
		trackcontour.show()
trackcontour.save('Track\\TrackP.JPG')
trackcontour.close()
#print(int(contour.height-rsy*((contour.height-originP[1])/cupor)), int(contour.height-rey*((contour.height-originP[1])/cupor)), originP[0]-int((rd/sdl)*contour.width))

#while math.fsum(pix[i,j]) > 100:
#	i = i+1
#	print(pix[i,j], i, j, (i-originP[0])/((contour.info['dpi'])[0]) )

#while math.fsum(pix[i-1,j]) > 100:
#	j = j+1
#	print(pix[i-1,j], i-1, j, (i-1-originP[0])/((contour.info['dpi'])[0]) )

#i = originP[0]
#j = originP[1]

#while math.fsum(pix[i,j]) < 100:
#	i = i+1
#	print(pix[i,j], i, j, (i-originP[0])/((contour.info['dpi'])[0]) )

#print(int(contour.height-rsy*((contour.height-originP[1])/cupor)), int(contour.height-rey*((contour.height-originP[1])/cupor)), originP[0]-int((rd/sdl)*contour.width))

#print(contour.height-int(rsy/pixsize), contour.height-int(rey/pixsize), originP[0]-int(rd/pixsize))

minP = float(input("Enter the minimum value of pressure: "))
maxP = float(input("Enter the maximum value of pressure: "))
minT = float(input("Enter the minimum value of temperature: "))
maxT = float(input("Enter the maximum value of temperature: "))
minU = float(input("Enter the minimum value of x-velocity: "))
maxU = float(input("Enter the maximum value of x-velocity: "))
minV = float(input("Enter the minimum value of y-velocity: "))
maxV = float(input("Enter the maximum value of y-velocity: "))
minN = float(input("Enter the minimum value of Number Density: "))
maxN = float(input("Enter the maximum value of Number Density: "))

def value(min, max, j, field):
	contour = Image.open(os.path.join(__location__,'Contours\\'+field+'.JPG'),'r')
	cell = list(contour.getdata())[j]
	legend = Image.open(os.path.join(__location__,'Legends\\'+'legend'+field+'.JPG'),'r')
	pix_legend = list(legend.getdata())
	weight0 = cell[0]/math.fsum(cell)
	weight1 = cell[1]/math.fsum(cell)
	weight2 = cell[2]/math.fsum(cell)
	index = 0
	for i in pix_legend[0:legend.width]:
		dif0 = abs(cell[0]-i[0])
		dif1 = abs(cell[1]-i[1])
		dif2 = abs(cell[2]-i[2])
		#total = dif0 + dif1 + dif2
		total = (dif0*weight0)+(dif1*weight1)+(dif2*weight2)
		if(pix_legend.index(i) == 0):
			mintotal = total
		if (mintotal > total):
			index = pix_legend.index(i)
			mintotal = total
			#print(mintotal, total, index, math.fsum(i))
		#print(i, pix_legend.index(i), total)
	print(mintotal, index, cell)
	contour.close()
	legend.close()
	#if(((max-min)*(index/legend.width)) < min):
	#	return min
	#if(((max-min)*(index/legend.width)) > max):
	#	return max	
	return (min+((max-min)*(index/legend.width)))
	
#print(value(minP, maxP, 98189, 'P'))

def totalweight(i,j,field):
	contour = Image.open(os.path.join(__location__,'Contours\\'+field+'.JPG'),'r')
	cell = list(contour.getdata())[(contour.width*j)+i]
	legend = Image.open(os.path.join(__location__,'Legends\\'+'legend'+field+'.JPG'),'r')
	pix_legend = list(legend.getdata())
	weight0 = cell[0]/math.fsum(cell)
	weight1 = cell[1]/math.fsum(cell)
	weight2 = cell[2]/math.fsum(cell)
	for i in pix_legend[0:legend.width]:
		dif0 = abs(cell[0]-i[0])
		dif1 = abs(cell[1]-i[1])
		dif2 = abs(cell[2]-i[2])
		total = (dif0*weight0)+(dif1*weight1)+(dif2*weight2)
		if(pix_legend.index(i) == 0):
			mintotal = total
		if (mintotal > total):
			mintotal = total
	contour.close()
	legend.close()
	return mintotal

#for i in range(0,contour.width):
#	print(i, pix[i,169], totalweight(i,169,'P'))

#for i in range(0,contour.width):
#	print(i, pix[i,185], totalweight(i,185,'P'))

def Track(start, end):                             # The code within the function Track() is the Bresenham's line algorithm which helps draw a line between two pixels.
    # Setup initial conditions                     # Source/Reference from which the Bresenham's line algorithm was taken : https://gist.github.com/goldsmith/5b7d874d9b3e0dc1779424a73e0ace7a
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

mg = float(input("Enter the mass of the gas molecule constituting the flow field (in amu): "))*1.66054*math.pow(10,-27)
rg = float(input("Enter the radius of the gas molecule constituting the flow field (in nm): "))
rp = float(input("Enter the radius of the MSP (in nm): "))
dp = float(input("Enter the density of the MSP (in grams per cubic centimeter): "))
mp = ((4/3)*(math.pi)*(math.pow(rp,3))*dp*(math.pow(10,-24)))
print("Mass of the MSP: ",mp," kg")
vxp = float(input("Enter the x-velocity of the free-stream: "))
vyp = float(input("Enter the y-velocity of the free-stream: "))
vzp = float(input("Enter the z-velocity of the free-stream: "))
np = int(input("Number of MSPs to be released from the release plane: "))
div = abs(rsy-rey)/np              # Length of the release surface divided by number of MSPs. Tells you the separation between each MSP
cellsize = float(ref*0.01/15)
x = sym.symbols('x')
cons = sym.symbols('cons')
I = sym.integrate(((x**2)*exp((-1)*((x-cons)**2))) - exp((-1)*((x+cons)**2)),(x,0,oo))
datafile.write('Mass of gas molecule : '+str(mg)+' kg \nRadius of gas molecule : '+str(rg)+' nm \n Radius of MSP : '+str(rp)+' nm \nDensity of MSP : '+str(dp)+' g/cm3 \nMass of MSP : '+str(mp)+' kg \nFree stream X-Velocity : '+str(vxp)+' m/s \nFree stream Y-Velocity : '+str(vyp)+' m/s \nFree stream Z-Velocity : '+str(vzp)+' m/s \nNumber of MSPs released : '+str(np)+'\nSeparation between each MSP : '+str(div)+' cm \nCell Size : '+str(cellsize)+' m \nSolution of Integral involved in program : '+str(I)+'\n\nCalculation Begins!\n\n')
contourT = Image.open(os.path.join(__location__,'Contours\\T.JPG'),'r')
contourU = Image.open(os.path.join(__location__,'Contours\\U.JPG'),'r')
contourV = Image.open(os.path.join(__location__,'Contours\\V.JPG'),'r')
contourN = Image.open(os.path.join(__location__,'Contours\\N.JPG'),'r')
mini = originP[0] - min(originP[0],originN[0],originT[0],originU[0],originV[0])
minj = originP[1] - min(originP[1],originN[1],originT[1],originU[1],originV[1])
maxi = min(contour.width-originP[0],contourN.width-originN[0],contourT.width-originT[0],contourU.width-originU[0],contourV.width-originV[0]) + originP[0]
maxj = min(contour.height-originP[1],contourN.height-originN[1],contourT.height-originT[1],contourU.height-originU[1],contourV.height-originV[1]) + originP[1]
#num = sdl*(math.sqrt(2)*maxN*math.pow((rp+rg)*math.pow(10,-9),2))*2  # the number of random numbers to generate for each set = (simulation domain length/ Mean free path (at max number density)) * 2
#print("Number of random numbers for each set: ",num)

trackcontour = Image.open(os.path.join(__location__,'Track\\TrackP.JPG'),'r')
pixels = trackcontour.load()

for n in range(0,np):
	if(rsy > rey):
		yp = (rsy - (n*div))*0.01
	else:
		yp = (rey - (n*div))*0.01
	xp = (cupd - rd)*0.01
	vxpi = vxp                # saving the initial velocity components of the MSPs as they can be re-assigned for the next MSP tracing
	vypi = vyp
	vzpi = vzp    
	i = int(xp/cellsize)
	j = contour.height-int(yp/cellsize) 
	print(xp,yp,i,j) 
	count = 1

	rnd = []
	k = 0

	print("Generating random numbers for each field set......")
	while k < 9:
		z = 0
		rnd.append([])
		random.seed(k+(9*n))
		while z < 10000: 
			rnd[k].append(random.random())
			z = z+1
		k = k+1

	flag = 0

	while ((math.fsum(pix[i,j]) >= 100) & (count < 10000) & (flag == 0)):
#		print("minimum i: ",mini,"minimum j: ",minj,"maximum i: ",maxi,"maximum j: ",maxj)
		print("Tracking MSP no: ",n+1," Walks: ",count)
		datafile.write('\nTracking MSP no: '+str(n+1)+' Walks: '+str(count)+'\n')
#		sigma = math.sqrt((1.38064852*math.pow(10,-23))*(value(minT, maxT, ((i-originP[0]+originT[0])+1)+((j-originP[1]+originT[1])*contourT.width), 'T')/(mg)*((1/2)-(3/(4*math.pi))))
#		print("Sigma: ",sigma)
		datafile.write('X co-ordinate: '+str(xp)+'\n'+'Y co-ordinate: '+str(yp)+'\n')
		datafile.write('Pixel X co-ordinate: '+str(i)+'\n'+'Pixel Y co-ordinate: '+str(j)+'\n')
		Temp = value(minT, maxT, ((i-originP[0]+originT[0])+1)+((j-originP[1]+originT[1])*contourT.width), 'T')
		NumD = value(minN, maxN, ((i-originP[0]+originN[0])+1)+((j-originP[1]+originN[1])*contourN.width), 'N')
		XVel = value(minU, maxU, ((i-originP[0]+originU[0])+1)+((j-originP[1]+originU[1])*contourU.width), 'U')
		YVel = value(minV, maxV, ((i-originP[0]+originV[0])+1)+((j-originP[1]+originV[1])*contourV.width), 'V')
		print("Cell Number Density: ",NumD)
		datafile.write('Cell Number Density: '+str(NumD)+' particles/m3 \n')
		print("Cell Temperature: ", Temp)
		datafile.write('Cell Temperature: '+str(Temp)+' K \n')
		print("Cell x-velocity: ", XVel)
		datafile.write('Cell x-velocity: '+str(XVel)+' m/s \n')
		print("Cell y-velocity: ", YVel)
		datafile.write('Cell y-velocity: '+str(YVel)+' m/s \n')
		Vgth = (math.sqrt((8*(1.38064852*math.pow(10,-23))*Temp)/((math.pi)*mg)))	# Mean thermal speed of the air molecule at a specific point
		print("Vgth: ",Vgth)
		datafile.write('Vgth: '+str(Vgth)+'\n')
		sign = 1
		if(rnd[6][0]<0.5):
			sign = -1   # If random number is less than 0.5 then the sign for one component of the velocity is minus. Which means along the negative direction. 
		Vgx = XVel +  (math.sqrt((2*(1.38064852*math.pow(10,-23))*Temp)/mg)*sym.erfinv(rnd[0][0])*sign)    # Total x-velocity of the molecule w.r.t. the cup at a point = x component of freestream velocity at that point + x component of the thermal speed of the gas molecule produced by the expression v = sqrt(2*k*T/m)*inverseerrorfunction(R) where R is a random number.
		print("Vgx: ",Vgx)
		datafile.write('Vgx: '+str(Vgx)+' , Associated random number: '+str(rnd[0][0])+'\n')
		sign = 1
		if(rnd[7][0]<0.5):
			sign = -1
		Vgy = YVel + (math.sqrt((2*(1.38064852*math.pow(10,-23))*Temp)/mg)*sym.erfinv(rnd[1][0])*sign)
		print("Vgy: ",Vgy)
		datafile.write('Vgy: '+str(Vgy)+' , Associated random number: '+str(rnd[1][0])+'\n')
		sign = 1
		if(rnd[8][0]<0.5):
			sign = -1
		Vgz = (math.sqrt((2*(1.38064852*math.pow(10,-23))*Temp)/mg)*sym.erfinv(rnd[2][0])*sign)
		print("Vgz: ",Vgz)
		datafile.write('Vgz: '+str(Vgz)+' , Associated random number: '+str(rnd[2][0])+'\n')
		Vmx = ((mp*vxp)+(mg*Vgx))/(mg+mp)   # x-velocity of the centre of mass of the MSP and the colliding gas molecule
		print("Vmx: ",Vmx)
		datafile.write('Vmx: '+str(Vmx)+'\n')
		Vmy = ((mp*vyp)+(mg*Vgy))/(mg+mp)
		print("Vmy: ",Vmy)
		datafile.write('Vmy: '+str(Vmy)+'\n')
		Vmz = ((mp*vzp)+(mg*Vgz))/(mg+mp)
		print("Vmz: ",Vmz)
		datafile.write('Vmz: '+str(Vmz)+'\n')
		theta = math.acos((2*(rnd[3][0]))-1)
		print("theta: ",theta)
		datafile.write('theta: '+str(theta)+' , Associated random number: '+str(rnd[3][0])+'\n')
		phi = 2*(math.pi)*(rnd[4][0])
		print("phi: ",phi)
		datafile.write('phi: '+str(phi)+' , Associated random number: '+str(rnd[4][0])+'\n')
		vrel = math.sqrt(math.pow(Vgx-vxp,2)+math.pow(Vgy-vyp,2)+math.pow(Vgz-vzp,2)) # relative speed between the collision partners
		print("vrel: ",vrel)
		datafile.write('vrel: '+str(vrel)+'\n')
		vrelx = vrel*math.sin(theta)*math.cos(phi)  # x-component of relative velocity after collision
		print("vrelx: ",vrelx)
		datafile.write('vrelx: '+str(vrelx)+'\n')
		vrely = vrel*math.sin(theta)*math.sin(phi)
		print("vrely: ",vrely)
		datafile.write('vrely: '+str(vrely)+'\n')
		vrelz = vrel*math.cos(theta)
		print("vrelz: ",vrelz)
		datafile.write('vrelz: '+str(vrelz)+'\n')
		vxpc = Vmx - ((mg/(mg+mp))*vrelx)            # x-component of post collision velocity          
		print("vxpc: ",vxpc)
		datafile.write('vxpc: '+str(vxpc)+'\n')
		vypc = Vmy - ((mg/(mg+mp))*vrely)
		print("vypc: ",vypc)
		datafile.write('vypc: '+str(vypc)+'\n')
		vzpc = Vmz - ((mg/(mg+mp))*vrelz)
		print("vzpc: ",vzpc)
		datafile.write('vzpc: '+str(vzpc)+'\n')
		#a = (2*vrel)/(math.sqrt(math.pi)*Vgth)
		a = (2*math.sqrt(math.pow(XVel-vxp,2)+math.pow(YVel-vyp,2)+math.pow(vzp,2)))/(math.sqrt(math.pi)*Vgth)
		print("a: ",a)
		datafile.write('a: '+str(a)+'\n')
		datafile.write('I: '+str(float(I.subs(cons,a)))+'\n')
		vrel_m = (Vgth*float(I.subs(cons,a)))/(2*a) # Mean relative speed
		print("vrel_m: ",vrel_m)
		datafile.write('vrel_m: '+str(vrel_m)+'\n')
		tcol = 1/((math.pi)*(math.pow((rp+rg)*math.pow(10,-9),2))*NumD*abs(vrel_m))   # Mean collision time
		print("tcol: ",tcol)
		datafile.write('tcol: '+str(tcol)+'\n')
		dt = (-1)*log(rnd[5][0])*tcol    # timestep
		print("dt: ",dt)
		datafile.write('dt: '+str(dt)+'\n')
		xpi = xp + (vxpc*dt)			# projected of position co-ordinates to half plane
		print("xpi: ",xpi)
		ypi = math.sqrt(pow(yp+(vypc*dt),2)+pow(vzpc*dt,2))
		print("ypi: ",ypi)
		datafile.write('Updated X co-ordinate: '+str(xpi)+'\n'+'Updated Y co-ordinate: '+str(ypi)+'\n')
		#if((i!=int(xp/cellsize)) & (j!=contour.height-int(yp/cellsize))):
		for pos in Track((i,j),(int(xpi/cellsize),contour.height-int(ypi/cellsize))):
			if ((pos[0] > maxi) | (pos[0] < mini) | (pos[1] > maxj) | (pos[1] < minj) | (pos[0] >= contour.width) | (pos[1] >= contour.height) | (math.fsum(pix[pos[0],pos[1]]) <= 100)):
				flag = 1
				break
			pixels[pos[0],pos[1]] = (0,0,0)
			#trackcontour.save('Track\\TrackP.JPG')
		vxp = vxpc                             # re-projected of velocities to half plane
		vyp = (((yp+(vypc*dt))*vypc)+(vzpc*vzpc*dt))/ypi
		vzp = (((yp+(vypc*dt))*vzpc)+(vypc*vzpc*dt))/ypi
		xp = xpi
		yp = ypi
		i = int(xp/cellsize)
		print("Latest i: ",i)
		j = contour.height-int(yp/cellsize)
		print("Latest j: ",j)
		datafile.write('Updated Pixel X co-ordinate: '+str(i)+'\n'+'Updated Pixel Y co-ordinate: '+str(j)+'\n')
		if ((i > maxi) | (i < mini) | (j > maxj) | (j < minj)):
			break
		#if((i>=contour.width-1) | i<=1):
		#	break
		#if((j>=contour.height-1) | j<=1):
		#	break
		print("RGB values at latest Pixel: ",pix[i,j])
		datafile.write('RGB values at latest Pixel: '+str(pix[i,j])+'\n')
		set = 0
		while set < 9:     # deleting the first element of the list of random number in the each set since it already has been assigned. That way each time the simulation progresses the program doesn't have to search for the nth random number thereby avoiding time wastage
			rnd[set].pop(0)
			set = set + 1
		count = count + 1
	trackcontour.show()
	if(n == np -1):
		trackcontour.save('Track\\TrackP.JPG')
	vxp = vxpi
	vyp = vypi
	vzp = vzpi
	del rnd 

contour.close()
contourT.close()
contourU.close()
contourV.close()
contourN.close()
trackcontour.close()
datafile.close()
