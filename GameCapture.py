from PIL import ImageGrab
import cv2
import numpy as np
from matplotlib import pyplot as plt


#===Your Capture Card======#
#0 is your default webcam, so if your cam is the feed, change to 1 for the secondary camera#
camera_port = 1
cam = cv2.VideoCapture(camera_port)

#===Templates for image comparison===#
template = cv2.imread('temps/temp.png',0)
gb_template = cv2.imread('temps/gb.png',0)
w, h = template.shape[::-1]
w, h = gb_template.shape[::-1]

#===Set up Vars for Screen===#
font = cv2.FONT_HERSHEY_SIMPLEX
canvas = np.zeros((480, 904, 3), np.uint8)

#===Banana Arrays===#
#0=Japes #
current_level =0
level_array=["Jungle Japes","Angry Aztec","Frantic Factory","Gloomy Galleon","Fungi Forest","Crystal Caves","Creepy Castle"]
dk_banana=['0','0','0','0','0','0','0']
diddy_banana =['0','0','0','0','0','0','0']
lanky_banana =['0','0','0','0','0','0','0']
tiny_banana  =['0','0','0','0','0','0','0']
chunky_banana=['0','0','0','0','0','0','0']
totalGB='0  '

logo = cv2.imread('temps\logo.png')


def test_num(img):
	this_number =''
	
	mask = np.zeros(img.shape[:2],np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)

	rect = (1,1,50,64)
	cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	img = img*mask2[:,:,np.newaxis]
	cv2.imwrite('kba.png',img)
	for x in range(0, 10):		
		num = cv2.imread('temps/numbers/'+str(x)+'.png',0)
		w, h = num.shape[::-1]
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		res = cv2.matchTemplate(img_gray,num,cv2.TM_CCOEFF_NORMED)
		threshold = 0.85		
		loc = np.where( res >= threshold)
		if zip(*loc[::-1]):
			for pt in zip(*loc[::-1]):
				this_number = str(x)
											
	return this_number			
		

				
def show_webcam():
	global current_level
	while True:
		ret_val, img = cam.read()

		#===Keyboard Commands========
		ch = cv2.waitKey(1)
		if  ch == 27:
			break  # esc to quit
		if ch == ord('a'): 
			cv2.imwrite('screenshot.png',img)
		if ch == 2555904:
			current_level+=1
		if ch == 2424832:
			current_level-=1
		
		if current_level<0:
			current_level=0
		if current_level>6:
			current_level=6
		check_colored_banana(img)
		check_golden_banana(img)
		
		cv2.rectangle(canvas, (0,0), (200,480), (0,0,0), -1)
		#canvas[0: 60, 0: 95] = logo
		cv2.putText(canvas,'Donkey ='+dk_banana[current_level],(10,90), font, 0.8,(255,255,255),2)
		cv2.putText(canvas,'Diddy ='+diddy_banana[current_level],(10,120), font, 0.8,(255,255,255),2)
		cv2.putText(canvas,'Larry ='+lanky_banana[current_level],(10,150), font, 0.8,(255,255,255),2)
		cv2.putText(canvas,'Tiny =' +tiny_banana[current_level],(10,180), font, 0.8,(255,255,255),2)	
		cv2.putText(canvas,'Chunky ='+chunky_banana[current_level],(10,210), font, 0.8,(255,255,255),2)			
		cv2.putText(canvas,'GB ='+totalGB,(10,430), font, 0.8,(255,255,255),2)	
		cv2.putText(canvas,level_array[current_level],(10,460), font, 0.6,(255,255,255),2)	
		canvas[0: 480, 200: 904] = img	
		#canvas[200: 241, 0: 35] = cv2.imread('numbers/0.png')	
		cv2.imshow('Counter App', canvas)	
	cv2.destroyAllWindows()
	

def check_golden_banana(img):
	global totalGB
	x=200
	y=330
	small_img = img[330: 460, 200: 370]
	#cv2.rectangle(img, (200,330), (370,460), (0,255,0), 1)	
	img_gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)		
	res = cv2.matchTemplate(img_gray,gb_template,cv2.TM_CCOEFF_NORMED)
	threshold = 0.85
	loc = np.where( res >= threshold)		
	if zip(*loc[::-1]):
		for pt in zip(*loc[::-1]):			
			#cv2.rectangle(img, (x+pt[1],y+pt[0]), (x+pt[1]+65,y+pt[0]+88), (255,0,0), 1)
			#print(pt[1])
			place_hold = list(totalGB)
			final = list(totalGB)
			if pt[1] <20:
				#=====Menu====================
				roi = img[y+pt[1]+25:(y+pt[1]+73), (x+pt[0]+60):(x+pt[0]+110)]
				roi2 = img[y+pt[1]+25:(y+pt[1]+73), (x+pt[0]+70):(x+pt[0]+123)]
				roi3 = img[y+pt[1]+25:(y+pt[1]+73), (x+pt[0]+117):(x+pt[0]+155)]
				place_hold[0]  =str(test_num(roi))
				place_hold[1]  =str(test_num(roi2))
				place_hold[2]  =str(test_num(roi3))
				#cv2.rectangle(img, (x+pt[0]+95,y+pt[1]+25), (x+pt[0]+143,y+pt[1]+73), (0,255,0), 1)	
				#cv2.rectangle(img, (x+pt[0]+117,y+pt[1]+25), (x+pt[0]+175,y+pt[1]+73), (255,0,0), 1)					
							
			else:
				#=====Gameplay====================
				roi = img[y+pt[1]+25:(y+pt[1]+73), (x+pt[0]+70):(x+pt[0]+120)]
				roi2 = img[y+pt[1]+25:(y+pt[1]+73), (x+pt[0]+105):(x+pt[0]+153)]
				roi3 = img[y+pt[1]+25:(y+pt[1]+73), (x+pt[0]+147):(x+pt[0]+185)]
				place_hold[0]  =str(test_num(roi))
				place_hold[1]  =str(test_num(roi2))	
				place_hold[2]  =str(test_num(roi3))
				#cv2.rectangle(img, (x+pt[0]+105,y+pt[1]+25), (x+pt[0]+153,y+pt[1]+73), (0,255,0), 1)	
				#cv2.rectangle(img, (x+pt[0]+140,y+pt[1]+25), (x+pt[0]+183,y+pt[1]+73), (255,0,0), 1)					
			
			#print(str(place_hold[0])+str(place_hold[1])+str(place_hold[2]))
			if place_hold[0] is not '':
				final[0] = place_hold[0]				
			if place_hold[1] is not '':
				final[1] = place_hold[1]
			if place_hold[2] is not '':
				final[2] = place_hold[2]
			totalGB = "".join(final)
			break;			

	

def check_colored_banana(img):
	small_img = img[0: 150, 0: 100]
	x=0	
	y=0	
	img_gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)		
	res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
	threshold = 0.8
	loc = np.where( res >= threshold)
	if zip(*loc[::-1]):
		for pt in zip(*loc[::-1]):
			color_b = ''
			#print(str(pt[0])+","+str(pt[1]))
			if pt[1] <25:
				#=====Menu====================
				roi = img[25:90, 85:125]
				roi2 =img[25:90, 112:175]
				color_b +=str(test_num(roi))
				color_b +=str(test_num(roi2))
				#cv2.rectangle(img, (85,25), (125,90), (0,255,0), 1)
				#cv2.rectangle(img, (115,25), (175,90), (0,0,255), 1)
				x=70
				y=40
			else:
				#=====Gameplay====================
				roi = img[pt[1]+10:(pt[1]+75), (pt[0]+65):(pt[0]+111)]
				roi2 =img[pt[1]+10:(pt[1]+75), (pt[0]+100):(pt[0]+153)]
				color_b +=str(test_num(roi))
				color_b +=str(test_num(roi2))
				#cv2.rectangle(img, (pt[0]+65,pt[1]), (pt[0]+110,pt[1]+75), (0,255,0), 1)
				#cv2.rectangle(img, (pt[0]+100,pt[1]), (pt[0]+150,pt[1]+75), (0,0,255), 1)	
				x=75
				y=65
			#=====Check to see which kong's banana it is by color==========
			print(small_img[40,70]) 
			#cv2.imwrite('red.png',small_img)75 65
			#red
			mask = cv2.inRange(small_img[y,x],np.array([0, 0, 190], dtype = "uint8") ,np.array([50, 50, 255], dtype = "uint8") )				
			if mask[0] == 255 and mask[1] == 255 and mask[2] == 255 :
				diddy_banana[current_level] =''
				diddy_banana[current_level] += color_b;
				cv2.imwrite('red.png',small_img)		
			#yellow
			mask = cv2.inRange(small_img[y,x],np.array([0, 90, 140], dtype = "uint8") ,np.array([20, 200, 255], dtype = "uint8") )
			#print(mask)			
			if mask[0] == 255 and mask[1] == 255 and mask[2] == 255 :
				dk_banana[current_level] =''
				dk_banana[current_level] += color_b;
			#blue
			mask = cv2.inRange(small_img[y,x],np.array([120, 90, 0], dtype = "uint8") ,np.array([230, 200, 100], dtype = "uint8") )
			if mask[0] == 255 and mask[1] == 255 and mask[2] == 255 :
				lanky_banana[current_level] =''
				lanky_banana[current_level] += color_b;			
			#purple
			mask = cv2.inRange(small_img[y,x],np.array([150, 15, 120], dtype = "uint8") ,np.array([230, 100, 220], dtype = "uint8") )
			print(mask)
			if mask[0] == 255 and mask[1] == 255 and mask[2] == 255 :
				tiny_banana[current_level] =''
				tiny_banana[current_level] += color_b;			
			#Green
			mask = cv2.inRange(small_img[y,x],np.array([0, 130, 23], dtype = "uint8") ,np.array([60, 210, 90], dtype = "uint8") )
			#print(mask)
			if mask[0] == 255 and mask[1] == 255 and mask[2] == 255 :
				chunky_banana[current_level] =''
				chunky_banana[current_level] += color_b;	
			break;	
	
	
	
def main():
	show_webcam()

if __name__ == '__main__':
	main()
 