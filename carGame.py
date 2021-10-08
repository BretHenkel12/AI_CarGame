#Made by Bret Henkel
#Fall 2021
#Built upon machineLearningLib2, a module also made by Bret


#Import all necessary libraries/modules
import pygame
import numpy as np
import pandas as pd
from colors import *
import sys
import cv2 as cv
import machineLearningLib2 as ML
import math
import pickle


#Game settings, general
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
spawnZone = True #Prevents cars from spawning on top of each other
player = False #Allows a player to drive around a car
load = False #Loads a previous file if True
resume = False #Resumes from a previous save point if True
yellowLines = True #Turns on and off the yellow lines emitting from each car
limitedFrameRate = True #Limits the frame rate to a maximum of 60 fps


#Game settings, fine tuning
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
numCars = 100 #Max number of cars
anglesDeg = pd.Series([-90, -40, -15, -5, 0, 5, 15, 40, 90]) #Angles that measurement lines emit at
lineMax = 300 #max length in pixels of yellow lines
turningCoeff = 40 #Helps determine the max turning rate of the car
steeringFactor = 10 #Determines the how much the steering affects the turning rate
limitTurningRate = 0 #The maximum turning rate for the cars, will be based on speed
maxAcceleration = 0.25 #The max acceleration of the car
maxSpeed = 8 #The max speed of the car
bestCarsRemembered = 5 #Amount of top cars remembered from all generations, these cars determine future gens
startingAngle = 170 #Starting angle for the cars
startingPos = (740,130) #Starting position for the cars


#Basic variables used in calculations and to determine game state
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
if load:
    loadFileName = input("Save File Name? \n") #Gets the file name from user
if resume:
    resumeFileName = input("Resume File Name? \n") #Gets the file name from user
numberOfInputs = len(anglesDeg) + 1
dim_array = [numberOfInputs, 8, 12, 10, 2]
carCount = 0
parentNumber = 0
initialLoop = True
carNumber = 0
generationCount = 0
reducingSigma = True
ultraReducingSigma = True
finished = 0
finishedCount = 0
playerBest = []
pygame.init()
screen = pygame.display.set_mode((1400, 700))
angles = anglesDeg * np.pi / 180
lapFinished = False
carImage = pygame.image.load("C:/Users/breth/Documents/Programming/Python/MachineLearning/carGame/MarcoCar.png").convert_alpha()
cars = pygame.sprite.Group()
state = np.array([0,0,0,0,0])
trials = 0
exit = False
bestCars = []    #[#car number, #parent number, #checkpoints, #checkpoint list, #np_weights, #np_biases]
oldBestCars = []
for i in range(bestCarsRemembered):
    bestCars.append([0,0,0,[],[],[]])
    oldBestCars.append([0,0,0,[],[],[]])
scores = 0


#Setting up pygame and the track
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#Reads the track file and creates a mask for the out of bounds
trackCV = cv.imread("C:/Users/breth/Documents/Programming/Python/MachineLearning/carGame/MarcoTrack.png")
track = pygame.image.load("C:/Users/breth/Documents/Programming/Python/MachineLearning/carGame/MarcoTrack.png")
trackMask = cv.inRange(trackCV, (77,148,92), (77,148,92))
trackMask_np = np.array(trackMask)
trackMask_np_bool = trackMask_np == 255
#Sets the mouse as invisible
pygame.mouse.set_visible(False)
#Sets up a clock
clock = pygame.time.Clock()
#Sets caption for window
pygame.display.set_caption('Car Racing')
#Sets up font
pygame.font.init() # you have to call this at the start,
                   # if you want to use this module.
myfont = pygame.font.SysFont('Comic Sans MS', 30)


#The class for each car, holds both properties and functions
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
class Car(pygame.sprite.Sprite):
    def __init__(self, weights, biases, carPicture, number, parent):
        super().__init__()
        self.pos = carPicture.get_rect()
        #Determines if the car should be removed from the sprites list
        self.dead = False
        #An ID number for the car
        self.number = number
        self.angle = 0
        self.turningRate = 0
        self.maxTurningrate = 0
        #The car has a starting speed of 2
        self.speed = 2
        self.angleR = self.angle * np.pi / 180
        #Saves the weights and biases list for the car
        self.weights = weights
        self.biases = biases
        #Generates the state array and saves it for the car
        state = []
        for i in range(dim_array[0]-1):
            state.append(0)
        self.state = np.array(state)
        #The checkpoint array saves the number of frames to reach each checkpoint
        self.checkPoints = []
        #Saves the frame count for how long the car has been alive
        self.time = 0
        #Saves the ID number of the car that the current car is based off of
        self.parent = parent

    #A function to draw the car to the screen
    def draw(self):
        if self.number == -1:
            #Draws the car as blue if it is the player's car
            rotatedImage = pygame.transform.rotate(playerImage, -int(self.angle))
        else:
            #Draws the car in green if it is an AI car
            rotatedImage = pygame.transform.rotate(carImage, -int(self.angle))
        self.angleR = self.angle * np.pi / 180
        self.pos = rotatedImage.get_rect()
        self.corner = [self.center[0] - self.pos.width / 2, self.center[1] - self.pos.height / 2]
        screen.blit(rotatedImage, self.corner)

    #A function to move the car forward
    def move(self):
        movement = (self.speed * np.cos(self.angleR), self.speed * np.sin(self.angleR))
        self.center = (self.center[0] + movement[0], self.center[1] + movement[1])
        self.time += 1

    #Finds the length of each line emitting from the car and returns an array of the distances
    def drawLines(self):
        returnValue = []
        for i in angles:
            returnValue.append(self.findIntersectionOfLine(self.center[0],self.center[1],i,lineMax))
        returnValue = np.array(returnValue).astype(float)
        return(returnValue)

    #Determines the intersection of each line and draws them to the screen
    def findIntersectionOfLine(self,x_0,y_0,relativeAngle,maxDistance):
        cos = np.cos(self.angleR+relativeAngle)
        sin = np.sin(self.angleR+relativeAngle)
        for i in range(maxDistance):
            x = int(x_0 + i * cos)
            y = int(y_0 + i * sin)
            if (x >= 1000 or x <= -1 or y >= 700 or y <= -1):
                break
            if (trackMask_np_bool[y,x]):
                break
        if yellowLines:
            pygame.draw.line(screen, yellow, (x_0,y_0), (x,y))
        return i

    #Determines if the car center has gone off of the track
    def checkCollision(self):
        try:
            if (trackMask_np_bool[int(self.center[1]),int(self.center[0])]):
                return True
        except:
            print("Failed")
            return True
        return False

    #Adjusts the speed of the car based off of the acceleration
    def adjustSpeed(self,acceleration):
        acceleration = maxAcceleration*acceleration
        self.speed += acceleration
        if self.speed < 0:
            self.speed = 0
        elif self.speed > maxSpeed:
            self.speed = maxSpeed

    #Turns the car based on the turning acceleration
    def turn(self,turningAcceleration):
        self.turningRate = turningAcceleration*steeringFactor
        try:
            self.maxTurningrate = turningCoeff/(self.speed*self.speed)
        except:
            self.maxTurningrate = limitTurningRate
        if self.maxTurningrate > limitTurningRate:
            self.maxTurningrate = limitTurningRate
        if self.turningRate > self.maxTurningrate:
            self.turningRate = self.maxTurningrate
        elif self.turningRate < -1*self.maxTurningrate:
            self.turningRate = -1*self.maxTurningrate
        self.angle += self.turningRate

    #Determines if the car has reached a new checkpoint (white line). If so appends the frame count to get to that
    #checkpoint to the end of the checkpoints list
    #Checkpoint locations are based on their x location with some tolerance due to the speed the cars are traveling
    def checkPos(self):
        global finished
        color = trackCV[int(self.center[1]),int(self.center[0])]
        if color[0] == 255:
            x = self.center[0]
            if abs(x-760)<10:
                if len(self.checkPoints) < 8 and abs(self.angle%360)<89:
                    finished += 1
                    return(True)
            if abs(x-347) < 10:
                if len(self.checkPoints) == 0:
                    self.checkPoints.append(self.time)
            if abs(x-217) < 10:
                if len(self.checkPoints) == 1:
                    self.checkPoints.append(self.time)
            if abs(x-118) < 10:
                if len(self.checkPoints) == 2:
                    self.checkPoints.append(self.time)
            if abs(x-170) < 10:
                if len(self.checkPoints) == 3:
                    self.checkPoints.append(self.time)
            if abs(x-290) < 10:
                if len(self.checkPoints) == 4:
                    self.checkPoints.append(self.time)
            if abs(x-900) < 10:
                if len(self.checkPoints) == 5:
                    self.checkPoints.append(self.time)
            if abs(x-850) < 10:
                if len(self.checkPoints) == 6:
                    self.checkPoints.append(self.time)
            if abs(x-660) < 10:
                if len(self.checkPoints) == 7:
                    self.checkPoints.append(self.time)
            return(False)
        return(False)

    #After a car has died, determines if its score qualifies it as one of the top in its round
    def checkScore(self):
        checkpointsAccomplished = len(self.checkPoints)
        carsScore = [self.number, self.parent, checkpointsAccomplished,self.checkPoints,self.weights,self.biases]
        number = -1
        if carsScore[2] > 0:
            for i in range(len(bestCars)):
                if bestCars[i][2] > 0:
                    if bestCars[i][2] == checkpointsAccomplished:
                        if bestCars[i][3][bestCars[i][2]-1] > carsScore[3][bestCars[i][2]-1]:
                            number = i
                            break
                if bestCars[i][2] < checkpointsAccomplished:
                    number = i
                    break
            if number != -1:
                bestCars.insert(number,carsScore)
                bestCars.pop(-1) #Removes the worst car in bestCars


#General functions for the game
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#Creates a car with a random weight and biase array
def createCar():
    # Gets the random weights and biases arrays
    pd_weights, np_weights, pd_biases, np_biases = ML.getWeightedArrays(dim_array)
    #Initializes the car
    car = Car(np_weights, np_biases, carImage, carNumber, -1)
    #Sets the center and angle of the car
    car.center = startingPos
    car.angle = startingAngle
    #Adds the car to the sprites list
    cars.add(car)

#Creates a car based off of a parent car
def createModdedCar(weights,biases,parent, sigma):
    #Finds weights and biase arrays based off of the parent arrays
    np_weights_adj, np_biases_adj = ML.returnUpdatedArrays(dim_array, weights, biases, sigma)
    #Initializes the car
    car = Car(np_weights_adj, np_biases_adj, carImage, carNumber,parent)
    #Sets the cars center and angle
    car.center = startingPos
    car.angle = startingAngle
    #Adds the car to the cars sprite list
    cars.add(car)

#Function to save the best cars so that they can be reloaded at a later point
def save(obj,filename):
    try:
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, -1)
    except:
        print("failed save")

#Function to save all cars including their current positions so that the game can be resumed
def saveGroup(group,filename):
    with open(filename, 'wb') as output:
        objList = []
        for x in group:
            objList.append(x)
        pickle.dump(objList, output, -1)

#Determines if there are cars in the spawn zone
def carsInSpawnZone():
    for car in cars:
        if car.center[0] < 760 and car.center[0] > 720:
            if car.center[1] < 150 and car.center[1] > 110:
                return True
    return False

#Prints different scores and stats to the screen on the right column
def screenScoreStuff():
    try:
        s1_score = str(oldBestCars[0][3][-1])
    except:
        s1_score = '0'
    s1 = str(oldBestCars[0][2]) + '   ' + str(s1_score) + '   ' + str(oldBestCars[0][0]) + '   ' + str(oldBestCars[0][1])
    score1 = myfont.render(s1, False, (255, 255, 255))
    try:
        s2_score = str(oldBestCars[1][3][-1])
    except:
        s2_score = '0'
    s2 = str(oldBestCars[1][2]) + '   ' + str(s2_score) + '   ' + str(oldBestCars[1][0]) + '   ' + str(oldBestCars[1][1])
    score2 = myfont.render(s2, False, (255, 255, 255))
    try:
        s3_score = str(oldBestCars[2][3][-1])
    except:
        s3_score = '0'
    s3 = str(oldBestCars[2][2]) + '   ' + str(s3_score) + '   ' + str(oldBestCars[2][0]) + '   ' + str(oldBestCars[2][1])
    score3 = myfont.render(s3, False, (255, 255, 255))
    try:
        s4_score = str(oldBestCars[3][3][-1])
    except:
        s4_score = '0'
    s4 = str(oldBestCars[3][2]) + '   ' + str(s4_score) + '   ' + str(oldBestCars[3][0]) + '   ' + str(oldBestCars[3][1])
    score4 = myfont.render(s4, False, (255, 255, 255))
    try:
        s5_score = str(oldBestCars[4][3][-1])
    except:
        s5_score = '0'
    s5 = str(oldBestCars[4][2]) + '   ' + str(s5_score) + '   ' + str(oldBestCars[4][0]) + '   ' + str(oldBestCars[4][1])
    score5 = myfont.render(s5, False, (255, 255, 255))
    try:
        s_length = len(playerBest)
        s = playerBest[-1]
    except:
        s = 0
    playerScore = myfont.render(str(s_length) + '   ' + str(s), False, (255, 255, 255))

    howManyFinished = myfont.render(str(finishedCount), False, (255, 255, 255))

    return(score1,score2,score3,score4,score5,playerScore,howManyFinished)


#Sets up the game
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#Set up the player car
#Generates weights and biases 
p_pd_weights, p_np_weights, p_pd_biases, p_np_biases = ML.getWeightedArrays(dim_array) 
#Finds the image of the car
playerImage = pygame.image.load("C:/Users/breth/Documents/Programming/Python/MachineLearning/carGame/BretCar.png").convert_alpha()
#Creates the car
car1 = Car(p_np_weights,p_np_biases,playerImage,-1,-2)
#Sets the center and angle of the car
car1.center = startingPos
car1.angle = startingAngle

#If the game is set to load, will load the best cars from the saved files
if load:
    loadFileName += '.pickle'
    try:
        with open(loadFileName, 'rb') as inp:
            oldBestCars = pickle.load(inp)
    except:
        None

#If the game is set to resume, will load the car positions and best cars from the saved files
if resume:
    resumeFileName2 = resumeFileName + '2.pickle'
    resumeFileName += '.pickle'
    with open(resumeFileName, 'rb') as data:
        carData = pickle.load(data)
        for car in carData:
            cars.add(car)
    with open(resumeFileName2, 'rb') as inp:
        oldBestCars = pickle.load(inp)


#The main game loop
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
while True:
    #Performs basic screen actions
    screen.fill((0,0,0)) #Fills the screen with black
    screen.blit(track, (0, 0)) #Blits the track to the screen
    try:
        screen.blit(network, (1000,0)) #Blits the network to the screen
    except:
        None
    if limitedFrameRate:
        clock.tick(60) #Limits the flame rate to 60 fps


    generations = myfont.render(str(generationCount), False, (255,255,255))
    score1, score2, score3, score4, score5, playerScore, howManyFinished = screenScoreStuff()


    #Determines what keys have been pressed
    keys = pygame.key.get_pressed()
    if keys[pygame.K_y]: #Turns yellow lines on and off
        if yellowLines:
            yellowLines = False
        else:
            yellowLines = True
        pygame.time.wait(250)
    if keys[pygame.K_r]:
        print("What would you like to name the resume file")
        resumeFile = ''
        resumeFile = input()
        print(resumeFile)
        resumeFile2 = resumeFile + '2.pickle'
        resumeFile = resumeFile + '.pickle'
        saveGroup(cars, resumeFile)
        save(oldBestCars,resumeFile2)
        print("printing cars")
        print(cars)
    elif keys[pygame.K_s]:
        print("What would you like to name the save file?")
        saveFile = ''
        saveFile = input()
        print(saveFile)
        saveFile = saveFile + '.pickle'
        save(oldBestCars, saveFile)
    if exit:
        print()
        print("printing old best cars")
        print(oldBestCars[0][0], oldBestCars[1][0], oldBestCars[2][0], oldBestCars[3][0], oldBestCars[4][0])
        print(oldBestCars[0][1], oldBestCars[1][1], oldBestCars[2][1], oldBestCars[3][1], oldBestCars[4][1])
        print()
        print(oldBestCars)
        break


    #Determines player movement
    if player:
        lapFinished = car1.checkPos() #Determines if the player has finished a lap
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            car1.turn(-1000)
        if keys[pygame.K_RIGHT]:
            car1.turn(1000)
        if keys[pygame.K_UP]:
            car1.adjustSpeed(1)
        if keys[pygame.K_DOWN]:
            car1.adjustSpeed(-1)
        car1.move() #Moves the car
        car1.draw() #Draws the car
        state = car1.drawLines() #Gets the length of the lines emitting from the car
        #If the car has crashed or finished a lap determines if it has a new high score and then
        #resets the car
        if (car1.checkCollision()) | lapFinished:
            if len(car1.checkPoints) > len(playerBest):
                playerBest = car1.checkPoints
            elif len(car1.checkPoints) == len(playerBest):
                try:
                    if playerBest[-1] > car1.checkPoints[-1]:
                        playerBest = car1.checkPoints
                except:
                    None
            #Resets the car
            car1.checkPoints = []
            car1.center = startingPos
            car1.angle = startingAngle
            car1.speed = 2
            car1.time = 0
            lapFinished = False
            p_pd_weights, p_np_weights, p_pd_biases, p_np_biases = ML.getWeightedArrays(dim_array)
            car1.dead = True
            cycles = 0
            state = np.array([0, 0, 0, 0, 0])


    #Check if cars finished a lap
    for car in cars:
        kill = car.checkPos() #Determines if the car has finished a lap
        if kill:
            car.checkScore() #Checks to see the cars score
            cars.remove(car) #Removes the car from the sprites list


    #Car movement logic
    for car in cars:
        car.state = ML.linNormalizeState(car.state, lineMax) #Divides the measured length of lines by the max possible
        car.state = np.append(car.state,car.speed) #Adds the cars current speed to the state array
        #Determines the movement of the car
        layer_a, layer_z = ML.getDecision(car.state, car.weights, car.biases, f=1,s=4)
        car.turn(layer_a[-1][0]) #Turns the car
        car.adjustSpeed(layer_a[-1][1]) #Adjusts the cars speed
        car.move() #Moves the car
        car.draw() #Draws the car
        car.state = car.drawLines() #Draws the yellow lines for the car

    #Blits all of the scores to the screen and updates the screen
    screen.blit(playerScore, (1000, 350))
    screen.blit(score1, (1000, 400))
    screen.blit(score2, (1000, 450))
    screen.blit(score3, (1000, 500))
    screen.blit(score4, (1000, 550))
    screen.blit(score5, (1000, 600))
    screen.blit(generations, (1000, 650))
    screen.blit(howManyFinished, (1000,300))
    pygame.display.update()


    #Determines if the car has gone off of the map or has come to a complete stop, kills the car on either condition
    for car in cars:
        if (car.checkCollision()):
            car.checkScore()
            cars.remove(car)
        if car.speed == 0:
            car.checkScore()
            cars.remove(car)


    #Sets exit to true if the x button is clicked
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit = True


    #Creating cars and handles creating them based off of a parent if the game is not in the first generation
    if carCount < numCars: #If there are less cars than allows
        if (carsInSpawnZone() == False) or (spawnZone == False):
            if initialLoop:
                createCar()
                carCount += 1
                carNumber += 1
            else:
                weights = oldBestCars[parentNumber][4]
                baises = oldBestCars[parentNumber][5]
                parent = [oldBestCars[parentNumber][0],oldBestCars[parentNumber][2]]
                try:
                    sigma = 1.5/math.sqrt(oldBestCars[parentNumber][2])
                except:
                    sigma = 1
                if generationCount > 10 and oldBestCars[4][2] == 8:
                    if reducingSigma:
                        print("reducing sigma")
                    reducingSigma = False
                    sigma = 0.10
                if generationCount > 25 and oldBestCars[4][2] == 8:
                    if ultraReducingSigma:
                        print("ultra reduction")
                    ultraReducingSigma = False
                    sigma = 0.075
                parentNumber += 1
                if parentNumber == bestCarsRemembered:
                    parentNumber = 0
                createModdedCar(weights,baises,parent, sigma)
                carCount += 1
                carNumber += 1


    #Determines if the generation has finished and will start a new generation
    if numCars - carCount == 0 and len(cars) == 0 and (car1.dead or player == False):
        car1.dead = False
        carCount = 0
        screenScoreStuff()
        generationCount += 1
        car1.center = startingPos
        car1.angle = startingAngle
        car1.speed = 2
        finishedCount = finished
        finished = 0
        car1.time = 0
        for car in bestCars:
            checkpointsAccomplished = car[2]
            number = -1
            if car[2] > 0:
                for i in range(len(bestCars)):
                    if oldBestCars[i][2] < checkpointsAccomplished:
                        number = i
                        break
                    if oldBestCars[i][2] > 0:
                        if oldBestCars[i][2] == checkpointsAccomplished:
                            if oldBestCars[i][3][oldBestCars[i][2]-1] > car[3][oldBestCars[i][2]-1]:
                                number = i
                                break
                if number != -1:
                    oldBestCars.insert(number,car)
                    oldBestCars.pop(-1)
        network = ML.displayNetwork(dim_array,oldBestCars[0][4])
        initialLoop = False
        bestCars = []
        for i in range(bestCarsRemembered):
            bestCars.append([0, 0, 0, [], [], []])


    #Removes cars that have taken too long to reach the next checkpoint (often are going backwards)
    for car in cars:
        if len(car.checkPoints) == 0:
            if car.time > 300:
                car.checkScore()
                cars.remove(car)
        elif (car.time - car.checkPoints[-1]) > 500:
            car.checkScore()
            cars.remove(car)


sys.exit()