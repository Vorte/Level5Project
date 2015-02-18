import pygame, time, struct, sys, copy
from pygame.locals import *
from _collections import deque
import random, Queue, threading, string

KEY_HEIGHT = 43
KEY_WIDTH = 73
USER_ID = sys.argv[1]
QUEUE = Queue.Queue(25)
START_TIME = time.time()

class TextRectException:
    def __init__(self, message = None):
        self.message = message
    def __str__(self):
        return self.message
    
class Looping(object):
    def __init__(self):
        self.isRunning = True
    
    def collectacceldata(self):
        print "Starting worker thread"        
        ACCEL_FILE = "/sys/devices/platform/lis3lv02d/position"
        writefile = open(USER_ID+ "_accel.txt","w")
        
        while self.isRunning:
            with open(ACCEL_FILE, 'r') as acc:
                rawxyz = acc.read().replace('(', '').replace(')', '').replace('\n', '')
                xyz = map(int, rawxyz.split(","))
                xyz_data = xyz[0], xyz[1], xyz[2]
                
                if QUEUE.full():
                    QUEUE.get_nowait()
                
                elapsedtime = str(int((time.time() - START_TIME)*1000))
                writefile.write(elapsedtime + " "+ str(xyz_data)+'\n')
                QUEUE.put_nowait(xyz_data)
                time.sleep(0.02)
        
        writefile.close()
        print "Exiting worker thread"

def render_textrect(rect, text, font, text_color, background_color, justification=0):
    """Returns a surface containing the passed text string, reformatted
    to fit within the given rect, word-wrapping as necessary. The text
    will be anti-aliased.

    Success - a surface object with the text rendered onto it.
    Failure - raises a TextRectException if the text won't fit onto the surface.
    """
    final_lines = []

    requested_lines = text.splitlines()

    # Create a series of lines that will fit on the provided
    # rectangle.

    for requested_line in requested_lines:
        if font.size(requested_line)[0] > rect.width-10:
            words = requested_line.split(' ')
            # if any of our words are too long to fit, return.
            for word in words:
                if font.size(word)[0] >= rect.width-10:
                    raise TextRectException, "The word " + word + " is too long to fit in the rect passed."
            # Start a new line
            accumulated_line = ""
            for word in words:
                test_line = accumulated_line + word + " "
                # Build the line while the words fit.    
                if font.size(test_line)[0] < rect.width-10:
                    accumulated_line = test_line
                else:
                    final_lines.append(accumulated_line)
                    accumulated_line = word + " "
            final_lines.append(accumulated_line)
        else:
            final_lines.append(requested_line)

    # Let's try to write the text out on the surface.

    surface = pygame.Surface(rect.size)
    surface.fill(background_color)
    gold_color = (255,215,0)

    accumulated_height = 0
    accum_width = 0
    for line in final_lines:
        if accumulated_height + font.size(line)[1] >= rect.height:
            raise TextRectException, "Once word-wrapped, the text string was too tall to fit in the rect."
        if line != "":
            if '\"' in line:
                words = line.strip().split('\"')
                firstsurface = font.render(words[0], 1, text_color)
                surface.blit(firstsurface, (10, accumulated_height))
                temp = text_color
                text_color= gold_color
                gold_color=temp
                secondsurface = font.render(words[1], 1, text_color)
                surface.blit(secondsurface, (10+firstsurface.get_width(), accumulated_height))
                accum_width = secondsurface.get_width()
                if len(words) == 3:
                    temp = text_color
                    text_color = gold_color
                    gold_color=temp
                    thirdsurface = font.render(words[2], 1, text_color)
                    surface.blit(thirdsurface, (10+firstsurface.get_width()+secondsurface.get_width(), accumulated_height))
                    accum_width = thirdsurface.get_width()
            else:    
                tempsurface = font.render(line, 1, text_color)
                surface.blit(tempsurface, (10, accumulated_height))
                accum_width = tempsurface.get_width()
        accumulated_height += font.size(line)[1]
        
    return surface, accumulated_height, accum_width

class Task(object):
    def __init__(self, sentences, instruction, twohand=False):
        self.sentences = copy.copy(sentences)
        self.instruction = instruction
        self.twohand = twohand
        
        random.shuffle(self.sentences)
        
    def start(self, taskno):
        self.upfile = open(USER_ID + "_" + str(taskno)+ "up.txt", 'w')
        self.downfile = open(USER_ID +"_" + str(taskno)+ "down.txt", 'w')
        self.upfile.write(self.instruction + '\n')
        self.downfile.write(self.instruction + '\n')    
    
    def nextsentence(self):
        if len(self.sentences)== 0:
            return None
        return self.sentences.pop()
        
    def close(self):
        self.upfile.close()
        self.downfile.close()

class Instruction(object):
    def __init__(self, background, screen, text, x, y):
        self.background = pygame.Surface((854, 480),SRCALPHA).convert_alpha()
        self.layer = pygame.Surface((854, 480),SRCALPHA).convert_alpha()
        self.screen = screen
        self.font = pygame.font.SysFont("Arial", 40)
         
        my_rect = pygame.Rect((0, 0, 480, 527))
         
        rendered_text = render_textrect(my_rect, text, self.font, (216, 216, 216), (48, 48, 48), 0)[0]
        if rendered_text:
            self.layer.blit(pygame.transform.rotate(rendered_text,90), my_rect.topleft)            
         
        my_rect = pygame.Rect((480, 70, 350, 40))
         
        rendered_text = render_textrect(my_rect, "touch screen to continue", 
                                        pygame.font.SysFont("Arial", 30), (216, 216, 216), (48, 48, 48), 0)[0]
        if rendered_text:
            self.layer.blit(pygame.transform.rotate(rendered_text,90), my_rect.topleft)  
            
        self.screen.blit(self.background,(x, y))
        self.screen.blit(self.layer,(x,y))
        pygame.display.flip()     
        
class TextInput(object):
    ''' Handles the text input box and manages the cursor '''
    def __init__(self, background, screen, text, x, y):
        if text == None:
            return None
        
        self.x = y
        self.y = x
        self.text = ""
        self.caption = text
        self.width = 527
        self.height = 480        
        self.font = pygame.font.Font(None, 50)
        self.cursorheight = 0
        self.cursorwidth = 0
        self.rect = Rect(self.x,self.y,self.height,self.width)
        self.layer = pygame.Surface((self.width,self.height),SRCALPHA).convert_alpha()
        self.background = pygame.Surface((self.width,self.height),SRCALPHA).convert_alpha()
        self.background.blit(background,(0,0),self.rect) # Store our portion of the background
        self.cursorlayer = pygame.transform.rotate(pygame.Surface((3,40)), 90)
        self.screen = screen
        self.cursorvis = True
        
        self.draw()
          
    
    def draw(self, text= None):
        ''' Draw the text input box '''       
        if text == None:
            text = self.caption

        my_rect = pygame.Rect((0, 0, 480, 400))
        
        rendered_data = render_textrect(my_rect, text, self.font, (216, 216, 216), (48, 48, 48), 0)        
        rendered_text = rendered_data[0]
        if rendered_text:
            self.layer.blit(pygame.transform.rotate(rendered_text,90), my_rect.topleft)
        
        my_rect = pygame.Rect((250, 0, 480, 400))
        
        rendered_data = render_textrect(my_rect, self.text, self.font, (216, 216, 216), (48, 48, 48), 0)        
        rendered_text, self.cursorheight, self.cursorwidth = rendered_data[0], rendered_data[1], rendered_data[2]
        
        if rendered_text:
            self.layer.blit(pygame.transform.rotate(rendered_text,90), my_rect.topleft)                
        
        self.screen.blit(self.background,(self.x, self.y))
        self.screen.blit(self.layer,(self.x,self.y))
        self.drawcursor()
        pygame.display.flip()       
            
    def flashcursor(self):
        ''' Toggle visibility of the cursor '''
        if self.cursorvis:
            self.cursorvis = False
        else:
            self.cursorvis = True
         
        self.screen.blit(self.background,(self.x, self.y))
        self.screen.blit(self.layer,(self.x,self.y))  
         
        if self.cursorvis:
            self.drawcursor()
        pygame.display.flip()
         
    def addcharatcursor(self, letter):
        
        ''' Add a character wherever the cursor is currently located '''
        self.text += letter
        self.draw()   
         
    def drawcursor(self):
        ''' Draw the cursor '''
        x = 250
        y = 465-self.cursorwidth
        if self.cursorheight>35:
            x = 215 + self.cursorheight
            y = y + 9 # don't know why
        
        self.screen.blit(self.cursorlayer,(x, y))
        
class TwoHandedInput(TextInput):
    def __init__(self, background, screen, text, x, y):
        self.thumb = text[1]
        super(TwoHandedInput, self).__init__(background, screen, text[0], x, y)
        
    def draw(self):
        text = "type \"" + self.caption + "\" with "+ self.thumb + " thumb" 
        if self.thumb == "both":
            text += "s"       
        super(TwoHandedInput, self).draw(text)                       

class VirtualKey(object):
    ''' A single key for the VirtualKeyboard '''
    def __init__(self, caption, x, y, w=KEY_WIDTH, h=KEY_HEIGHT):
        self.x = y
        self.y = x
        self.caption = caption
        self.width = w
        self.height = h
        self.font = None
        self.keylayer = pygame.Surface((self.width,self.height)).convert()
        self.keylayer.fill((0, 0, 0))
        self.keylayer.set_alpha(160)
        # Pre draw the border and store in my layer
        pygame.draw.rect(self.keylayer, (255,255,255), (0,0,self.width,self.height), 1)
        
    def draw(self, screen, background):
        '''  Draw one key if it needs redrawing '''
        
        myletter = self.caption        
        position = Rect(self.x, self.y, self.width, self.height)
        
        # put the background back on the screen so we can shade properly
        screen.blit(background, (self.x,self.y), position)      
        
        # Put the shaded key background into my layer
        color = (0,0,0)
        
        # Copy my layer onto the screen using Alpha so you can see through it
        pygame.draw.rect(self.keylayer, color, (1,1,self.width-2,self.height-2))                
        screen.blit(self.keylayer,(self.x,self.y))    
                
        # Create a new temporary layer for the key contents
        # This might be sped up by pre-creating both selected and unselected layers when
        # the key is created, but the speed seems fine unless you're drawing every key at once
        templayer = pygame.Surface((self.width,self.height))
        templayer.set_colorkey((0,0,0))
                       
        color = (255,255,255)
        text = self.font.render(myletter, 1, (255, 255, 255))
        text = pygame.transform.rotate(text, 90)
        textpos = text.get_rect()
        blockoffx = (self.width / 2)
        blockoffy = (self.height / 2)
        offsetx = blockoffx - (textpos.width / 2)
        offsety = blockoffy - (textpos.height / 2)
        templayer.blit(text,(offsetx, offsety))
        
        screen.blit(templayer, (self.x,self.y))

class VirtualKeyboard(object):
    ''' Implement a basic full screen virtual keyboard for touchscreens '''
    def run(self, screen, task):
        # First, make a backup of the screen        
        self.screen = screen
        self.background = pygame.Surface((854,480))        
        
        # Copy original screen to self.background
        self.background.blit(screen,(0,0))
        
        # Shade the background surrounding the keys
        self.keylayer = pygame.Surface((854,480))
        self.keylayer.fill((0, 0, 0))
        self.keylayer.set_alpha(100)
        self.screen.blit(self.keylayer,(0,0))
        
        self.keys = []
        
        pygame.font.init() # Just in case 
        self.font = pygame.font.Font(None, 40)   
    
        self.input = Instruction(self.background,self.screen, task.instruction,0,0)
        self.task = task
        self.extcap = open('/dev/extcap', 'r')
        
        self.addkeys()          
        self.paintkeys()        
        
        time.sleep(2)
        pygame.event.clear()
        
        counter = 0
        instruction = True
        lasttouchdown = False
        clicksound = pygame.mixer.Sound("click.wav")
        # My main event loop (hog all processes since we're on top, but someone might want
        # to rewrite this to be more event based.  Personally it works fine for my purposes ;-)
        while 1:
            time.sleep(.05)
            events = pygame.event.get()
            if events <> None:
                for e in events: 
                    if e.type == MOUSEBUTTONDOWN and not lasttouchdown:
                        clicksound.play()
                        if not instruction:
                            self.selectatmouse()
                        lasttouchdown = True   
                    if e.type == MOUSEBUTTONUP and lasttouchdown:                        
                        if instruction or self.clickatmouse(): 
                            instruction = False
                            sentence = self.task.nextsentence()
                            if sentence == None:
                                self.extcap.close()
                                return
                            if task.twohand:
                                self.input = TwoHandedInput(self.background,self.screen, sentence,0,0)
                            else:
                                self.input = TextInput(self.background,self.screen, sentence,0,0)                        
                        lasttouchdown = False                                                 
                        
            counter += 1
            if counter > 10 and instruction == False:                
                self.input.flashcursor()
                counter = 0             
    
    def clickatmouse(self):
        ''' Check to see if the user is pressing down on a key and draw it selected '''
        data = self.collectdata()
        self.task.upfile.write(data)
        
        length = len(self.input.text)
        self.input.addcharatcursor(self.input.caption[length:length+1])
        if self.input.caption == self.input.text:
            return True
        return False        
            
    def collectdata(self):
        mousepos = pygame.mouse.get_pos()
        
        extcap_struct = struct.Struct("<" + "H" * 24)
        extcap_data = [0 for x in range(24)]
        ed = self.extcap.read(extcap_struct.size) 
        if ed and len(ed) == extcap_struct.size:
            extcap_data = extcap_struct.unpack(ed)
        
        length = len(self.input.text)
        character = self.input.caption[length:length+1]
        
        elapsedtime = str(int((time.time() - START_TIME)*1000))
        data = character + '\t' +elapsedtime+'\t'+str(mousepos)+ '\t'+str(list(QUEUE.queue))+'\t' + str(extcap_data) + '\n'
        if self.task.twohand:
            data = self.input.thumb + '\t' + data 
        return data        
        
    def selectatmouse(self):
        ''' User has clicked a key, let's use it '''
        ''' Data stored as elapsedtime space mouseposition 
        space accelerometer space bod newline
        For example, 2584 (23, 45) (123, 34, 231) (12, 23, 456 ...) \n '''
        data = self.collectdata()
        self.task.downfile.write(data)        
            
    def addkeys(self):
        ''' Adds the setup for the keys.  This would be easy to modify for additional keys '''
        
        x = 5
        y = 537
        
        row = ['q','w','e','r','t','y','u','i','o','p']
        for item in reversed(row):
            onekey = VirtualKey(item,x,y)
            onekey.font = self.font
            self.keys.append(onekey)
            x += KEY_HEIGHT+4
        
        x = 30
        y += KEY_WIDTH+6
        row = ['a','s','d','f','g','h','j','k','l']
        for item in reversed(row):
            onekey = VirtualKey(item,x,y)
            onekey.font = self.font
            self.keys.append(onekey)
            x += KEY_HEIGHT+4
        
        x = 85
        y += KEY_WIDTH+6   
                
        row = ['z','x','c','v','b','n','m']
        for item in reversed(row):
            onekey = VirtualKey(item,x,y)
            onekey.font = self.font
            self.keys.append(onekey)
            x += KEY_HEIGHT+4
             
        x = 120
        y += KEY_WIDTH+6  
        
        onekey = VirtualKey('SPACE',x,y, h= 250)
        onekey.font = self.font
        self.keys.append(onekey)
                            
        
    def paintkeys(self):
        ''' Draw the keyboard (but only if they're dirty.) '''
        for key in self.keys:
            key.draw(self.screen, self.background)
        
        pygame.display.flip()        
        

pygame.init()
pygame.mixer.init()
pygame.display.set_caption('keyboard')
mode = pygame.display.list_modes()[0]

screen = pygame.display.set_mode(mode, pygame.FULLSCREEN)

looping = Looping()
thread = threading.Thread(target=looping.collectacceldata)
thread.daemon = True
thread.start()
mvykeys = VirtualKeyboard()

with open("dataset.txt") as f:
    lines = f.read().splitlines()
    sentences = deque(lines)

random.shuffle(sentences)
letters = list(string.ascii_lowercase)
letterstwice = letters+letters
twothumbletters = map(lambda x: (x, "left"), letterstwice)+ map(lambda x: (x, "right"), letterstwice)
twothumbsentences = [(sentences.pop(), "left") for x in range(2)]+[(sentences.pop(), "right") for x in range(2)]+[(sentences.pop(), "both") for x in range(4)]
    
set1 = []
set2 = []
set3 = []

set1.append(Task(letterstwice, "Please hold the phone in your right hand and type with you right thumb"))
set1.append(Task(twothumbletters, "Please hold the phone in both hands. At each letter you will be instructed which thumb to use", True))
set1.append(Task(letterstwice, "Please hold the phone in your right hand and type with your left index finger"))
set1.append(Task(letterstwice, "Please hold the phone in your left hand and type with your left thumb"))
set1.append(Task(letterstwice, "Please hold the phone in your right hand and type with your left index finger"))
set1.append(Task(twothumbletters, "Please hold the phone in both hands. At each letter you will be instructed which thumb to use", True))
set1.append(Task(letterstwice, "Please hold the phone in your right hand and type with you right thumb"))
set1.append(Task(letterstwice, "Please hold the phone in your left hand and type with your left thumb"))
 
set2.append(Task(twothumbletters, "Please hold the phone in both hands. At each letter you will be instructed which thumb to use", True))
set2.append(Task(letterstwice, "Please hold the phone in your right hand and type with you right thumb"))
set2.append(Task(letterstwice, "Please hold the phone in your left hand and type with your left thumb"))
set2.append(Task(letterstwice, "Please hold the phone in your right hand and type with your left index finger"))
 
set2.append(Task([sentences.pop() for x in range(2)], "From now on you will be typing sentences rather than letters. Please hold the phone in your right hand and type with you right thumb"))
set2.append(Task([sentences.pop() for x in range(2)], "Please hold the phone in your left hand and type with your left thumb"))
set2.append(Task(twothumbsentences, "Please hold the phone in both hands. At each letter you will be instructed which thumb to use", True))
set2.append(Task([sentences.pop() for x in range(2)], "Please hold the phone in your right hand and type with your left index finger"))

set3.append(Task(twothumbsentences, "Please hold the phone in both hands. At each letter you will be instructed which thumb to use", True))
set3.append(Task([sentences.pop() for x in range(2)], "Please hold the phone in your right hand and type with your left index finger"))
set3.append(Task([sentences.pop() for x in range(2)], "Please hold the phone in your left hand and type with your left thumb"))
set3.append(Task([sentences.pop() for x in range(2)], "Please hold the phone in your right hand and type with you right thumb"))
set3.append(Task(twothumbsentences, "Please hold the phone in both hands. At each letter you will be instructed which thumb to use", True))
set3.append(Task([sentences.pop() for x in range(2)], "Please hold the phone in your left hand and type with your left thumb"))
set3.append(Task([sentences.pop() for x in range(2)], "Please hold the phone in your right hand and type with you right thumb"))
set3.append(Task([sentences.pop() for x in range(2)], "Please hold the phone in your right hand and type with your left index finger"))

breaktask = Task([], "You can now take a short break if you wish")

taskno = 1
for task in set1:
    task.start(taskno)
    mvykeys.run(screen, task)
    task.close()
    taskno += 1
         
mvykeys.run(screen, breaktask)

for task in set2:
    task.start(taskno)
    mvykeys.run(screen, task)
    task.close()
    taskno += 1

mvykeys.run(screen, breaktask)    
    
for task in set3:
    task.start(taskno)
    mvykeys.run(screen, task)
    task.close()
    taskno += 1
    
mvykeys.run(screen, Task([], "This is the end of the experiment. Please return the device to the experimenter"))        
        
pygame.quit()
looping.isRunning = False










