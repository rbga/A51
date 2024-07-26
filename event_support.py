from queue import Empty
from logprinter import print_simple_log


################################################################################################
#                  INFORMATION
#
#            File Name  :   event_support.py
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
#   First Stable Build  :   15th MAY 2024
#             Use Case  :   Custom Python Library related to Multiprocessing Event Handling.
#
#                 Type  :   Function(s)
#               Inputs  :   Many
#
#               Output  :   Many
#          Description  :   This Python file manages the state of a system based on events 
#                           received from an event queue. It includes state and orientation 
#                           tracking, as well as functions to update the system's state and 
#                           handle events, including specific and larger event types. The 
#                           system seems to have various modes or events denoted by letters 
#                           and possibly object names.
#
# ------------------------------------------------------------------
#               LAST MODIFICATION
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
# Date of Modification  :   25th JULY 2024
#
#          Description  :   Added Information Block and Code Module 
#                           Block for every Code Module in the file.
#------------------------------------------------------------------
#
################################################################################################


rejected_obj = {}


state = {
    "Starter"           : True,
    "event_Y"           : False,
    "prompted_for_name" : False,
    "event_S"           : False,
    "event_W"           : False,
    "ievent_C"          : False,
    "got_name"          : False,
    "termi"             : False,
    "tempState"         : False,
    "inner"             : False,
    "X"                 : False,
    "event_R"           : False
}

orientations = {
    "front"     : False,
    "left"      : False,
    "back"      : False,
    "right"     : False,
    "top"       : False,
    "bottom"    : False
}





###----------------------------------------------------------------------
#           update_state()
#       Inputs    :     event: An event identifier, typically a single character or a string.
#                       state: A dictionary representing the current state of the system.
#
#       Output    :     Modifies the state dictionary in place based on the event received.
#   Description   :     Updates the system's state according to the specific event provided. 
#                       This may include setting boolean flags or storing object names.
###----------------------------------------------------------------------
def update_state(event, state):
    if event == 'Q':
        state["termi"]              = True

    elif event == 'S':
        state["event_S"]            = True
        state["Starter"]            = False
      
    elif event == 'Y':
        state["event_Y"]            = True
        state["event_S"]            = False
    
    elif event == 'W':
        state["event_W"]            = True
        state["event_Y"]            = False

    elif len(event) > 1:
        state["obj_name"]           = event
        state["got_name"]           = True
        state["event_W"]            = False

    elif event == 'C':
        state["ievent_C"]           = True
        state["got_name"]           = False

    elif event == 'T':
        state["tempState"]          = True
        state["ievent_C"]           = False

    elif event == 'N':
        state["inner"]              = True
        state["event_R"]            = False

    elif event == 'R':
        state["event_R"]            = True

    elif event == 'X':
        state["X"]                  = True



###----------------------------------------------------------------------
#           handle_event()
#       Inputs    :     event_queue: A queue from which events are fetched.
#                       state: A dictionary representing the current state of the system.
#
#       Output    :     Modifies the state dictionary and logs the event; exits the program 
#                       if a termination event is encountered.
#
#   Description   :     Handles events by fetching them from the event queue, logging them, 
#                       and updating the state accordingly. If the event indicates a termination 
#                       condition, the program exits.
###----------------------------------------------------------------------
def handle_event(event_queue, state):
    try:
        event = event_queue.get_nowait()
        print_simple_log(event)
        update_state(event, state)
        if state["termi"]:
            exit()
    except Empty:
        pass





###----------------------------------------------------------------------
#        handle_large_event()
#       Inputs    :     event_queue: A queue from which events are fetched.
#                       state: A dictionary representing the current state of the system.
#
#       Output    :     Modifies the state dictionary and logs the event; exits the program 
#                       if a termination event is encountered.
#
#   Description   :     Similar to handle_event, but specifically processes events that are 
#                       strings longer than 2 characters. It updates the state and handles termination.
###----------------------------------------------------------------------
def handle_large_event(event_queue, state):
    try:
        event = event_queue.get_nowait()
        # Check if event is a string and has more than 3 characters
        if len(event) > 2:
            print_simple_log(event)
            update_state(event, state)
            if state["termi"]:
                exit()
    except Empty:
        pass





UNKN_state = {
    "Yes": False,
    "Begin": False,
    "Got_Name": False,
    "termi": False,
    "obj_name": ""
}

UNKN_orientations = {
    "front": False,
    "Front_Begin": False,
    "left": False,
    "back": False,
    "right": False,
    "top": False,
    "bottom": False
}




###----------------------------------------------------------------------
#         UNKN_update_state()
#       Inputs    :     event: An event identifier, typically a single character or a string.
#                       UNKN_state: A dictionary representing an alternative or additional state of the system.
#
#       Output    :     Modifies the UNKN_state dictionary in place based on the event received.
#   Description   :     Updates a secondary or unknown state dictionary based on the event. This might be part of a 
#                       different system or subsystem.
###----------------------------------------------------------------------
def UNKN_update_state(event, UNKN_state):
    if event == 'Y':
        UNKN_state["Yes"] = True
    if event == 'A':
        UNKN_state["Begin"] = True
    if len(event) > 1:
        UNKN_state["obj_name"] = event
        UNKN_state["got_name"] = True
    if event == 'F':
        UNKN_orientations["Front_Begin"] = True
    if event == 'Q' or event == 'N':
        UNKN_state["termi"] = True





###----------------------------------------------------------------------
#          UNKN_handle_event()
#       Inputs    :     event: An event identifier, typically a single character or a string.
#                       UNKN_state: A dictionary representing an alternative or additional state of the system.
#
#       Output    :     Modifies the UNKN_state dictionary and logs the event; exits the program if a termination event is encountered.
#   Description   :     Handles events for the unknown or secondary state, updating the state based on the event and logging it. 
#                       It also checks for termination events to exit the program.
###----------------------------------------------------------------------
def UNKN_handle_event(event_queue, UNKN_state):
    try:
        event = event_queue.get(timeout=0.01)
        print_simple_log(event)
        UNKN_update_state(event, UNKN_state)
        if UNKN_state["termi"]:
            exit()
    except Empty:
        pass














"""def putqueue(item, event_queue):
    while event_queue.full():
        try:
            event_queue.get_nowait()  # Remove the oldest frame to make space
        except Empty:
            pass
    event_queue.put(item) 


def handle_events(event, event_queue):
    print_simple_log("Event received:")
    if event.type == pygame.KEYDOWN:
        print_simple_log("Keydown event received")
        handle_keydown_event(event, event_queue)

    elif event.type == pygame.MOUSEBUTTONDOWN:
        print_simple_log("Mouse button down event received")
        handle_mouse_button_down_event(event, event_queue)

    elif event.type == pygame.QUIT:
        print_simple_log("Quit event received")
        putqueue('quit', event_queue)
        #event_queue.put('quit')  # Handle window close event


def handle_keydown_event(event, event_queue):
    global user_text
    
    if event.key == pygame.K_s:
        print_simple_log("Key 's' pressed")
        putqueue('start', event_queue)
        #event_queue.put('start')  # Command to save a frame
    elif event.key == pygame.K_q:
        print_simple_log("Key 'q' pressed")
        putqueue('quit', event_queue)
        #event_queue.put('quit')  # Command to quit the worker processes
    elif event.key == pygame.K_c:
        print_simple_log("Key 'c' pressed")
        putqueue('continue', event_queue)
        event_queue.put('continue')  # Command to pause processing
    elif event.key == pygame.K_w:
        print_simple_log("Key 'w' pressed")
        handle_text_input(event_queue)


def handle_text_input(event_queue):
    user_text = ''
    input_finished = False
    print_simple_log("Waiting for text input...")
    while not input_finished:
        for inner_event in pygame.event.get():
            print_simple_log("Inner event:")
            if inner_event.type == pygame.KEYDOWN:
                print_simple_log("Keydown event received")
                if inner_event.key == pygame.K_BACKSPACE:
                    print_simple_log("Backspace pressed")
                    user_text = user_text[:-1]  # Remove last character
                elif inner_event.key == pygame.K_RETURN:
                    print_simple_log("Return pressed")
                    input_finished = True  # Exit loop on 'Enter'
                else:
                    print_simple_log("Character input:" + str(inner_event.unicode))
                    user_text += inner_event.unicode  # Append character
            elif inner_event.type == pygame.QUIT:
                print_simple_log("Quit event received")
                input_finished = True  # Exit loop on window close

    print_simple_log("User input:" + str(user_text))
    putqueue('user_input', event_queue)
    #event_queue.put(('user_input', user_text))


def handle_mouse_button_down_event(event, event_queue):
    if event.button == 1:  # Left mouse button
        print_simple_log("Left mouse button clicked")
        pos = pygame.mouse.get_pos()
        event_queue.put(('mouse_click', pos))  # Send mouse click position"""
