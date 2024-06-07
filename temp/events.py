import pygame
from queue import Empty
from logprinter import print_log, print_simple_log, log_std

def putqueue(item, event_queue):
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
        event_queue.put(('mouse_click', pos))  # Send mouse click position
