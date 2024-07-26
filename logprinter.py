################################################################################################
#                  INFORMATION
#
#            File Name  :   logprinter.py
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
#   First Stable Build  :   26th MAY 2024
#             Use Case  :   Log Message Printing Functions
#                 
#                 Type  :   Functions
#               Inputs  :   Message (String)
#
#               Output  :   Text Displayed
#          Description  :   The file contains logging functions for formatting and displaying 
#                           messages in a console. These functions provide various styles for 
#                           logging information, including bordered and simple formats.
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



###----------------------------------------------------------------------
#            print_log()
#       Inputs    :     message: A string to be logged.
#
#       Output    :     None
#   Description   :     Prints the message with a border of equal signs (=) to make it stand out. 
#                       The width of the border is set to 30 characters. The message is centered 
#                       within this border, with padding calculated based on the message length.
###----------------------------------------------------------------------
def print_log(message):
    log_width = 30
    message_padding = (log_width - len(message)) // 2
    print(f"{'=' * log_width}")
    print(f"{'=' * message_padding}{message}{'=' * message_padding}")
    print(f"{'=' * log_width}")



###----------------------------------------------------------------------
#              log_std()
#       Inputs    :     message: A string to be logged.
#
#       Output    :     None
#   Description   :     Prints the message with a border of hyphens (-) for a simpler format. 
#                       The width of the border is set to 20 characters. The message is centered 
#                       within this border, with padding calculated based on the message length. 
#                       Note that the border is not fully rendered in this function as the equal 
#                       signs are commented out.
###----------------------------------------------------------------------
def log_std(message):
    log_width = 20
    message_padding = (log_width - len(message)) // 2
    #print(f"{'=' * log_width}")
    print(f"{'-' * message_padding}{message}{'-' * message_padding}")
    #print(f"{'=' * log_width}")



###----------------------------------------------------------------------
#         print_simple_log()
#       Inputs    :     message: A string to be logged.
#
#       Output    :     None
#   Description   :     Prints the message with a simple [LOG] prefix for basic logging. This 
#                       format provides a straightforward way to output log messages without 
#                       additional styling.
###----------------------------------------------------------------------
def print_simple_log(message):
    print(f"[LOG] {message}")
