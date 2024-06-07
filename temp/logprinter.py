
def print_log(message):
    log_width = 30
    message_padding = (log_width - len(message)) // 2
    print(f"{'=' * log_width}")
    print(f"{'=' * message_padding}{message}{'=' * message_padding}")
    print(f"{'=' * log_width}")

def log_std(message):
    log_width = 20
    message_padding = (log_width - len(message)) // 2
    #print(f"{'=' * log_width}")
    print(f"{'-' * message_padding}{message}{'-' * message_padding}")
    #print(f"{'=' * log_width}")

def print_simple_log(message):
    print(f"[LOG] {message}")
