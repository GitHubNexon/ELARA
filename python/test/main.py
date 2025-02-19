import time
import sympy
import json
import moment
import re
import requests
import subprocess
import os
from datetime import datetime
from multiprocessing import Process, Queue

def load_commands():
    """Loads predefined commands from a JSON file."""
    try:
        with open('../json/commands.json', 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def typing_animation(text, typing_speed=0.05):
    """Simulates typing animation for responses."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(typing_speed)
    print()

def get_greeting():
    """Returns a greeting based on the current time."""
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Good morning"
    elif current_hour < 18:
        return "Good afternoon"
    return "Good evening"

def perform_math(expression):
    """Evaluates mathematical expressions safely."""
    try:
        return sympy.sympify(expression)
    except (sympy.SympifyError, Exception):
        return None

def get_time_response(command):
    """Checks if the command asks for the time and returns it."""
    if re.search(r"\b(time|hour|clock)\b", command):
        return f"It's {moment.now().format('HH:mm:ss')}."
    return None

def get_date_response(command):
    """Checks if the command asks for the date and returns it."""
    if re.search(r"\b(date|today)\b", command):
        return f"Today is {moment.now().format('MMMM D, YYYY')}."
    return None

def handle_greetings(command):
    """Checks if the command is a greeting."""
    return any(greeting in command for greeting in ["hello", "hi", "hey", "morning", "afternoon", "evening"])

def get_word_definition(word):
    """Fetches the definition of a given word from an API."""
    try:
        response = requests.get(f'https://api.dictionaryapi.dev/api/v2/entries/en/{word}')
        if response.status_code == 200:
            data = response.json()
            definitions = data[0]['meanings'][0]['definitions']
            return "\n".join([definition['definition'] for definition in definitions])
        return "Sorry, I couldn't find a definition for that word."
    except requests.RequestException as e:
        return f"An error occurred while fetching the definition: {e}"

def clear_conversation():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')
    typing_animation("ELARA: Conversation has been cleared. How can I assist you now?")

def call_deepseek(prompt, queue):
    """Calls DeepSeek R1 model using Ollama with a timeout using multiprocessing."""
    command = ["ollama", "run", "deepseek-r1:1.5b"]

    try:
        result = subprocess.run(command, input=prompt, capture_output=True, text=True, check=True, encoding="utf-8", timeout=30)
        queue.put(result.stdout.strip())  # Send result to queue
    except subprocess.TimeoutExpired:
        queue.put("ELARA process took too long to respond and was timed out.")
    except Exception as e:
        queue.put(f"Error communicating with ELARA: {e}")

def process_command(command, commands):
    """Processes user input and determines the appropriate response."""
    # Check if the command matches any greeting
    if handle_greetings(command):
        typing_animation(f"ELARA: {get_greeting()}! How can I assist you today?")

    # Check for math operations
    elif any(op in command for op in commands["math_operators"]):
        if (result := perform_math(command)) is not None:
            typing_animation(f"ELARA: Here's the result: {result}.")
        else:
            typing_animation("ELARA: Sorry, I couldn't understand that math operation.")

    # Handle time and date responses
    elif any(time_command in command for time_command in commands["time_commands"]):
        typing_animation(f"ELARA: {get_time_response(command)}")
    elif any(date_command in command for date_command in commands["date_commands"]):
        typing_animation(f"ELARA: {get_date_response(command)}")

    # Handle word definition
    elif "define" in command:
        word_match = re.search(r"\bdefine (\w+)\b", command)
        if word_match:
            typing_animation(f"ELARA: {get_word_definition(word_match.group(1))}")
        else:
            typing_animation("ELARA: Please provide a word to define.")

    # Handle goodbye or exit commands
    elif any(exit_command in command for exit_command in commands["goodbye"]):
        typing_animation("ELARA: Goodbye! Take care and have a wonderful day ahead!")
        return False

    # Handle 'clear' command
    elif any(clear_command in command for clear_command in commands["clear"]):
        clear_conversation()

    else:
        # If input is not recognized, send it to DeepSeek R1
        typing_animation("ELARA: Thinking...")
        queue = Queue()
        process = Process(target=call_deepseek, args=(command, queue))
        process.start()
        process.join()  # Wait for the process to finish
        response = queue.get()  # Retrieve the result from the queue
        typing_animation(f"ELARA: {response}")

    return True

def get_multiline_input():
    """Allow multiple lines of input from the user, processing each separately."""
    print("Enter your input (type '.' on a new line to finish):")
    lines = []
    while True:
        line = input("> ")
        if line.lower() == ".":  # Ends the input collection when 'done' is entered
            break
        lines.append(line)
    return lines

def main():
    """Main loop to handle user interactions."""
    print("ELARA: Booting up... Just a moment!")
    time.sleep(2)

    # Load commands once at the beginning
    commands = load_commands()

    while True:
        try:
            user_input_lines = get_multiline_input()
            if user_input_lines:
                # Process each line separately
                for line in user_input_lines:
                    if not process_command(line.lower(), commands):
                        break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nELARA: ...")
            continue_choice = input("Do you want to continue (y/n)? ").lower()
            if continue_choice != 'y':
                print("ELARA: Goodbye!")
                break

if __name__ == "__main__":
    main()
