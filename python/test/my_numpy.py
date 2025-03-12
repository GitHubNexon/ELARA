import numpy as np
import os
import time

def generate_matrix():
    return np.random.randint(1, 10, (3, 3))

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def matrix_game():
    print("Welcome to the Matrix Game!")
    score = 0 

    while True:
        A = generate_matrix()
        B = generate_matrix()

        # Show the matrices
        print("Matrix A:")
        print(A)
        print("\nMatrix B:")
        print(B)

        operation = np.random.choice(['+', '-'])

        if operation == '+':
            result = A + B
            print("\nOperation: A + B")
        else:
            result = A - B
            print("\nOperation: A - B")

        row = np.random.randint(0, 3)
        col = np.random.randint(0, 3)
        print(f"\nQuestion: What is the value of {row+1}{col+1}?")

        correct_answer = result[row, col]

        user_answer = input("Your answer (or type 'exit' or 'quit' to stop): ")

        if user_answer.lower() in ['exit', 'quit']:
            print(f"Thanks for playing! Your final score is {score}.")
            break

        try:
            user_answer = int(user_answer)
            if user_answer == correct_answer:
                score += 1 
                print("Correct! Well done.")
            else:
                print(f"Oops! The correct answer is {correct_answer}.")
        except ValueError:
            print("Please enter a valid integer.")
        
        time.sleep(3)  
        clear_screen()
        print(f"Scoreboard: Your current score is {score}\n")

if __name__ == "__main__":
    matrix_game()
