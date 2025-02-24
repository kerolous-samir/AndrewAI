from src.andrew_ai import Andrew_Ai

def main():
    ai = Andrew_Ai()
    user_prompt = input("Enter your prompt: ")
    response = ai.chat(user_prompt)
    print(response)

if __name__ == "__main__":
    main()