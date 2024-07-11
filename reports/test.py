# Specify the file path and the position you want to check
file_path = "README.md"
position = 14393

try:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    if position < len(text):
        print(f"The character at position {position} is: '{text[position]}'")
    else:
        print(f"The position {position} is out of the file's length which is {len(text)} characters.")
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
