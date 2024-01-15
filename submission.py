#Aqsa Noreen
import os

def get_output_of_file(filename):
    return os.popen(f'python {filename}').read()

if __name__ == "__main__":
    # Get output for nltk_intro.py
    nltk_intro_output = get_output_of_file("nltk_into.py")
    print("The outputs for nltk_into.py:")
    print(nltk_intro_output)

    # Get output for chunking.py
    chunk_output = get_output_of_file("chunking.py")
    print("The output for chunking.py:")
    print(chunk_output)

    # Get output for nb.py
    nb_output = get_output_of_file("nb.py")
    print("The outputs for nb.py:")
    print(nb_output)


    # Get output for chunking.py
    chunk_output = get_output_of_file("chunking.py")
    print("The output for chunking.py:")
    print(chunk_output)

