import os



def create_directory(complete_path):

    if not os.path.exists(complete_path):

        os.makedirs(complete_path)
