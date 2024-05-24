import os


# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
def get_SOURCE_DIRECTORY(proj_name):
   return os.path.join(ROOT_DIRECTORY, "SOURCE_DOCUMENTS", proj_name)

def get_PERSIST_DIRECTORY(proj_name):
   return os.path.join(ROOT_DIRECTORY, proj_name, "DB")

