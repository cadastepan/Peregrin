import os

class FilesAndFolders:

    def __init__(self, folder):
        self.folder = folder

    def clean(folder):
        for filename in os.listdir(folder): # Czcle for deletion of files in selected folder
            file_path = os.path.join(folder, filename)
            try:
                # Check if it's a file - if so, delete it;
                if os.path.isfile(file_path) or os.path.islink(file_path): 
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                # Ceck if it's a directory - if so, skip it;
                elif os.path.isdir(file_path):
                    print(f"Skipped directory: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        print("All files have been deleted.")


class PlotParams:

    @staticmethod
    def x_span(df, index=0.25):
        return len(df) * index
