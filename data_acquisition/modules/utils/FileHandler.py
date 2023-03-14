import os
import time


def create_csv_file(path: str = os.path.realpath(""), folder_name: str = "EEG_Records", prefix: str = "") -> str:
    """
    Creates a CSV file for recording EEG data with a unique file name based on the current date and time.

    :param path: The path of the folder that contain the records folder.

    :param folder_name: The records folder name.

    :param prefix: The prefix that will be added at the beginning of the CSV file name.

    :return: The full path of the CSV file.
    """

    # Create a folder for records.
    folder_full_path = path + '/' + folder_name
    if not os.path.exists(folder_full_path):
        try:
            os.mkdir(folder_full_path)
        except Exception as msg:
            print("Failed to Create Directory: '" + folder_full_path + "' \r\n Please Check Permissions. ")
            print(str(msg))
            return ""

    # Add timestamp to the name of recorded file.
    csv_file_name = prefix + "_" if prefix != "" else ""
    csv_file_name += str(time.strftime("%d.%m.%y_%H.%M.%S"))

    # Create a CSV record file.
    file_path = folder_full_path + "/" + csv_file_name + '.csv'

    return file_path
