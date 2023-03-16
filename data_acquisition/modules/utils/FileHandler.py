import os
import time


def create_csv_file(prefix: str, file_name: str, suffix: str = "", path: str = os.path.realpath(""),
                    folder_name: str = "EEG_Records") -> str:
    """
    Creates a CSV file for recording EEG data with a unique file name based on the current date and time.

    :param prefix: The prefix that will be added at the beginning of the CSV file name.

    :param file_name: The name of the csv file.

    :param suffix: The suffix that will be added at the end of the CSV file name.

    :param path: The path of the folder that contain the records folder.

    :param folder_name: The records folder name.

    :return: The full path of the CSV file.
    """

    # Create a folder for records.
    folder_full_path = path + '/' + folder_name
    create_folder(folder_full_path)

    # Add timestamp to the name of recorded file.
    prefix = f"{prefix}_" if prefix != "" else ""
    suffix = f"_{suffix}" if suffix != "" else ""

    csv_file_name = f"{prefix}{file_name}{suffix}"

    # Create a CSV record file.
    file_path = folder_full_path + "/" + csv_file_name + '.csv'

    return file_path


def create_folder(folder_full_path):
    if not os.path.exists(folder_full_path):
        try:
            os.mkdir(folder_full_path)
        except Exception as msg:
            print("Failed to Create Directory: '" + folder_full_path + "' \r\n Please Check Permissions. ")
            print(str(msg))
            return ""
