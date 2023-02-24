import os
import sys


class Main():
    def __init__(self):
        # Add packages folder to sys.path
        dir_path = os.path.dirname(os.path.realpath(__file__))
        packages_path = os.path.join(dir_path, "packages")
        sys.path.append(packages_path)


if __name__ == "__main__":
    Main()

    from modules.headset.EEG import EEG

    epoc_headset = EEG()
    epoc_headset.get_headset_info()

    epoc_headset.record_csv()

    # while 1:
    #     while epoc_headset.is_tasks_empty():
    #         pass
    #     print(epoc_headset.get_data())


